"""Record the NinaPro DB5 learning curves: every configuration of configs/db5_space.yaml, every subject,
both learning-rate schedules, T = 50 epochs.

    python scripts/db5_train.py --raw data/raw/db5 --out db5_curves_new.csv.gz [--subjects 1 2]

The raw data (Zenodo record 1000116, s1.zip .. s10.zip, checked against Zenodo's MD5) are downloaded
into --raw.  Their licence, CC BY-ND 4.0, does not allow distributing the signals or anything derived
from them; cite Pizzolato et al., PLoS ONE 12(10): e0186132, 2017.  On the CPU the script reproduces
the curves in data/ (torch.manual_seed(777) before every training); a GPU gives slightly different numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import math
import pathlib
import sys
import urllib.request
import zipfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from save.data import db5

URL = "https://zenodo.org/records/1000116/files/s{k}.zip?download=1"
MD5 = {1: "bb63c22179b4a750bac2499aecd72021", 2: "9d91c5a524123e950cd7cab3b42887ae", 3: "700d15737cd5219cf9e9a0370670f246",
       4: "0eac484016793b28481d786a0f73e270", 5: "0ff40dd987a67f5b2770671504fc9b23", 6: "850e853dd6dfd0170fa38189509133d1",
       7: "e55ee99af5fb924a6a8d8500752b6d78", 8: "281c62eade1ffd58579b57aa36df579d", 9: "622e562482f5b9c9b10d6cfe7b3d1636",
       10: "765d22dbe4241c67f3b6dbacf79fc901"}
OFFSET = {1: 0, 2: 12, 3: 29}
N_CLASSES, CHANNELS, WINDOW, STEP = 53, 16, 40, 10
SPLITS = {"train": (1, 3, 4), "val": (6,), "test": (2, 5)}
SEED = 777
FEATURE_THRESHOLD = 0.05


def fetch(k, raw):
    path = raw / f"s{k}.zip"
    if not path.exists():
        raw.mkdir(parents=True, exist_ok=True)
        print(f"downloading s{k}.zip from Zenodo", flush=True)
        with urllib.request.urlopen(URL.format(k=k), timeout=120) as r, open(path.with_suffix(".part"), "wb") as f:
            f.writelines(iter(lambda: r.read(1 << 20), b""))
        path.with_suffix(".part").replace(path)
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if md5 != MD5[k]:
        raise SystemExit(f"{path}: md5 {md5} differs from Zenodo's {MD5[k]}; delete it and run again")
    return path


def fill_rest_repetitions(label, rep):
    rep = rep.astype(np.int64).copy()
    known = rep > 0
    if not known.any():
        return rep
    last = np.maximum.accumulate(np.where(known, np.arange(len(rep)), -1))
    first = np.flatnonzero(known)[0]
    return np.where(known, rep, np.where(last >= 0, rep[np.maximum(last, 0)], rep[first]))


def read_subject(archive, k):
    """emg (samples, 16) of the three exercises and the segment table (start, end, label, repetition)"""
    import scipy.io
    emg, table, start = [], [], 0
    with zipfile.ZipFile(archive) as z:
        for ex in (1, 2, 3):
            m = scipy.io.loadmat(io.BytesIO(z.read(f"s{k}/S{k}_E{ex}_A1.mat")))
            stim = m["restimulus"].ravel().astype(np.int64)
            label = np.where(stim > 0, stim + OFFSET[ex], 0)
            rep = fill_rest_repetitions(label, m["rerepetition"].ravel())
            change = np.flatnonzero((np.diff(label) != 0) | (np.diff(rep) != 0)) + 1
            for b, e in zip(np.r_[0, change], np.r_[change, len(label)]):
                table.append((start + b, start + e, int(label[b]), int(rep[b])))
            emg.append(m["emg"].astype(np.float64))
            start += len(label)
    return np.concatenate(emg), np.array(table, np.int64)


def windows(table, rng):
    out = {}
    for split, reps in sorted(SPLITS.items()):
        starts, labels = [], []
        for b, e, label, rep in table:
            if rep in reps and e - b >= WINDOW:
                w = np.arange(b, e - WINDOW + 1, STEP)
                starts.append(w)
                labels.append(np.full(len(w), label))
        starts, labels = np.concatenate(starts), np.concatenate(labels)
        keep_rest = int(round(np.bincount(labels, minlength=N_CLASSES)[1:].mean()))
        rest = np.flatnonzero(labels == 0)
        drop = np.setdiff1d(rest, rng.choice(rest, size=min(keep_rest, len(rest)), replace=False))
        keep = np.setdiff1d(np.arange(len(starts)), drop)
        out[split] = (starts[keep], labels[keep])
    return out


def features(X):
    """windows (n, 40, 16) -> (n, 80): MAV, RMS, WL, ZC, SSC per channel"""
    d = np.diff(X, axis=1)
    mav, rms = np.abs(X).mean(axis=1), np.sqrt((X ** 2).mean(axis=1))
    wl = np.abs(d).sum(axis=1)
    zc = ((X[:, :-1] * X[:, 1:] < 0) & (np.abs(d) >= FEATURE_THRESHOLD)).sum(axis=1)
    ssc = ((d[:, :-1] * -d[:, 1:]) >= FEATURE_THRESHOLD ** 2).sum(axis=1)
    return np.concatenate([mav, rms, wl, zc, ssc], axis=1).astype(np.float32)


def subject_data(raw, k):
    emg, table = read_subject(fetch(k, raw), k)
    split = windows(table, np.random.default_rng([SEED, k]))
    covered = np.zeros(len(emg), bool)
    for off in range(WINDOW):
        covered[split["train"][0] + off] = True
    mean, sd = emg[covered].mean(axis=0), emg[covered].std(axis=0)
    emg = ((emg - mean) / np.where(sd > 0, sd, 1.0)).astype(np.float32)
    out = {}
    for name, (starts, labels) in split.items():
        X = emg[starts[:, None] + np.arange(WINDOW)]                       # (n, 40, 16)
        out["X_" + name] = np.ascontiguousarray(X.transpose(0, 2, 1))
        out["F_" + name] = features(X)
        out["y_" + name] = labels.astype(np.int64)
    m, s = out["F_train"].mean(axis=0), out["F_train"].std(axis=0)
    for name in SPLITS:
        out["F_" + name] = (out["F_" + name] - m) / np.where(s > 0, s, 1.0)
    return out


def build(c):
    from torch import nn
    if c["family"] == "cnn":
        layers, ch, length = [], CHANNELS, WINDOW
        for _ in range(c["conv_layers"]):
            layers += [nn.Conv1d(ch, c["channels"], c["kernel"], padding=c["kernel"] // 2),
                       nn.BatchNorm1d(c["channels"]), nn.ReLU(), nn.MaxPool1d(2)]
            ch, length = c["channels"], length // 2
        layers += [nn.Flatten(), nn.Linear(ch * length, c["fc"]), nn.ReLU(), nn.Dropout(c["dropout"]),
                   nn.Linear(c["fc"], N_CLASSES)]
        return nn.Sequential(*layers)
    layers, width = [], 5 * CHANNELS
    for _ in range(c["layers"]):
        layers += [nn.Linear(width, c["width"]), nn.ReLU(), nn.Dropout(c["dropout"])]
        width = c["width"]
    return nn.Sequential(*(layers + [nn.Linear(width, N_CLASSES)]))


def learning_rate(schedule, lr, t, T):
    return lr if schedule == "constant" else 0.5 * lr * (1.0 + math.cos(math.pi * (t - 1) / T))


def train(data, c, schedule, T, device, epochs=None):
    """[(val accuracy, test accuracy, mean training loss) for epochs 1..T (or the first `epochs`)]"""
    import torch
    torch.manual_seed(SEED)
    key = "X_" if c["family"] == "cnn" else "F_"
    model = build(c).to(device)
    parts = {s: (torch.from_numpy(data[key + s]).to(device), torch.from_numpy(data["y_" + s]).to(device)) for s in SPLITS}
    opt = torch.optim.SGD(model.parameters(), lr=c["lr"], momentum=0.9, nesterov=True, weight_decay=c["weight_decay"])
    X, y = parts["train"]
    out = []
    for t in range(1, (epochs or T) + 1):
        for g in opt.param_groups:
            g["lr"] = learning_rate(schedule, c["lr"], t, T)
        perm = torch.randperm(len(y), device=device)
        total = 0.0
        for a in range(0, len(y), c["batch"]):
            b = perm[a:a + c["batch"]]
            if len(b) < 2:                                   # batch norm needs two samples
                continue
            loss = torch.nn.functional.cross_entropy(model(X[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        acc = {}
        with torch.no_grad():
            for s in ("val", "test"):
                Xs, ys = parts[s]
                hit = sum((model(Xs[i:i + 2048]).argmax(1) == ys[i:i + 2048]).sum().item() for i in range(0, len(ys), 2048))
                acc[s] = hit / float(len(ys))
        model.train()
        out.append((acc["val"], acc["test"], total / len(y)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default="data/raw/db5")
    ap.add_argument("--out", default="db5_curves_new.csv.gz")
    ap.add_argument("--subjects", nargs="*", type=int, default=list(range(1, 11)))
    ap.add_argument("--schedules", nargs="*", default=list(db5.SCHEDULES))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--configs", nargs="*", type=int, help="only these configuration indices")
    ap.add_argument("--epochs", type=int, help="stop after this many epochs (the schedule still spans T)")
    a = ap.parse_args()
    raw = pathlib.Path(a.raw)
    configs = db5.configs()
    rows = ["recipe,subject,config,epoch,val_accuracy,test_accuracy,train_loss"]
    for k in a.subjects:
        data = subject_data(raw, k)
        print(f"s{k:02d}: train {len(data['y_train'])}, val {len(data['y_val'])}, test {len(data['y_test'])} windows",
              flush=True)
        for sched in a.schedules:
            for c in configs if a.configs is None else [configs[i] for i in a.configs]:
                for t, (v, te, loss) in enumerate(train(data, c, sched, db5.T, a.device, a.epochs), start=1):
                    rows.append(f"{sched},s{k:02d},{c['index']},{t},{v!r},{te!r},{loss!r}")
                print(f"  {sched} config {c['index']}: final val {v:.4f}", flush=True)
    import gzip
    with gzip.open(a.out, "wt", newline="\n") as f:
        f.write("\n".join(rows) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

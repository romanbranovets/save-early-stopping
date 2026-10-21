"""LCBench (Zimmer et al. 2021): 35 OpenML datasets x 2000 MLP configurations, validation accuracy after
every epoch, read from data_2k_lw.json.  The search sees epochs 1..50 (the first logged value is the
evaluation before training); accuracies pass through float32 once, as in the paper's runs.
Configurations 0-1199 are the development part and 1200-1999 the test part."""
from __future__ import annotations

import json

import numpy as np

from . import DATA

T = 50
SPLITS = {"dev": (0, 1200), "test": (1200, 2000), "all": (0, 2000)}
# LCBench search space: name -> (low, high, log scale), used to scale hyperparameters to [0, 1]
SPACE = {"batch_size": (16, 512, True), "learning_rate": (1e-4, 1e-1, True), "momentum": (0.1, 0.99, False),
         "weight_decay": (1e-5, 0.1, False), "num_layers": (1, 5, False), "max_units": (64, 1024, True),
         "max_dropout": (0.0, 1.0, False)}


def _top_level(text):
    """(dataset, raw json) of every top-level key, without parsing the whole file at once."""
    i = text.index("{") + 1
    n = len(text)
    while i < n:
        while i < n and text[i] in ", \n\r\t":
            i += 1
        if i >= n or text[i] == "}":
            return
        j = text.index('"', i + 1)
        key = text[i + 1:j]
        k = text.index(":", j) + 1
        while text[k] in " \n\r\t":
            k += 1
        depth, p, in_str, esc = 0, k, False, False
        while p < n:
            ch = text[p]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    p += 1
                    break
            p += 1
        yield key, text[k:p]
        i = p


def _hp(cfg):
    out = []
    for k, (lo, hi, lg) in SPACE.items():
        try:
            v = float(cfg.get(k, np.nan))
        except (TypeError, ValueError):
            v = np.nan
        v = (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) if lg else (v - lo) / (hi - lo)
        out.append(float(np.clip(np.nan_to_num(v, nan=0.5), 0.0, 1.0)))
    return out


def prepare(json_path):
    """data_2k_lw.json -> data/lcbench/<dataset>.npz: acc (2000, 50), err52 (2000, 52) = 1 - acc/100 of
    all logged values, lr52 (2000, 52), hp (2000, 7)."""
    out = DATA / "lcbench"
    out.mkdir(parents=True, exist_ok=True)
    text = open(json_path, encoding="utf-8").read()
    for name, blob in _top_level(text):
        obj = json.loads(blob)
        ids = sorted(obj, key=int)
        assert [int(c) for c in ids] == list(range(2000)), name
        err = np.stack([1.0 - np.asarray(obj[c]["log"]["Train/val_accuracy"], float) / 100.0 for c in ids])
        lr = np.stack([np.asarray(obj[c]["log"]["Train/lr"], float) for c in ids])
        hp = np.array([_hp(obj[c]["config"]) for c in ids], np.float32)
        acc = (1.0 - err[:, 1:T + 1]).astype(np.float32).astype(np.float64)
        np.savez_compressed(out / f"{name}.npz", acc=acc, err52=err, lr52=lr, hp=hp)


def datasets():
    return sorted(p.stem for p in (DATA / "lcbench").glob("*.npz"))


def load(name, split, what="acc"):
    lo, hi = SPLITS[split]
    return np.load(DATA / "lcbench" / f"{name}.npz")[what][lo:hi]


def quantum(acc):
    """smallest gap between recorded accuracies of the population"""
    v = np.unique(np.round(np.asarray(acc, np.float64), 9))
    d = np.diff(v)
    return float(d[d > 1e-7].min())


def error_quantum(name):
    """the quantum the parametric forecast uses: over the errors of all 2000 configurations"""
    e = np.load(DATA / "lcbench" / f"{name}.npz")["err52"][:, 1:]
    ys = np.unique(np.round(e, 9))
    q = np.diff(ys)
    return float(q[q > 1e-7].min())

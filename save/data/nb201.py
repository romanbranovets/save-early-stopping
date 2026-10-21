"""NAS-Bench-201 from the NATS-Bench topology records (NATS-tss-v1_0-3ffb9-simple): seed 777, every
second of 200 epochs (T = 100 grid steps).  cifar10-valid accuracies are rounded to 1/25000, the
resolution of its 25k evaluation images; the stored values carry float32 noise."""
from __future__ import annotations

import bz2
import pathlib
import pickle
from multiprocessing import Pool

import numpy as np

from . import DATA, REPO

TASKS = ("cifar10-valid", "cifar100", "ImageNet16-120")
EVAL_SET = {"cifar10-valid": "x-valid", "cifar100": "ori-test", "ImageNet16-120": "ori-test"}
N_ARCH, EPOCHS, STRIDE, SEED = 15625, 200, 2, 777
T = EPOCHS // STRIDE
ROUND_TO = {"cifar10-valid": 25000}
# smallest step of the recorded accuracy (1 / evaluation images), as measured on the records
QUANTUM = {"cifar10-valid": 3.999999999992898e-05, "cifar100": 9.999999999998899e-05,
           "ImageNet16-120": 0.000166665999999982}
OPS = ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")
EDGES = ("edge_0_1", "edge_0_2", "edge_1_2", "edge_0_3", "edge_1_3", "edge_2_3")


def _read(args):
    path, idx = args
    with bz2.open(pathlib.Path(path) / f"{idx:06d}.pickle.pbz2", "rb") as f:
        d = pickle.load(f)[str(EPOCHS)]
    out = {}
    for (ds, seed), r in d["all_results"].items():
        if ds in EVAL_SET and int(seed) == SEED:
            es = EVAL_SET[ds]
            out[ds] = np.array([r["eval_acc1es"].get(f"{es}@{e}", np.nan) for e in range(EPOCHS)], float)
    return idx, d["arch_str"], out


def prepare(records_dir, workers=8):
    """records -> data/nb201_<task>.npz with acc (15625, 100) and the architecture strings."""
    with Pool(workers) as pool:
        res = sorted(pool.map(_read, [(str(records_dir), i) for i in range(N_ARCH)], chunksize=64), key=lambda r: r[0])
    arch = np.array([r[1] for r in res])
    cols = np.arange(STRIDE, EPOCHS + 1, STRIDE) - 1
    for task in TASKS:
        a = np.stack([r[2][task] for r in res])[:, cols] / 100.0
        assert np.isfinite(a).all() and a.shape == (N_ARCH, T)
        if task in ROUND_TO:
            a = np.round(a * ROUND_TO[task]) / ROUND_TO[task]
        np.savez_compressed(DATA / f"nb201_{task}.npz", acc=a, arch=arch)


def split_index(task, split):
    """indices into the 15 625 architectures: split = dev (1000), prelim (2000) or test (12 625)"""
    return np.loadtxt(REPO / "splits" / "nb201" / f"{task}_{split}.txt", dtype=np.int64)


def load(task, split):
    return np.load(DATA / f"nb201_{task}.npz")["acc"][split_index(task, split)]


def arch_strings(task="cifar10-valid"):
    return [str(s) for s in np.load(DATA / f"nb201_{task}.npz")["arch"]]


def hyperparameters(task, split):
    """(n, 6) edge operations encoded as i / 4 in the order of OPS (FT-PFN inputs)."""
    arch = arch_strings(task)
    code = {op: i / (len(OPS) - 1) for i, op in enumerate(OPS)}
    out = []
    for g in split_index(task, split):
        ops = {}
        for j, node in enumerate(arch[g].split("+"), start=1):
            for tok in node.strip("|").split("|"):
                op, i = tok.split("~")
                ops[f"edge_{int(i)}_{j}"] = op
        out.append([code[ops[e]] for e in EDGES])
    return np.asarray(out, np.float32)


def lr_scale():
    """lr_t / lr_1 of the cosine schedule (lr 0.1 to 0 over 200 epochs, per iteration), epoch means, on the grid."""
    u = np.arange(EPOCHS)[:, None] + (np.arange(100)[None, :] + 0.5) / 100
    lr = (0.1 * 0.5 * (1.0 + np.cos(np.pi * u / EPOCHS))).mean(axis=1)[np.arange(STRIDE, EPOCHS + 1, STRIDE) - 1]
    return lr / lr[0]

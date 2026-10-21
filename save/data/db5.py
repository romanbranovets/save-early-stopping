"""NinaPro DB5 learning curves recorded for the paper (data/db5_curves.csv.gz): 10 subjects x 2
learning-rate schedules, 60 configurations each, T = 50.  Cite Pizzolato et al. (2017) when you use them."""
from __future__ import annotations

import functools
import math
import re

import numpy as np
import pandas as pd

from . import REPO

SUBJECTS = [f"s{i:02d}" for i in range(1, 11)]
SCHEDULES = ("cosine", "constant")
T, N_CONFIGS = 50, 60


@functools.cache
def _curves():
    return pd.read_csv(REPO / "data" / "db5_curves.csv.gz", float_precision="round_trip")


@functools.cache
def _subjects():
    return pd.read_csv(REPO / "data" / "db5_subjects.csv").set_index("subject")


def load(schedule, subject, metric="val_accuracy"):
    d = _curves()
    d = d[(d.recipe == schedule) & (d.subject == subject)]
    a = d.pivot(index="config", columns="epoch", values=metric).sort_index().to_numpy(np.float64)
    assert a.shape == (N_CONFIGS, T)
    return a


def quantum(subject):
    return 1.0 / int(_subjects().loc[subject, "n_val"])


def orders(schedule, subject, n=40, seed=777):
    """The orders of the recorded searches (ordering 0 is the one of the live search)."""
    k = int(re.search(r"(\d+)$", subject).group(1))
    rng = np.random.default_rng([int(seed), k, SCHEDULES.index(schedule)])
    return np.array([rng.permutation(N_CONFIGS) for _ in range(n)])


def lr_scale(schedule):
    """lr_t / lr_1 for t = 1..T + 1 (cosine to 0 over T epochs, or constant)."""
    if schedule == "constant":
        return np.ones(T + 1)
    return np.array([0.5 * (1.0 + math.cos(math.pi * (t - 1) / T)) for t in range(1, T + 2)])


def configs():
    import yaml
    return yaml.safe_load((REPO / "configs" / "db5_space.yaml").read_text())["configs"]


def hyperparameters():
    """(60, 7) in [0, 1]: family, depth, width or channels, lr, weight decay, dropout, batch (FT-PFN inputs)."""
    def lg(x, lo, hi):
        return (np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo))
    out = []
    for c in configs():
        cnn = c["family"] == "cnn"
        n = c["conv_layers"] if cnn else c["layers"]
        w = c["channels"] if cnn else c["width"]
        out.append([float(cnn), (n - 1) / 2, lg(w, 16, 256), lg(c["lr"], 1e-3, 3e-1), lg(c["weight_decay"], 1e-5, 1e-2),
                    c["dropout"] / 0.5, lg(c["batch"], 64, 256)])
    return np.clip(np.asarray(out, np.float32), 0.0, 1.0)

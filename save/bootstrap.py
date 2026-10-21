"""Percentile bootstrap over independent units (orders within a task, datasets, subjects)."""
from __future__ import annotations

import numpy as np


def unit_weights(seed, D, n=2000):
    """(n, D) weights of a bootstrap over D units; each row is a resample's counts divided by D."""
    rng = np.random.default_rng(seed)
    reps = rng.integers(0, D, size=(n, D))
    return np.stack([np.bincount(r, minlength=D) / D for r in reps])


def ci(x, level=95):
    a = (100 - level) / 2
    return tuple(float(v) for v in np.nanpercentile(np.asarray(x, float), [a, 100 - a]))

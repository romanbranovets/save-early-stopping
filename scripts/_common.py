"""Helpers shared by the scripts: result files and setting overrides."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def parse_set(items):
    """["price=0.001", "paths=100"] -> {"price": 0.001, "paths": 100}"""
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = None if v == "none" else int(v) if v.lstrip("-").isdigit() else float(v)
    return out


def label(method, overrides):
    if not overrides:
        return method
    return method + "@" + ",".join(f"{k}={v!r}" for k, v in sorted(overrides.items()))


def frame_tag(pool_size, final_phase):
    return ("main" if pool_size is None else f"pool{pool_size}") + ("-final" if final_phase else "")


def result_path(out, bench, split, tag, lab, unit):
    return pathlib.Path(out) / bench / split / tag / lab / f"{unit.replace('/', '_')}.npz"


def has_result(out, bench, split, tag, lab, unit, n_orders):
    """a result file of this label exists and covers exactly n_orders orders (or pools)"""
    f = result_path(out, bench, split, tag, lab, unit)
    return f.exists() and len(np.load(f)["order_ids"]) == n_orders


def load_result(out, bench, split, tag, lab, unit):
    f = result_path(out, bench, split, tag, lab, unit)
    if not f.exists():
        raise FileNotFoundError(f"{f}: run scripts/run.py first")
    return dict(np.load(f))


def mean_regret(r):
    """(budget,) regret averaged over the orders of one unit"""
    return (r["best_acc"][:, None] - r["best_e"]).mean(axis=0)

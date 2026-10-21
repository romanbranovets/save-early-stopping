"""Units, populations, orders, pools, settings and bootstrap weights of the benchmarks.  A unit is a
NAS-Bench-201 task, an LCBench dataset, or a DB5 subject under one schedule ("cosine/s01")."""
from __future__ import annotations

import copy
import functools

import numpy as np
import yaml

from . import seeds
from .bootstrap import unit_weights
from .data import REPO, db5, lcbench, nb201

BENCHMARKS = ("nb201", "lcbench", "db5")


@functools.cache
def config(bench):
    return yaml.safe_load((REPO / "configs" / f"{bench}.yaml").read_text())


def frame(bench):
    c = config(bench)
    return dict(T=c["T"], budget=c["budget"], slots=c["slots"], t_min=c["t_min"])


def settings(bench, unit):
    """{method: parameters} for one unit (LCBench: the default updated with the dataset's own picks)."""
    c = config(bench)
    if bench != "lcbench":
        return copy.deepcopy(c["methods"])
    out = copy.deepcopy(c["default"]["methods"])
    for m, p in (c.get("per_dataset") or {}).get(unit, {}).items():
        out[m].update(p)
    return out


def price_sensitivity(bench):
    c = config(bench)
    return (c["default"] if bench == "lcbench" else c)["price_sensitivity"]


def units(bench):
    if bench == "nb201":
        return list(nb201.TASKS)
    if bench == "lcbench":
        return lcbench.datasets()
    return [f"{s}/{p}" for s in db5.SCHEDULES for p in db5.SUBJECTS]


def test_sets(bench):
    """{set name: units}"""
    if bench == "nb201":
        return {t: [t] for t in nb201.TASKS}
    if bench == "lcbench":
        return {"lcbench": units(bench)}
    return {s: [f"{s}/{p}" for p in db5.SUBJECTS] for s in db5.SCHEDULES}


def population(bench, unit, split="test"):
    """(N, T) accuracy curves of the configurations available to the search."""
    if bench == "nb201":
        return nb201.load(unit, split)
    if bench == "lcbench":
        return lcbench.load(unit, split)
    return db5.load(*unit.split("/"))


def floor(bench, unit, acc):
    """lower bound of the SAVE path noise: half the recording quantum"""
    if bench == "nb201":
        return 0.5 * nb201.QUANTUM[unit]
    if bench == "lcbench":
        return 0.5 * lcbench.quantum(acc)
    return 0.5 * db5.quantum(unit.split("/")[1])


def orders(bench, unit, split="test", n=40):
    """(n, N) random orders of the population (indices into it)."""
    if bench == "db5":
        return db5.orders(*unit.split("/"), n=n)
    N = len(population(bench, unit, split))
    rng = np.random.default_rng(seeds.orders_seed(bench, split, unit))
    return np.array([rng.permutation(N) for _ in range(n)])


def pools(bench, unit, P, split="test", n=40):
    """(n, P) random pools of P configurations; the pool in its drawn order is the order of the search."""
    N = len(population(bench, unit, split))
    return np.stack([np.random.default_rng(seeds.pool_seed(bench, unit, P, s)).permutation(N)[:P]
                     for s in range(n)]).astype(np.int64)


def bootstrap_weights(bench, set_name, split="test", n_units=None, n=2000):
    if bench == "nb201":
        return unit_weights(seeds.bootstrap_seed("nb201", split, set_name), n_units or 40, n)
    if bench == "lcbench":
        return unit_weights(seeds.bootstrap_seed("lcbench"), n_units or 35, n)
    return unit_weights(seeds.bootstrap_seed("db5"), n_units or 10, n)


def pool_bootstrap_weights(bench, unit, P, n_pools=40, n=2000):
    return unit_weights(seeds.pool_bootstrap_seed(bench, unit, P), n_pools, n)


def hyperparameters(bench, unit, split="test"):
    if bench == "nb201":
        return nb201.hyperparameters(unit, split)
    if bench == "lcbench":
        return lcbench.load(unit, split, "hp")
    return db5.hyperparameters()

"""One method on one unit and a set of orders (or pools): stop epochs and incumbent trajectories."""
from __future__ import annotations

import pathlib

import numpy as np

from . import benchmarks as bm
from .baselines.cl2017 import ChandrashekaranLane
from .baselines.rank import ASHA, FixedBudget, Hyperband, MedianStopping, NoStop, successive_halving
from .baselines.tables import TablePolicy, load_table
from .method import SAVE, PerfectForecast, cached_banks
from .oracles import BestOnAverage, default_horizons, envelope, nostop_target
from .search import run_search

METHODS = ("no-stop", "save", "save-threshold", "save-completed-donors", "save-multiplicative-uniform",
           "median", "successive-halving", "hyperband", "asha", "cl2017", "cl2017-value",
           "lcpfn-threshold", "lcpfn-value", "parametric-threshold", "parametric-value", "fixed-budget", "fixed-budget-tmin",
           "ftpfn-threshold", "ftpfn-value",
           "oracle-perfect-forecast", "oracle-best-on-average", "oracle-all-knowing")
TABLE_METHODS = {"lcpfn-threshold": ("lcpfn", "delta"), "lcpfn-value": ("lcpfn", "price"),
                 "parametric-threshold": ("parametric", "delta"), "parametric-value": ("parametric", "price")}


def table_path(table_dir, bench, unit, split, source, kind, value):
    return pathlib.Path(table_dir) / bench / split / f"{unit.replace('/', '_')}__{source}__{kind}{value!r}.npz"


def make_policy(method, p, acc, T, fl, ctx):
    """ctx: bench, unit, split, table_dir, ftpfn model and hyperparameters."""
    if method == "no-stop":
        return NoStop()
    if method in ("fixed-budget", "fixed-budget-tmin"):
        return FixedBudget(p["epochs"])
    if method == "median":
        return MedianStopping(p["grace"], p["patience"])
    if method == "asha":
        return ASHA(p["r"], p["eta"])
    if method == "hyperband":
        return Hyperband(p["r"], p["eta"], p.get("max_resource", T))
    if method == "save":
        return SAVE(p["price"] / T, fl, M=p.get("paths", 400))
    if method == "save-threshold":
        return SAVE(0.0, fl, M=p.get("paths", 400), rule="threshold", delta=p["delta"])
    if method == "save-completed-donors":
        return SAVE(p["price"] / T, fl, M=p.get("paths", 400), completed_only=True)
    if method == "save-multiplicative-uniform":
        return SAVE(p["price"] / T, fl, M=p.get("paths", 400), continuation="mult", weighted=False)
    if method == "cl2017":
        return ChandrashekaranLane(delta=p["delta"])
    if method == "cl2017-value":
        return ChandrashekaranLane(rule="value", price=p["price"] / T, Z=cached_banks(400, T)[0][:, 0])
    if method == "oracle-perfect-forecast":
        return PerfectForecast(p["price"] / T)
    if method in TABLE_METHODS:
        source, kind = TABLE_METHODS[method]
        value = p[kind] / T if kind == "price" else p[kind]
        f = table_path(ctx["table_dir"], ctx["bench"], ctx["unit"], ctx["split"], source, kind, float(value))
        if not f.exists():
            raise FileNotFoundError(f"{f}: make it with scripts/tables.py")
        return TablePolicy(load_table(f))
    if method in ("ftpfn-threshold", "ftpfn-value"):
        from .baselines.ftpfn import FTPFNPolicy
        if method == "ftpfn-threshold":
            return FTPFNPolicy(ctx["ftpfn"], ctx["hp"], rule="threshold", delta=p["delta"])
        return FTPFNPolicy(ctx["ftpfn"], ctx["hp"], rule="value", price=p["price"] / T)
    raise KeyError(method)


def run(bench, unit, method, params, split="test", n_orders=40, pool_size=None, final_phase=False,
        table_dir="tables", order_ids=None, ftpfn=None, horizon=None):
    """Returns dict(K (S, slots), best_e (S, budget), used (S,), best_acc (S,), orders (S, slots))."""
    F = bm.frame(bench)
    T, B, t_min = F["T"], F["budget"], F["t_min"]
    pop = bm.population(bench, unit, split)
    fl = bm.floor(bench, unit, pop)
    if pool_size is None:
        O = bm.orders(bench, unit, split, n_orders)
        slots = F["slots"]
    else:
        O = bm.pools(bench, unit, pool_size, split, n_orders)
        slots = pool_size
    ids = range(len(O)) if order_ids is None else order_ids
    ctx = dict(bench=bench, unit=unit, split=split, table_dir=table_dir, ftpfn=ftpfn,
               hp=bm.hyperparameters(bench, unit, split) if method.startswith("ftpfn") else None)
    dp = None
    if method == "oracle-best-on-average" and pool_size is None:
        dp = BestOnAverage(pop, B, t_min)
        if horizon is None:
            ns = [run_search(pop, o, NoStop(), B, slots, t_min).best_e[-1] for o in O]
            horizon = dp.horizon(float(pop.max()) - float(np.mean(ns)), default_horizons(T), float(pop.max()))
    Ks, bes, used, refs, kept = [], [], [], [], []
    for i in ids:
        o = O[i]
        if pool_size is None:
            acc, order, ref = pop, o, float(pop.max())
        else:                                   # a pool is the population of its own search
            acc, order, ref = pop[o], np.arange(pool_size), float(pop[o].max())
        K, be, u = _one(method, params, acc, order, T, B, slots, t_min, fl, ctx, final_phase, dp, horizon)
        Ks.append(K[:slots] if len(K) >= slots else np.r_[K, np.zeros(slots - len(K), np.int64)])
        bes.append(be)
        used.append(u)
        refs.append(ref)
        kept.append(np.asarray(o[:slots]))
    return dict(K=np.stack(Ks).astype(np.int32), best_e=np.stack(bes), used=np.array(used),
                best_acc=np.array(refs), orders=np.stack(kept).astype(np.int32), order_ids=np.array(list(ids)),
                horizon=np.array(-1 if horizon is None else int(horizon)))


def _one(method, p, acc, order, T, B, slots, t_min, fl, ctx, final_phase, dp, horizon):
    if method == "successive-halving":
        be, K, u = successive_halving(acc, order, p["r"], p["eta"], B, t_min, final_phase=final_phase)
        return K, be, u
    if method == "oracle-all-knowing":
        return np.zeros(slots, np.int64), envelope(acc, order, B, t_min), B
    if method == "oracle-best-on-average":
        if dp is None:                                   # a pool: the model is the pool itself
            dp = BestOnAverage(acc, B, t_min)
            horizon = dp.horizon(nostop_target(acc, B), default_horizons(T), float(acc.max()))
        policy = dp.policy(horizon)
    else:
        policy = make_policy(method, p, acc, T, fl, ctx)
    r = run_search(acc, order, policy, B, slots, t_min, final_phase=final_phase)
    return r.K, r.best_e, r.used

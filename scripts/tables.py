"""Decision tables of the forecasts that use only the current run's own curve: LC-PFN and the parametric
forecast.  One file per unit and setting, read by scripts/run.py.

    python scripts/tables.py --benchmark nb201 --source parametric --jobs 16
    python scripts/tables.py --benchmark lcbench --source lcpfn --device cuda
    python scripts/tables.py --benchmark nb201 --source lcpfn --split dev --grid

Without --grid the tables are made for the settings of configs/<benchmark>.yaml (or --set); with
--grid for every value of configs/grids.yaml, as needed by scripts/select_settings.py.  Every setting
is computed from the same forecast pass over the curves.  LC-PFN needs the optional dependencies.
"""
from __future__ import annotations

import argparse
import time

import _common as C
import numpy as np
import yaml
from joblib import Parallel, delayed

from save import benchmarks as bm
from save import seeds
from save.baselines.parametric import tables_for_curve
from save.baselines.tables import save_table
from save.data import db5, lcbench, nb201
from save.runner import table_path

METHODS = {"parametric": ("parametric-threshold", "parametric-value"), "lcpfn": ("lcpfn-threshold", "lcpfn-value")}


def decisions(bench):
    T = bm.frame(bench)["T"]
    # on NAS-Bench-201 the forecast is also drawn at t = 1, which fixes the random numbers of later t
    return range(1, T) if bench == "nb201" else range(2, T)


def parametric_inputs(bench, unit, split):
    """per curve: error series, accuracy seen by the search, lr scale, quantum, seeds, path length"""
    acc = bm.population(bench, unit, split)
    if bench == "nb201":
        ids = nb201.split_index(unit, split)
        q, sc = nb201.QUANTUM[unit], nb201.lr_scale()
        return [(1.0 - acc[i], acc[i], sc, q, seeds.parametric_seeds("nb201", unit, int(g)), nb201.T)
                for i, g in enumerate(ids)]
    if bench == "lcbench":
        lo = lcbench.SPLITS[split][0]
        err = lcbench.load(unit, split, "err52")[:, 1:]
        lr = lcbench.load(unit, split, "lr52")[:, 1:]
        q = lcbench.error_quantum(unit)
        return [(err[i], acc[i], lr[i] / lr[i, 0], q, seeds.parametric_seeds("lcbench", unit, lo + i), err.shape[1])
                for i in range(len(acc))]
    sched, subj = unit.split("/")
    sc, q = db5.lr_scale(sched), db5.quantum(subj)
    return [(1.0 - acc[i], acc[i], sc, q, seeds.parametric_seeds("db5", f"{sched}_{subj}", i), db5.T + 1)
            for i in range(len(acc))]


def one_curve(x, T, dec, prices, deltas):
    err, acc, sc, q, sd, T_all = x
    return tables_for_curve(err, acc, sc, q, sd, dec, T, prices=prices, deltas=deltas, T_all=T_all)


def settings(bench, unit, source, grid, over):
    """(deltas, prices per full training) to tabulate"""
    thr_m, val_m = METHODS[source]
    if grid:
        g = yaml.safe_load((C.ROOT / "configs" / "grids.yaml").read_text())
        return list(g[thr_m]["delta"]), list(g[val_m]["price"])
    st = bm.settings(bench, unit)
    deltas = [over.get("delta", st[thr_m]["delta"])] if thr_m in st else []
    prices = [over.get("price", st[val_m]["price"])] if val_m in st else []
    return deltas, prices


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", required=True, choices=bm.BENCHMARKS)
    ap.add_argument("--source", required=True, choices=sorted(METHODS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--units", nargs="*")
    ap.add_argument("--configs", type=int, help="only the first n configurations (for a quick look)")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--set", action="append", help="price=<cT> or delta=<threshold>")
    ap.add_argument("--tables", default="tables")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    bench = a.benchmark
    T = bm.frame(bench)["T"]
    dec = list(decisions(bench))
    over = C.parse_set(a.set)
    net = None
    t0 = time.time()
    for u in a.units or bm.units(bench):
        deltas, cts = settings(bench, u, a.source, a.grid, over)
        prices = [c / T for c in cts]
        files = [table_path(a.tables, bench, u, a.split, a.source, "delta", float(d)) for d in deltas] + \
                [table_path(a.tables, bench, u, a.split, a.source, "price", float(c)) for c in prices]
        if all(f.exists() for f in files):
            continue
        if a.source == "parametric":
            xs = parametric_inputs(bench, u, a.split)[:a.configs]
            res = Parallel(n_jobs=a.jobs)(delayed(one_curve)(x, T, dec, prices, deltas) for x in xs)
            val = np.stack([v for v, _ in res], axis=1)
            thr = np.stack([t for _, t in res], axis=1)
        else:
            from save.baselines import lcpfn
            net = net or lcpfn.load_model(a.device)
            acc = bm.population(bench, u, a.split)[:a.configs]
            thr, val = lcpfn.tables(acc, dec, deltas=deltas, prices=prices, device=a.device, net=net)
        for j, d in enumerate(deltas):
            save_table(files[j], thr[:, :, j], delta=d)
        for j, c in enumerate(prices):
            save_table(files[len(deltas) + j], val[:, :, j], price=c)
        print(f"[{time.time() - t0:7.0f}s] {a.source} {u}: {len(files)} tables", flush=True)


if __name__ == "__main__":
    main()

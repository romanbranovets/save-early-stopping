"""Searches over random pools of P configurations, with and without the final phase: failing units and
speed-ups per pool size, and how often the best configuration of a pool is finally trained to T.

    python scripts/pool_size.py --benchmark lcbench --methods save median asha lcpfn-value --jobs 8
    python scripts/pool_size.py --benchmark nb201 --sizes 60 1000 --methods save median

Missing results are computed (40 pools per unit and size); no-stop is always included.
"""
from __future__ import annotations

import argparse

import _common as C
import numpy as np
import pandas as pd
from evaluate import panels
from joblib import Parallel, delayed

from save import benchmarks as bm
from save.runner import run


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", required=True, choices=["nb201", "lcbench"])
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--sizes", nargs="*", type=int)
    ap.add_argument("--units", nargs="*", help="default: all units")
    ap.add_argument("--pools", type=int, default=40)
    ap.add_argument("--tables", default="tables")
    ap.add_argument("--out", default="results")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--csv")
    a = ap.parse_args()
    bench = a.benchmark
    units = a.units or bm.units(bench)
    sizes = a.sizes or bm.config(bench)["pool_sizes"]
    methods = ["no-stop"] + [m for m in a.methods if m != "no-stop"]
    rows = []
    for P in sizes:
        for fp in (False, True):
            tag = C.frame_tag(P, fp)
            todo = [(m, u) for m in methods for u in units
                    if not C.has_result(a.out, bench, "test", tag, m, u, a.pools)]
            res = Parallel(n_jobs=a.jobs)(delayed(run)(bench, u, m, bm.settings(bench, u)[m], "test", a.pools, P, fp,
                                                       a.tables) for m, u in todo)
            for (m, u), r in zip(todo, res):
                f = C.result_path(a.out, bench, "test", tag, m, u)
                f.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(f, **r)
            Ps = panels(bench, "test", tag, methods, a.out, units)
            for m in methods:
                fails = sum(not np.isfinite(Ps[u].point[Ps[u].ix[m]]) for u in units)
                kept = []
                for u in units:
                    r = C.load_result(a.out, bench, "test", tag, m, u)
                    acc = bm.population(bench, u, "test")
                    fin = acc[r["orders"]][:, :, -1]
                    kept.append(np.mean([K[int(np.argmax(f))] >= bm.frame(bench)["T"] for K, f in zip(r["K"], fin)]))
                row = dict(pool_size=P, final_phase=fp, method=m, failing_units=int(fails), n_units=len(units),
                           best_config_trained_to_T=float(np.mean(kept)))
                if bench == "lcbench":
                    s = Ps["lcbench"].summary(m)
                    row.update(speedup=s["speedup"], lo=s["lo"], hi=s["hi"])
                rows.append(row)
    df = pd.DataFrame(rows)
    with pd.option_context("display.max_rows", None, "display.width", 160, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))
    if a.csv:
        df.to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()

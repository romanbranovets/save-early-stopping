"""Simulate a method on a benchmark: one result file per unit with the stop epochs and the incumbent
after every epoch, for every order (or pool).

    python scripts/run.py --benchmark nb201 --method save
    python scripts/run.py --benchmark lcbench --method asha --split dev
    python scripts/run.py --benchmark db5 --method median --final-phase
    python scripts/run.py --benchmark lcbench --method save --pool-size 60 --final-phase
    python scripts/run.py --benchmark nb201 --method save --set price=0.0  --set paths=100

Settings come from configs/<benchmark>.yaml (per dataset on LCBench); --set overrides one of them
(prices are per full training, cT).  Results go to results/<benchmark>/<split>/<frame>/<label>/.
"""
from __future__ import annotations

import argparse
import time

import _common as C
import numpy as np
from joblib import Parallel, delayed

from save import benchmarks as bm
from save.runner import METHODS, run


def job(lab, bench, unit, method, params, split, n_orders, pool_size, final_phase, tables, ids, ftpfn_ckpt, device):
    model = None
    if method.startswith("ftpfn"):
        from save.baselines.ftpfn import load
        model = load(ftpfn_ckpt, device)
    return lab, unit, run(bench, unit, method, params, split, n_orders, pool_size, final_phase, tables, ids,
                          ftpfn=model)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", required=True, choices=bm.BENCHMARKS)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--split", default="test", choices=["dev", "prelim", "test"])
    ap.add_argument("--units", nargs="*", help="default: all units of the benchmark")
    ap.add_argument("--orders", type=int, default=40, help="number of orders (or pools) per unit")
    ap.add_argument("--first-orders", type=int, help="run only orders 0..n-1 (e.g. 5 for FT-PFN)")
    ap.add_argument("--pool-size", type=int, help="random pools of this many configurations")
    ap.add_argument("--final-phase", action=argparse.BooleanOptionalAction, default=None,
                    help="resume stopped runs when all configurations are started (default: from the config)")
    ap.add_argument("--set", action="append", help="override a setting, e.g. price=0.001")
    ap.add_argument("--price-sensitivity", action="store_true",
                    help="run once for every price of price_sensitivity in the config")
    ap.add_argument("--tables", default="tables", help="directory of LC-PFN / parametric decision tables")
    ap.add_argument("--ftpfn-checkpoint", help="FT-PFN checkpoint (.pt)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=5, help="orders per job")
    a = ap.parse_args()

    bench = a.benchmark
    units = a.units or bm.units(bench)
    final_phase = bm.config(bench).get("final_phase", False) if a.final_phase is None else a.final_phase
    n_run = a.first_orders or a.orders
    tag = C.frame_tag(a.pool_size, final_phase)
    variants = [C.parse_set(a.set)]
    if a.price_sensitivity:
        variants = [dict(variants[0], price=c) for c in bm.price_sensitivity(bench)]
    jobs = []
    for over in variants:
        lab = C.label(a.method, over)
        for u in units:
            p = bm.settings(bench, u).get(a.method, {})
            p.update(over)
            if C.has_result(a.out, bench, a.split, tag, lab, u, n_run):
                continue
            for i in range(0, n_run, a.chunk):
                jobs.append((lab, bench, u, a.method, p, a.split, a.orders, a.pool_size, final_phase, a.tables,
                             list(range(i, min(i + a.chunk, n_run))), a.ftpfn_checkpoint, a.device))
    t0 = time.time()
    parts = {}
    for lab, unit, r in Parallel(n_jobs=a.jobs, return_as="generator_unordered")(delayed(job)(*j) for j in jobs):
        parts.setdefault((lab, unit), []).append(r)
        if sum(len(x["order_ids"]) for x in parts[(lab, unit)]) == n_run:
            rs = sorted(parts.pop((lab, unit)), key=lambda x: x["order_ids"][0])
            out = {k: np.concatenate([x[k] for x in rs]) for k in rs[0] if k != "horizon"}
            out["horizon"] = rs[0]["horizon"]
            f = C.result_path(a.out, bench, a.split, tag, lab, unit)
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f, **out)
            print(f"[{time.time() - t0:7.0f}s] {lab} {unit} -> {f}", flush=True)


if __name__ == "__main__":
    main()

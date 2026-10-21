"""The three oracle ceilings and the split of SAVE's gap to the best-on-average oracle into the part due
to the forecast and the part due to the rule.

    python scripts/oracles.py --benchmark nb201 --jobs 3
    python scripts/oracles.py --benchmark db5 --final-phase
    python scripts/oracles.py --benchmark lcbench --summary-only

The best-on-average oracle aims at the smallest horizon at which its model's expected regret reaches
the target level.  On LCBench the horizon of each dataset is instead chosen on the other 34 datasets
(no failing dataset, highest mean speed-up), as the settings of the baselines.  Requires the results
of no-stop and save in the same frame (scripts/run.py).
"""
from __future__ import annotations

import argparse
import time

import _common as C
import numpy as np
from joblib import Parallel, delayed

from save import benchmarks as bm
from save.metrics import Panel, saved
from save.oracles import BestOnAverage, default_horizons
from save.runner import run
from save.search import run_search

ORACLES = ("oracle-all-knowing", "oracle-perfect-forecast", "oracle-best-on-average")


def all_horizons(bench, unit, split, n_orders, final_phase):
    """Every horizon of the grid for one unit, with one DP."""
    F = bm.frame(bench)
    pop = bm.population(bench, unit, split)
    O = bm.orders(bench, unit, split, n_orders)
    dp = BestOnAverage(pop, F["budget"], F["t_min"])
    out = {}
    for h in default_horizons(F["T"]):
        rs = [run_search(pop, o, dp.policy(h), F["budget"], F["slots"], F["t_min"], final_phase=final_phase) for o in O]
        out[int(h)] = dict(K=np.stack([r.K for r in rs]).astype(np.int32), best_e=np.stack([r.best_e for r in rs]),
                           used=np.array([r.used for r in rs]), best_acc=np.full(len(O), float(pop.max())),
                           orders=O[:, :F["slots"]].astype(np.int32), order_ids=np.arange(len(O)), horizon=np.array(h))
    return unit, out


def save_result(a, bench, tag, lab, unit, r):
    f = C.result_path(a.out, bench, a.split, tag, lab, unit)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **r)


def summary(bench, split, tag, out, units=None):
    from evaluate import panels
    P = panels(bench, split, tag, ["no-stop", "save", *ORACLES], out, units)
    print(f"{'set':16s} {'method':26s} speed-up [95% CI]        saved")
    for name, Pn in P.items():
        for m in ["save", *ORACLES]:
            s = Pn.summary(m)
            print(f"{name:16s} {m:26s} {s['speedup']:6.2f} [{s['lo']:.2f}, {s['hi']:.2f}]   {s['saved']:.3f}")
        f = Pn.diff("oracle-perfect-forecast", "save")
        r = Pn.diff("oracle-best-on-average", "oracle-perfect-forecast")
        ia, io = Pn.ix["save"], Pn.ix["oracle-best-on-average"]
        share = saved(Pn.bs[:, ia]) / np.maximum(saved(Pn.bs[:, io]), 1e-12)
        print(f"{name:16s} gap due to the forecast {f['diff']:+.3f} [{f['lo']:+.3f}, {f['hi']:+.3f}], "
              f"due to the rule {r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}], share of the oracle's saving "
              f"{saved(Pn.point[ia]) / saved(Pn.point[io]):.2f} [{np.percentile(share, 2.5):.2f}, "
              f"{np.percentile(share, 97.5):.2f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", required=True, choices=bm.BENCHMARKS)
    ap.add_argument("--split", default="test")
    ap.add_argument("--units", nargs="*", help="default: all units")
    ap.add_argument("--orders", type=int, default=40)
    ap.add_argument("--final-phase", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    bench = a.benchmark
    fp = bm.config(bench).get("final_phase", False) if a.final_phase is None else a.final_phase
    tag = C.frame_tag(None, fp)
    units = a.units or bm.units(bench)
    t0 = time.time()
    if not a.summary_only:
        for m in ("oracle-all-knowing", "oracle-perfect-forecast") + (() if bench == "lcbench" else ("oracle-best-on-average",)):
            todo = [u for u in units if not C.has_result(a.out, bench, a.split, tag, m, u, a.orders)]
            res = Parallel(n_jobs=a.jobs)(delayed(run)(bench, u, m, bm.settings(bench, u)[m], a.split, a.orders,
                                                       None, fp) for u in todo)
            for u, r in zip(todo, res):
                save_result(a, bench, tag, m, u, r)
            print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)
        if bench == "lcbench":
            res = dict(Parallel(n_jobs=a.jobs)(delayed(all_horizons)(bench, u, a.split, a.orders, fp) for u in units))
            hs = sorted(res[units[0]])
            keys = [f"h={h}" for h in hs]
            cur = {f"h={h}": np.stack([C.mean_regret(res[u][h]) for u in units]) for h in hs}
            cur["no-stop"] = np.stack([C.mean_regret(C.load_result(a.out, bench, a.split, tag, "no-stop", u)) for u in units])
            W = bm.bootstrap_weights(bench, "lcbench", a.split, len(units))
            pick = Panel(cur, W, bm.frame(bench)["budget"]).lodo_pick(keys)
            for u, k in zip(units, pick):
                save_result(a, bench, tag, "oracle-best-on-average", u, res[u][int(k[2:])])
            print(f"[{time.time() - t0:7.0f}s] oracle-best-on-average (horizons {sorted(set(pick))})", flush=True)
    summary(bench, a.split, tag, a.out, units)


if __name__ == "__main__":
    main()

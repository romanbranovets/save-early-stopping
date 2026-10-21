"""Selection of settings on the selection data.

    python scripts/select_settings.py --benchmark nb201 --method asha --jobs 8
    python scripts/select_settings.py --benchmark lcbench --method asha --jobs 8
    python scripts/select_settings.py --benchmark lcbench --method asha --all-datasets --jobs 8
    python scripts/select_settings.py --criterion worst-ratio --method save --jobs 8

Every setting of the method's grid (configs/grids.yaml) is simulated on the selection data; missing
results are computed.  The rule: among the settings that fail on no selection unit, the one with the
highest mean speed-up; if every setting fails somewhere, the one with the fewest failing units and then
the highest mean speed-up with a failure counted as 1; ties go to the more cautious setting (more
epochs per run).

  NAS-Bench-201   the development architectures of the three tasks.
  LCBench         for every dataset, the test configurations of the other 34 datasets.
  --all-datasets  one setting for all 35 LCBench datasets, on all their configurations (the setting
                  that NinaPro DB5 uses, and median stopping on LCBench).
  worst-ratio     SAVE's price or the threshold of save-threshold: on the development data of LCBench and
                  of the three NAS-Bench-201 tasks, the speed-up divided by that of median stopping;
                  the value with the largest worst ratio wins (a failure is the worst outcome, ties go
                  to the larger value).

C&L's threshold on LCBench and DB5 is the threshold selected for parametric-threshold; C&L with the value
rule on LCBench uses the price of SAVE, and fixed-budget evaluation is not selected on LCBench.
Methods with decision tables (LC-PFN, parametric) need the tables of the whole grid first:
scripts/tables.py --grid --split <split>.  FT-PFN was selected on LCBench on the first three orders
(--orders 3 --ftpfn-checkpoint ...).  Epoch-valued settings in grids.yaml are LCBench epochs and
are doubled on the NAS-Bench-201 grid.
"""
from __future__ import annotations

import argparse
import itertools
import math

import _common as C
import numpy as np
import yaml
from evaluate import panels
from joblib import Parallel, delayed

from save import benchmarks as bm
from save.runner import run

EPOCH_PARAMS = {"grace", "patience", "r", "epochs"}
NOT_SELECTED_ON_LCBENCH = {"fixed-budget": "uses 5 epochs", "cl2017-value": "uses the price of SAVE",
                           "cl2017": "uses the threshold selected for parametric-threshold"}


def grid(method, bench):
    g = yaml.safe_load((C.ROOT / "configs" / "grids.yaml").read_text())[method]
    scale = bm.frame(bench)["T"] / 50
    keys = sorted(g)
    out = []
    for vals in itertools.product(*(g[k] for k in keys)):
        p = {}
        for k, v in zip(keys, vals):
            if v == "none":
                v = None
            elif k in EPOCH_PARAMS:
                v = max(1, int(math.floor(v * scale + 0.5)))
            p[k] = v
        out.append(p)
    return out


def ensure(bench, method, settings, split, n_orders, out, jobs, tables, units, ftpfn=None):
    """Results of every (setting, unit) on `split`, and of no-stop."""
    todo = [(C.label(method, p), u, p) for p in settings for u in units]
    todo += [("no-stop", u, {}) for u in units]
    todo = [x for x in todo if not C.has_result(out, bench, split, "main", x[0], x[1], n_orders)]

    def one(lab, u, p):
        m = "no-stop" if lab == "no-stop" else method
        q = bm.settings(bench, u).get(m, {})
        q.update(p)
        return run(bench, u, m, q, split, n_orders, None, False, tables, ftpfn=ftpfn if m == method else None)

    res = Parallel(n_jobs=1 if ftpfn is not None else jobs)(delayed(one)(*x) for x in todo)
    for (lab, u, _), r in zip(todo, res):
        f = C.result_path(out, bench, split, "main", lab, u)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, **r)


def epochs_per_run(bench, split, lab, units, out):
    K = np.concatenate([C.load_result(out, bench, split, "main", lab, u)["K"].ravel() for u in units])
    K = K[K > 0]
    return float(K.mean()) if K.size else 0.0


def best(cands):
    """cands: (label, speed-ups per selection unit (NaN = fail), epochs per run) -> label"""
    def nfail(c):
        return int(np.sum(~np.isfinite(c[1])))
    fewest = min(nfail(c) for c in cands)
    pool = [c for c in cands if nfail(c) == fewest]
    return max(pool, key=lambda c: (round(float(np.mean(np.where(np.isfinite(c[1]), c[1], 1.0))), 12), c[2]))[0]


def per_unit_speedups(bench, split, labels, out, units):
    """{label: (speed-up per unit,)} on the main frame; NAS-Bench-201 units are tasks, LCBench datasets."""
    P = panels(bench, split, "main", ["no-stop"] + labels, out, units)
    if bench == "nb201":
        return {k: np.array([P[t].point[P[t].ix[k]] for t in units]) for k in labels}
    Pn = P["lcbench"]
    return {k: Pn.per[Pn.ix[k]] for k in labels}


def select(bench, method, split, n_orders, out, jobs, tables, per_dataset, units=None, ftpfn=None):
    settings = grid(method, bench)
    labels = [C.label(method, p) for p in settings]
    units = units or bm.units(bench)
    ensure(bench, method, settings, split, n_orders, out, jobs, tables, units, ftpfn)
    sp = per_unit_speedups(bench, split, labels, out, units)
    if not per_dataset:
        ep = {k: epochs_per_run(bench, split, k, units, out) for k in labels}
        return {"all": best([(k, sp[k], ep[k]) for k in labels])}
    picks = {}
    for d, u in enumerate(units):
        other = [x for x in units if x != u]
        keep = np.arange(len(units)) != d
        ep = {k: epochs_per_run(bench, split, k, other, out) for k in labels}
        picks[u] = best([(k, sp[k][keep], ep[k]) for k in labels])
    return picks


def worst_ratio(method, n_orders, out, jobs, tables, units=None):
    key = "price" if method == "save" else "delta"
    values = yaml.safe_load((C.ROOT / "configs" / "grids.yaml").read_text())[method][key]
    labels = [C.label(method, {key: v}) for v in values]
    ratio = {v: [] for v in values}
    for bench in ("lcbench", "nb201"):
        us = [u for u in units or [] if u in bm.units(bench)] or bm.units(bench)
        ensure(bench, method, [{key: v} for v in values], "dev", n_orders, out, jobs, tables, us)
        ensure(bench, "median", [{}], "dev", n_orders, out, jobs, tables, us)
        for Pn in panels(bench, "dev", "main", ["no-stop", "median"] + labels, out, us).values():
            med = Pn.point[Pn.ix["median"]]
            for v, k in zip(values, labels):
                s = Pn.point[Pn.ix[k]]
                ratio[v].append(s / med if np.isfinite(s) else -np.inf)
    worst = {v: min(r) for v, r in ratio.items()}
    pick = max(values, key=lambda v: (worst[v], v))
    for v in values:
        print(f"{key} {v:.6g}: worst ratio to median stopping {worst[v]:.3f}" + ("  <- selected" if v == pick else ""))
    return pick


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", choices=["nb201", "lcbench"])
    ap.add_argument("--method", required=True)
    ap.add_argument("--criterion", default="per-benchmark", choices=["per-benchmark", "worst-ratio"])
    ap.add_argument("--all-datasets", action="store_true")
    ap.add_argument("--units", nargs="*", help="default: all units")
    ap.add_argument("--orders", type=int, default=40)
    ap.add_argument("--tables", default="tables")
    ap.add_argument("--ftpfn-checkpoint", help="needed for --method ftpfn-threshold")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    if a.criterion == "worst-ratio":
        if a.method not in ("save", "save-threshold"):
            raise SystemExit("worst-ratio selects save or save-threshold")
        print("selected:", worst_ratio(a.method, a.orders, a.out, a.jobs, a.tables, a.units))
        return
    if a.benchmark == "nb201":
        picks = select("nb201", a.method, "dev", a.orders, a.out, a.jobs, a.tables, False, a.units)
    else:
        method = "parametric-threshold" if a.method == "cl2017" else a.method
        if a.method in NOT_SELECTED_ON_LCBENCH and a.method != "cl2017":
            raise SystemExit(f"{a.method} is not selected on LCBench: it {NOT_SELECTED_ON_LCBENCH[a.method]}")
        whole = a.all_datasets or a.method == "median"
        ftpfn = None
        if a.method.startswith("ftpfn"):
            from save.baselines.ftpfn import load
            ftpfn = load(a.ftpfn_checkpoint, a.device)
        picks = select("lcbench", method, "all" if whole else "test", a.orders, a.out, a.jobs, a.tables,
                       not whole, a.units, ftpfn)
        if a.method == "cl2017":
            print("C&L uses the threshold selected for parametric-threshold:")
    for unit, lab in picks.items():
        print(f"{unit}: {lab}")


if __name__ == "__main__":
    main()

"""Speed-ups, saved fractions, failures, bootstrap intervals, paired differences and shares of the best
method, from the result files of scripts/run.py.

    python scripts/evaluate.py --benchmark nb201 --methods no-stop save median asha --paired save
    python scripts/evaluate.py --benchmark db5 --frame main-final --methods no-stop save median --paired save
    python scripts/evaluate.py --benchmark lcbench --frame pool60-final --methods no-stop save median

Every method is compared with the reference "no-stop" of the same frame, which must have been run.
On NAS-Bench-201 each task is a test set and the bootstrap resamples its orders; with --paired the
mean difference over the three tasks is reported too.  On LCBench (datasets) and DB5 (subjects of a
schedule) the regret curves are averaged over the units of a set; failures are counted per unit.
Pool frames (poolP...) are evaluated per unit with a bootstrap over pools.
"""
from __future__ import annotations

import argparse
import pathlib

import _common as C
import numpy as np
import pandas as pd

from save import benchmarks as bm
from save.metrics import Panel, pooled_diff, shares_of_best


def panels(bench, split, tag, methods, out, units=None):
    """{set name: Panel}.  Main frames: test sets; pool frames: one panel per unit over its pools."""
    B = bm.frame(bench)["budget"]
    units = units or bm.units(bench)
    res = {m: {u: C.load_result(out, bench, split, tag, m, u) for u in units} for m in methods}
    P = {}
    if tag.startswith("pool"):
        size = int(tag[4:].split("-")[0])
        for u in units:
            cur = {m: res[m][u]["best_acc"][:, None] - res[m][u]["best_e"] for m in methods}
            P[u] = Panel(cur, bm.pool_bootstrap_weights(bench, u, size, len(cur["no-stop"])), B)
        if bench == "lcbench":                  # all datasets together, as in the main frame
            cur = {m: np.stack([C.mean_regret(res[m][u]) for u in units]) for m in methods}
            P["lcbench"] = Panel(cur, bm.bootstrap_weights(bench, "lcbench", split, len(units)), B)
            P["lcbench"].unit_names = units
        return P
    for name, us in bm.test_sets(bench).items():
        us = [u for u in us if u in units]
        if not us:
            continue
        if bench == "nb201":                    # units of the bootstrap are the orders of the task
            cur = {m: res[m][us[0]]["best_acc"][:, None] - res[m][us[0]]["best_e"] for m in methods}
        else:
            cur = {m: np.stack([C.mean_regret(res[m][u]) for u in us]) for m in methods}
        P[name] = Panel(cur, bm.bootstrap_weights(bench, name, split, len(cur["no-stop"])), B)
        if bench != "nb201":
            P[name].unit_names = us
    return P


def curve_rows(bench, name, Pn, methods):
    """mean regret curve of every method over the units of a panel, long format"""
    r0 = float(Pn.ns[:, -1].mean())
    rows = []
    for m in methods:
        curve = Pn.K[Pn.ix[m]].mean(axis=0)
        rows += [dict(benchmark=bench, set=name, method=m, epochs=b + 1, regret=float(v), target=r0)
                 for b, v in enumerate(curve)]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", required=True, nargs="+", choices=bm.BENCHMARKS)
    ap.add_argument("--methods", required=True, nargs="+", help="result labels, including no-stop")
    ap.add_argument("--split", default="test")
    ap.add_argument("--units", nargs="*", help="default: all units")
    ap.add_argument("--frame", default=None, help="main, main-final, poolP, poolP-final (default: main, "
                                                  "main-final on DB5)")
    ap.add_argument("--paired", help="report saved(X) - saved(this method) for every other method")
    ap.add_argument("--out", default="results")
    ap.add_argument("--csv", help="write the table to this file")
    ap.add_argument("--curves", help="write the mean regret curves of the methods to this CSV file")
    a = ap.parse_args()
    rows, sp, curves = [], {}, []
    for bench in a.benchmark:
        tag = a.frame or ("main-final" if bm.config(bench).get("final_phase") else "main")
        Ps = panels(bench, a.split, tag, a.methods, a.out, [u for u in a.units or [] if u in bm.units(bench)] or None)
        for name, Pn in Ps.items():
            if a.curves:
                curves += curve_rows(bench, name, Pn, a.methods)
            for m in a.methods:
                s = Pn.summary(m)
                failed = [u for u, f in zip(getattr(Pn, "unit_names", None) or [], Pn.fail[Pn.ix[m]]) if f]
                if failed:
                    print(f"{bench} {name} {m}: fails on {', '.join(failed)}")
                rows.append(dict(benchmark=bench, set=name, method=m, metric="speedup", value=s["speedup"],
                                 lo=s["lo"], hi=s["hi"], fail_units=s["fail_units"], n_units=s["n_units"]))
                rows.append(dict(benchmark=bench, set=name, method=m, metric="saved", value=s["saved"],
                                 lo=s["saved_lo"], hi=s["saved_hi"], fail_units=s["fail_units"], n_units=s["n_units"]))
                sp.setdefault(f"{bench}:{name}", {})[m] = s["speedup"]
                if a.paired and m != a.paired:
                    d = Pn.diff(a.paired, m)
                    rows.append(dict(benchmark=bench, set=name, method=f"{a.paired} - {m}", metric="saved difference",
                                     value=d["diff"], lo=d["lo"], hi=d["hi"]))
        if bench == "nb201" and a.paired and not tag.startswith("pool"):
            for m in a.methods:
                if m != a.paired:
                    d = pooled_diff(list(Ps.values()), a.paired, m)
                    rows.append(dict(benchmark=bench, set="mean of tasks", method=f"{a.paired} - {m}",
                                     metric="saved difference", value=d["diff"], lo=d["lo"], hi=d["hi"]))
    if len(sp) > 1 and not (a.frame or "").startswith("pool"):
        for m, (mean, worst, n_best) in shares_of_best(sp).items():
            rows.append(dict(benchmark="all", set="all", method=m, metric="share of best (mean)", value=mean))
            rows.append(dict(benchmark="all", set="all", method=m, metric="share of best (worst)", value=worst))
            rows.append(dict(benchmark="all", set="all", method=m, metric="best on sets", value=n_best))
    df = pd.DataFrame(rows)
    with pd.option_context("display.max_rows", None, "display.width", 160, "display.float_format", "{:.4f}".format):
        print(df.to_string(index=False))
    for path, table in ((a.csv, df), (a.curves, pd.DataFrame(curves))):
        if path:
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(path, index=False)


if __name__ == "__main__":
    main()

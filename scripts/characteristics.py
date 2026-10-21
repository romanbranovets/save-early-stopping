"""How late the curves of a benchmark improve and how early its best configurations are visible.

    python scripts/characteristics.py --benchmark nb201 lcbench db5 [--split test] [--csv out.csv]

Per curve (best accuracy so far b, gain = b(T) - y(1); curves with a gain below the recording quantum
are left out): the share of the gain obtained in the last quarter of training, and the fraction of
training after which 95 % of the gain is reached; both are medians over the curves of a unit.
Per search over 60 configurations (40 random pools; on DB5 the unit itself): the Spearman correlation
between the accuracy at t_min and the final best accuracy, and the share of the final top 5 % that rank
below the median at t_min; both are averaged over the pools of a unit.  Sets report the median over
their units.
"""
from __future__ import annotations

import argparse
import math

import _common  # noqa: F401  (puts the repository on the path)
import numpy as np
import pandas as pd
from scipy import stats

from save import benchmarks as bm

POOL = 60


def curve_stats(acc, quantum):
    T = acc.shape[1]
    b = np.maximum.accumulate(acc, axis=1)
    gain = b[:, -1] - acc[:, 0]
    keep = gain > quantum
    late = (b[keep, -1] - b[keep, math.ceil(0.75 * T) - 1]) / gain[keep]
    reached = (b[keep] - acc[keep, :1]) >= 0.95 * gain[keep, None] - 1e-12
    t95 = (np.argmax(reached, axis=1) + 1) / T
    return float(np.median(late)), float(np.median(t95))


def search_stats(acc, t_min):
    early, final = acc[:, t_min - 1], acc.max(axis=1)
    rho = stats.spearmanr(early, final).statistic if early.std() > 0 and final.std() > 0 else np.nan
    top = np.argsort(-final, kind="stable")[:math.ceil(0.05 * len(acc))]
    return rho, float((early[top] < np.median(early)).mean())


def unit_row(bench, unit, split):
    acc = bm.population(bench, unit, split)
    quantum = 2.0 * bm.floor(bench, unit, acc)
    t_min = bm.frame(bench)["t_min"]
    late, t95 = curve_stats(acc, quantum)
    pools = [np.arange(len(acc))] if bench == "db5" else list(bm.pools(bench, unit, POOL, split))
    s = np.array([search_stats(acc[o], t_min) for o in pools])
    return dict(benchmark=bench, unit=unit, gain_last_quarter=late, gain95_at=t95,
                early_ranking=float(np.nanmean(s[:, 0])), late_winners=float(s[:, 1].mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", nargs="+", required=True, choices=bm.BENCHMARKS)
    ap.add_argument("--split", default="test")
    ap.add_argument("--units", nargs="*", help="default: all units")
    ap.add_argument("--csv", help="write the per-unit values to this file")
    a = ap.parse_args()
    rows = [unit_row(b, u, a.split) for b in a.benchmark
            for u in [x for x in a.units or [] if x in bm.units(b)] or bm.units(b)]
    df = pd.DataFrame(rows)
    cols = ["gain_last_quarter", "gain95_at", "early_ranking", "late_winners"]
    sets = []
    for b in a.benchmark:
        groups = {"nb201": {"nb201": bm.units(b)}, "lcbench": {"lcbench": bm.units(b)}}.get(b, bm.test_sets(b))
        for name, us in groups.items():
            x = df[(df.benchmark == b) & df.unit.isin(us)]
            sets.append(dict(set=name, units=len(x), **{c: float(x[c].median()) for c in cols}))
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))
        print()
        print(pd.DataFrame(sets).to_string(index=False))
    if a.csv:
        df.to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()

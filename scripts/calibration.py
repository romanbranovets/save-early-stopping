"""How the forecast sees each configuration at its first decision point (t = t_min), in searches over
60 configurations: the PIT of the final best accuracy, whether it lies above the 80 % interval, and
whether SAVE continues the run.  The winner is the configuration with the highest final accuracy among
those of the search; "rest" are the configurations ranked below third.

    python scripts/calibration.py --benchmark lcbench nb201 db5
    python scripts/calibration.py --benchmark db5 --forecast ftpfn --ftpfn-checkpoint path/to/ftpfn.pt

The donors of a decision are the earlier runs of the same search at the epochs SAVE trained them; the
SAVE searches are those of scripts/run.py (LCBench and NAS-Bench-201: pools of 60 without the final
phase, DB5: the 40 recorded orders), computed here if missing.
"""
from __future__ import annotations

import argparse

import _common as C
import numpy as np
import pandas as pd

from save import benchmarks as bm
from save.forecast import forecast_paths
from save.method import cached_banks
from save.metrics import pit
from save.rule import value_continue
from save.runner import run


def searches(bench, unit, out):
    tag, P = ("main", None) if bench == "db5" else ("pool60", 60)
    f = C.result_path(out, bench, "test", tag, "save", unit)
    if not f.exists():
        r = run(bench, unit, "save", bm.settings(bench, unit)["save"], "test", 40, P, False)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, **r)
    r = C.load_result(out, bench, "test", tag, "save", unit)
    pop = bm.population(bench, unit, "test")
    return pop, r


def final_rank(fb):
    o = np.argsort(-fb, kind="stable")
    r = np.empty(len(fb), np.int64)
    r[o] = np.arange(1, len(fb) + 1)
    return r


def unit_rows(bench, unit, out, model=None):
    pop, r = searches(bench, unit, out)
    F = bm.frame(bench)
    T, t = F["T"], F["t_min"]
    fl = bm.floor(bench, unit, pop)
    price = bm.settings(bench, unit)["save"]["price"] / T
    Z, U, Us = cached_banks(400, T)
    hp = bm.hyperparameters(bench, unit) if model is not None else None
    rows = []
    for s, (o, K) in enumerate(zip(r["orders"], r["K"])):
        acc = pop[o]                                # the search's own configurations, in its order
        runmax = np.maximum.accumulate(acc, axis=1)
        fr = final_rank(runmax[:, -1])
        n = int((K > 0).sum())
        prev = -np.inf
        for pos in range(n):
            if pos >= 1:
                truth, bt = runmax[pos, -1], runmax[pos, t - 1]
                u = max(prev, bt)
                if model is None:
                    p = forecast_paths(acc[pos, :t], acc[:pos], K[:pos], Z, U, Us, fl)
                    pred = np.maximum(bt, p.max(axis=1))
                    lo, hi = np.quantile(pred, [0.1, 0.9])
                    rows.append(dict(bench=bench, unit=unit, order=s, pos=pos, final_rank=int(fr[pos]),
                                     pit=pit(pred, truth), above80=bool(truth > hi), below80=bool(truth < lo),
                                     kept=value_continue(p, bt, u, price)))
                elif fr[pos] == 1:
                    from save.baselines.ftpfn import final_best_pit
                    runs = [(j, int(K[j])) for j in range(pos)]
                    v, lo, med, hi = final_best_pit(model, acc, hp[o], runs, pos, pos, t, T)
                    rows.append(dict(bench=bench, unit=unit, order=s, pos=pos, final_rank=1, pit=v,
                                     above80=bool(truth > hi), below80=bool(truth < lo), kept=np.nan))
            prev = max(prev, runmax[pos, K[pos] - 1])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", nargs="+", required=True, choices=bm.BENCHMARKS)
    ap.add_argument("--units", nargs="*", help="default: all units")
    ap.add_argument("--forecast", default="save", choices=["save", "ftpfn"])
    ap.add_argument("--ftpfn-checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results")
    ap.add_argument("--csv")
    a = ap.parse_args()
    model = None
    if a.forecast == "ftpfn":
        from save.baselines.ftpfn import load
        model = load(a.ftpfn_checkpoint, a.device)
    rows = []
    for bench in a.benchmark:
        for u in [x for x in a.units or [] if x in bm.units(bench)] or bm.units(bench):
            rows += unit_rows(bench, u, a.out, model)
    df = pd.DataFrame(rows)
    df["set"] = np.where(df.bench == "db5", df.unit.str.split("/").str[0], df.bench)
    out = []
    for name, x in df.groupby("set", sort=False):
        w, rest = x[x.final_rank == 1], x[x.final_rank > 3]
        out.append(dict(set=name, mean_pit_winner=w.pit.mean(), mean_pit_rest=rest.pit.mean() if len(rest) else np.nan,
                        winner_above_80=w.above80.mean(), winner_below_80=w.below80.mean(),
                        winner_kept=w.kept.astype(float).mean(), n_winner=len(w)))
    res = pd.DataFrame(out)
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))
    if a.csv:
        df.to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()

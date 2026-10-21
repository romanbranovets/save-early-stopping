"""Speed-up, saved fraction, failures, bootstrap intervals and paired differences.

A failing method (it never reaches the final regret of no-stop) has speed-up NaN; in a bootstrap
replicate it counts as speed-up 1 and saved fraction 0."""
from __future__ import annotations

import numpy as np

from .bootstrap import ci


def regret(best_e, best_acc):
    return best_acc - np.asarray(best_e, np.float64)


def speedup(curves, ref, budget):
    """curves (..., budget), ref (...) -> speed-up, NaN where the curve never reaches ref."""
    hit = curves <= ref[..., None] + 1e-12
    first = np.where(hit.any(axis=-1), hit.argmax(axis=-1) + 1, -1)
    return np.where(first > 0, budget / np.maximum(first, 1), np.nan)


def saved(S):
    S = np.asarray(S, float)
    return np.where(np.isfinite(S), 1 - 1 / S, 0.0)


def fail_as_one(S):
    S = np.asarray(S, float)
    return np.where(np.isfinite(S), S, 1.0)


class Panel:
    """curves {method: (D, budget)} must contain "no-stop"; W (n, D) bootstrap weights over units."""

    def __init__(self, curves, W, budget, reference="no-stop"):
        self.budget = budget
        self.keys = list(curves)
        self.ix = {k: i for i, k in enumerate(self.keys)}
        self.K = np.stack([np.asarray(curves[k], np.float64) for k in self.keys])
        self.ns = np.asarray(curves[reference], np.float64)
        self.W = W
        self.per = speedup(self.K, self.ns[None, :, -1], budget)                 # per unit
        self.fail = np.isnan(self.per)
        self.point = speedup(self.K.mean(1), np.full(len(self.keys), self.ns[:, -1].mean()), budget)
        self.bs = np.empty((len(W), len(self.keys)))
        for a in range(0, len(W), 50):
            w = W[a:a + 50]
            mc = np.einsum("bd,mde->bme", w, self.K)
            ref = w @ self.ns[:, -1]
            self.bs[a:a + 50] = speedup(mc, np.broadcast_to(ref[:, None], mc.shape[:2]), budget)

    def add(self, key, curve):
        p = Panel({key: curve, "no-stop": self.ns}, self.W, self.budget)
        self.keys.append(key)
        self.ix[key] = len(self.keys) - 1
        self.K = np.concatenate([self.K, np.asarray(curve, np.float64)[None]])
        self.per = np.vstack([self.per, p.per[:1]])
        self.fail = np.vstack([self.fail, p.fail[:1]])
        self.point = np.append(self.point, p.point[0])
        self.bs = np.hstack([self.bs, p.bs[:, :1]])

    def n_fail(self, key):
        return int(self.fail[self.ix[key]].sum())

    def summary(self, key):
        """speed-up (NaN = fail), its CI, saved, its CI, failing units"""
        j = self.ix[key]
        b = self.bs[:, j]
        return dict(speedup=float(self.point[j]), lo=ci(fail_as_one(b))[0], hi=ci(fail_as_one(b))[1],
                    saved=float(saved(self.point[j])), saved_lo=ci(saved(b))[0], saved_hi=ci(saved(b))[1],
                    fail_units=self.n_fail(key), n_units=self.K.shape[1])

    def diff_replicates(self, a, b):
        """saved(a) - saved(b): point and replicates (paired, same replicates)."""
        ia, ib = self.ix[a], self.ix[b]
        return float(saved(self.point[ia]) - saved(self.point[ib])), saved(self.bs[:, ia]) - saved(self.bs[:, ib])

    def diff(self, a, b):
        d, r = self.diff_replicates(a, b)
        lo, hi = ci(r)
        return dict(diff=d, lo=lo, hi=hi)

    def lodo_pick(self, keys):
        """For every unit: among `keys`, the one with no failing unit on the other units and the highest
        mean per-unit speed-up there (the first key if none qualifies)."""
        D = self.ns.shape[0]
        js = [self.ix[k] for k in keys]
        out = []
        for f in range(D):
            oth = np.arange(D) != f
            ok = [i for i, j in enumerate(js) if not self.fail[j][oth].any()]
            out.append(keys[max(ok, key=lambda i: np.nanmean(self.per[js[i]][oth]))] if ok else keys[0])
        return out

    def compose(self, name, picks):
        """A new method whose curve on unit d is that of picks[d]."""
        self.add(name, np.stack([self.K[self.ix[k]][d] for d, k in enumerate(picks)]))


def pooled_diff(panels, a, b):
    """Mean over several panels (e.g. the tasks of one benchmark) of saved(a) - saved(b), with the
    replicates of the panels combined index by index (resampled independently within each panel)."""
    pts, reps = zip(*(p.diff_replicates(a, b) for p in panels))
    r = np.mean(reps, axis=0)
    lo, hi = ci(r)
    return dict(diff=float(np.mean(pts)), lo=lo, hi=hi)


def shares_of_best(speedups):
    """speedups {set: {method: S (NaN = fail)}} -> {method: (mean share, worst share, sets where best)}."""
    methods = sorted({m for v in speedups.values() for m in v})
    share = {m: [] for m in methods}
    best_on = {m: 0 for m in methods}
    for v in speedups.values():
        s = {m: (v[m] if np.isfinite(v.get(m, np.nan)) else 0.0) for m in methods}
        top = max(s.values())
        for m in methods:
            share[m].append(s[m] / top if top > 0 else 0.0)
            best_on[m] += int(s[m] == top and top > 0)
    return {m: (float(np.mean(share[m])), float(np.min(share[m])), best_on[m]) for m in methods}


def pit(pred, truth):
    """Probability integral transform of `truth` under predictive samples (ties count one half)."""
    pred = np.asarray(pred)
    return float((pred < truth).mean() + 0.5 * (pred == truth).mean())

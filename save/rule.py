"""Stopping rules on forecast paths of a run at epoch t with best accuracy b(t) and incumbent u.

value rule:      continue iff max_{k>t} [G(k) - c (k - t)] > 0, G(k) = E[max(B(k), u)] - u
threshold rule:  continue iff max_{k>t} P(path(k) > u) > delta
For tables, the value rule is stored as the incumbent u* at which the left-hand side is zero."""
from __future__ import annotations

import numpy as np


def gain_curve(paths, b_t, u):
    """G(k) for every future epoch of `paths` (M, L): (L,)."""
    B = np.maximum(b_t, np.maximum.accumulate(paths, axis=1))
    return np.maximum(B - u, 0.0).mean(axis=0)


def value_continue(paths, b_t, u, price):
    G = gain_curve(paths, b_t, u)
    return bool((G - price * np.arange(1, len(G) + 1)).max() > 0)


def threshold_continue(paths, u, delta):
    return bool((paths > u).mean(axis=0).max() > delta)


def value_thresholds_from_paths(paths, b_t, prices):
    """u*(c) for every price, from joint paths (n, L).

    Sorting B(k) as x_1 >= ... >= x_n with partial sums S_m, G is piecewise linear in u:
    G(x_m) = g_m, g_1 = 0, g_{m+1} = g_m + m (x_m - x_{m+1}) / n; on [x_{m+1}, x_m]
    G(u) = (S_m - m u) / n.  Returns (C,)."""
    paths = np.asarray(paths, np.float64)
    n, L = paths.shape
    B = np.maximum(float(b_t), np.maximum.accumulate(paths, axis=1))
    x = -np.sort(-B, axis=0)
    S = np.cumsum(x, axis=0)
    g = np.zeros((n, L))
    g[1:] = np.cumsum(np.arange(1, n)[:, None] * (x[:-1] - x[1:]), axis=0) / n
    lev = np.asarray(prices, np.float64)[None, :] * np.arange(1, L + 1, dtype=np.float64)[:, None]
    out = np.empty(lev.shape)
    for k in range(L):
        ms = np.searchsorted(g[:, k], lev[k], side="right")
        out[k] = (S[ms - 1, k] - n * lev[k]) / ms
    return out.max(axis=0)


def prob_thresholds_from_paths(paths, deltas):
    """'max_k P(path(k) > y) <= delta' is 'y >= thr' (empirical probabilities of n paths)."""
    paths = np.asarray(paths, np.float64)
    n = paths.shape[0]
    srt = -np.sort(-paths, axis=0)
    m = np.floor(n * np.asarray(deltas, np.float64) + 1e-9).astype(int)
    out = np.full(len(m), -np.inf)
    ok = m < n
    out[ok] = srt[m[ok], :].max(axis=1)
    return out


def value_thresholds_from_bars(probs, borders, prices):
    """u*(c) when only per-epoch marginals are known (LC-PFN): G_marg(k) = E[max(Y(k), u)] - u with
    piecewise-uniform bar densities.  probs (L, nb) for epochs t+1..t+L, borders (nb + 1,)."""
    p = np.asarray(probs, np.float64)
    b = np.asarray(borders, np.float64)
    L, nb = p.shape
    w = np.diff(b)
    mid = 0.5 * (b[:-1] + b[1:])
    T0 = np.zeros((L, nb + 1))
    T1 = np.zeros((L, nb + 1))
    T0[:, :nb] = np.cumsum(p[:, ::-1], axis=1)[:, ::-1]
    T1[:, :nb] = np.cumsum((p * mid)[:, ::-1], axis=1)[:, ::-1]
    Gb = np.minimum.accumulate(np.maximum(T1 - b[None, :] * T0, 0.0), axis=1)
    lev = np.asarray(prices, np.float64)[None, :] * np.arange(1, L + 1, dtype=np.float64)[:, None]
    out = np.empty(lev.shape)
    for k in range(L):
        cnt = np.searchsorted(-Gb[k], -lev[k], side="left")
        below = cnt == 0
        j = np.clip(cnt - 1, 0, nb - 1)
        d = lev[k] - Gb[k, j + 1]
        a = p[k, j] / (2.0 * w[j])
        t0 = T0[k, j + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            xx = 2.0 * d / (t0 + np.sqrt(t0 * t0 + 4.0 * a * d))
        xx = np.where(d > 0, xx, 0.0)
        out[k] = np.where(below, b[0] - (lev[k] - Gb[k, 0]), b[j + 1] - xx)
    return out.max(axis=0)


def table_stop_epoch(acc_c, prev, thr_c, t_lo, t_hi):
    """First decision epoch t in [t_lo, t_hi] with max(prev, b(t)) >= thr_c[t]; T if none.
    thr_c (T + 1,) is indexed by the decision epoch."""
    T = len(acc_c)
    yb = np.maximum(prev, np.maximum.accumulate(acc_c))
    th = np.asarray(thr_c)[1:T + 1]
    t = np.arange(1, T + 1)
    hit = (yb >= th) & (t >= t_lo) & (t <= t_hi)
    return int(hit.argmax()) + 1 if hit.any() else T

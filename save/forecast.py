"""The SAVE forecast: joint future paths of the current run built from earlier runs of the same search.

The random numbers come from fixed banks shared by all decisions, so a search is a deterministic
function of the curves."""
from __future__ import annotations

import numpy as np

from .seeds import bank_seed, stage_bank_seed


def banks(M, T):
    """Z (M, T) standard normals, U (M,) uniforms for the first donor, Us (T + 1, M) uniforms for
    the donors chosen when a path is stitched."""
    rng = np.random.default_rng(bank_seed(M, T))
    Z, U = rng.standard_normal((M, T)), rng.random(M)
    Us = np.random.default_rng(stage_bank_seed(M, T)).random((T + 1, M))
    return Z, U, Us


def noise_sd(y_prefix, floor):
    d = np.diff(np.asarray(y_prefix, np.float64))
    if len(d) == 0:
        return float(floor)
    return max(float(d[-5:].std()), float(floor))


def _slope(Y, t, w):
    return (Y[..., t - 1] - Y[..., t - 1 - w]) / w if w > 0 else np.zeros_like(Y[..., t - 1])


def _bandwidth(diff):
    """median heuristic; 1 if all distances are zero"""
    h = np.median(diff)
    return h if h > 0 else 1.0


def forecast_paths(y_prefix, donors, donor_epochs, Z, U, Us, floor, continuation="add", weighted=True):
    """(M, T - t) joint paths of the accuracy at epochs t+1..T of a run with prefix y(1..t).

    donors        (n, T) accuracy curves of the earlier runs (values past a donor's end are ignored)
    donor_epochs  (n,) epochs each donor was trained; at least one donor must reach T
    continuation  "add": y_path(k) = y(tau) + y_j(k) - y_j(tau);  "mult": the donor's error is scaled
    weighted      donors drawn with the level/slope kernel (True) or uniformly (False)
    """
    y = np.asarray(y_prefix, np.float64)
    DY = np.asarray(donors, np.float64)
    De = np.asarray(donor_epochs, np.int64)
    t, T, M = len(y), DY.shape[1], len(U)
    assert (De >= T).any(), "no donor reaches the end of training"
    P = np.empty((M, T))
    P[:, :t] = y[None, :]
    tau = np.full(M, t, np.int64)
    live = np.arange(M)
    cols = np.arange(T)
    stage = 0
    while live.size:
        tl = tau[live]
        for tv in np.unique(tl):
            idx = live[tl == tv] if stage else live
            elig = np.flatnonzero(De > tv)
            n = len(elig)
            w_ = min(3, tv - 1)
            if not weighted:
                cw = np.cumsum(np.full(n, 1.0 / n))
                cw = cw / cw[-1]
                u = (idx + (U[idx] if stage == 0 else Us[stage, idx])) / M
                jj = np.minimum(np.searchsorted(cw, u, side="right"), n - 1)
            elif stage == 0:                         # the query is the current run itself
                L, Sl = DY[elig, tv - 1], _slope(DY[elig], tv, w_)
                lev = y[t - 1]
                sl = (y[t - 1] - y[t - 1 - w_]) / w_ if w_ > 0 else 0.0
                d2 = np.zeros(n)
                for a, b in ((lev, L), (sl, Sl)):
                    diff = np.abs(a - b)
                    d2 += (diff / _bandwidth(diff)) ** 2
                w = np.exp(-0.5 * d2)
                w = w / w.sum()
                cw = np.cumsum(w)
                cw /= cw[-1]
                u = (idx + U[idx]) / M
                jj = np.minimum(np.searchsorted(cw, u, side="right"), n - 1)
            else:                                    # one query per path, one bandwidth per stitch epoch
                L, Sl = DY[elig, tv - 1], _slope(DY[elig], tv, w_)
                qlev = P[idx, tv - 1]
                qsl = (P[idx, tv - 1] - P[idx, tv - 1 - w_]) / w_ if w_ > 0 else np.zeros(len(idx))
                d2 = np.zeros((len(idx), n))
                for a, b in ((qlev, L), (qsl, Sl)):
                    diff = np.abs(a[:, None] - b[None, :])
                    d2 += (diff / _bandwidth(diff)) ** 2
                w = np.exp(-0.5 * (d2 - d2.min(axis=1, keepdims=True)))
                cw = np.cumsum(w, axis=1)
                cw = cw / cw[:, -1:]
                u = (idx + Us[stage, idx]) / M
                jj = np.minimum((cw <= u[:, None]).sum(axis=1), n - 1)
            j = elig[jj]
            end = np.minimum(De[j], T)
            base = DY[j, tv - 1]
            p0 = P[idx, tv - 1][:, None]
            seg = DY[j, tv:int(end.max())]
            inside = cols[None, tv:int(end.max())] < end[:, None]
            with np.errstate(invalid="ignore"):
                if continuation == "add":
                    val = p0 + seg - base[:, None]
                else:
                    e0 = (1.0 - base)[:, None]
                    ratio = np.where(e0 > 0, (1.0 - seg) / np.where(e0 > 0, e0, 1.0), 1.0)
                    val = 1.0 - (1.0 - p0) * ratio
            P[idx, tv:int(end.max())] = np.where(inside, val, P[idx, tv:int(end.max())])
            tau[idx] = end
            if stage == 0:
                break
        live = live[tau[live] < T]
        stage += 1
    return P[:, t:] + noise_sd(y, floor) * Z[:, t:T]

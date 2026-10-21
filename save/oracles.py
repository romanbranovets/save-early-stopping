"""Oracle ceilings: the all-knowing envelope and the best-on-average dynamic programme over (epochs
left, incumbent), whose next configurations are independent draws from the unit's curves."""
from __future__ import annotations

import math

import numpy as np

from .search import Policy

BS = 256


def envelope(acc, order, budget, t_min=2):
    """(budget,) Env(b) for one order."""
    T = acc.shape[1]
    runmax = np.maximum.accumulate(acc, axis=1)
    o = np.asarray(order)[:budget // t_min + 1]
    b = np.arange(1, budget + 1)
    out = np.full(budget, -np.inf)
    before = np.r_[-np.inf, np.maximum.accumulate(runmax[o, t_min - 1])[:-1]]
    for j in range(len(o)):
        rest = b - t_min * j
        ok = rest >= 1
        if not ok.any():
            break
        v = np.full(budget, -np.inf)
        v[ok] = np.maximum(before[j], runmax[o[j], np.minimum(rest[ok], T) - 1])
        out = np.maximum(out, v)
    return out


class Options:
    """Stop options of every curve: epochs >= t_min at which the running minimum of the error
    strictly improves, plus the first allowed epoch; values on the grid U of distinct values."""

    def __init__(self, err, t_min):
        err = np.asarray(err, np.float64)
        N, T = err.shape
        R = np.minimum.accumulate(err, axis=1)
        ks = np.arange(t_min, T + 1)
        Rk = R[:, ks - 1]
        keep = np.ones(Rk.shape, bool)
        keep[:, 1:] = Rk[:, 1:] < Rk[:, :-1]
        ptr = np.r_[0, np.cumsum(keep.sum(axis=1))]
        rows, cols = np.nonzero(keep)
        self.k = ks[cols]
        v = Rk[rows, cols]
        self.N, self.P = N, len(self.k)
        self.U = np.unique(v)
        self.nU = len(self.U)
        self.iv = np.searchsorted(self.U, v)
        self.p0 = ptr[:-1].copy()
        self.pl = ptr[1:] - 1
        self.iv0, self.ivl = self.iv[self.p0], self.iv[self.pl]
        L = np.diff(ptr)
        self.Lmax = int(L.max())
        col = np.arange(self.P) - np.repeat(self.p0, L)
        self.opt = np.full((N, self.Lmax), -1)
        self.opt[rows, col] = np.arange(self.P)
        self.valid = self.opt >= 0
        self.optc = np.where(self.valid, self.opt, 0)
        nf = np.flatnonzero(col > 0)                     # options that start a piece
        self.pc_opt, self.pc_lo, self.pc_hi, self.pc_cfg = nf, self.iv[nf] + 1, self.iv[nf - 1] + 1, rows[nf]
        self.n_inf = np.cumsum(np.bincount(self.ivl, minlength=self.nU)[::-1])[::-1]

    def sufmin(self, a):
        """per option: min of a over this option and the later options of the same curve"""
        A = np.where(self.valid, a[self.optc], np.inf)
        S = np.empty_like(A)
        cur = np.full(self.N, np.inf)
        for j in range(self.Lmax - 1, -1, -1):
            cur = np.where(A[:, j] <= cur, A[:, j], cur)
            S[:, j] = cur
        sm = np.empty(self.P)
        sm[self.opt[self.valid]] = S[self.valid]
        return sm

    def forced_sum(self, sm):
        b = np.bincount(self.iv0, weights=sm[self.p0], minlength=self.nU)
        return np.r_[0.0, np.cumsum(b)][:self.nU]


def piece_query(lo, hi, val, q, a, nU, bs=BS):
    """Pieces cover grid indices [lo, hi) with value val.  For every query (q, a): number of pieces
    covering q, number of those with val < a, and the sum of their values."""
    P = len(val)
    o = np.argsort(val, kind="stable")
    lo, hi, val = lo[o], hi[o], val[o]
    nb = (P + bs - 1) // bs
    blk = np.arange(P) // bs
    W = nU + 1
    fv = np.where(np.isfinite(val), val, 0.0)
    ix = np.r_[(blk + 1) * W + lo, (blk + 1) * W + hi]
    cnt = np.bincount(ix, weights=np.r_[np.ones(P), -np.ones(P)], minlength=(nb + 1) * W)
    sm = np.bincount(ix, weights=np.r_[fv, -fv], minlength=(nb + 1) * W)
    CC = np.cumsum(np.cumsum(cnt.reshape(nb + 1, W), axis=1), axis=0)
    CS = np.cumsum(np.cumsum(sm.reshape(nb + 1, W), axis=1), axis=0)
    q = np.asarray(q)
    nl = np.searchsorted(val, a, side="left")
    fb = nl // bs
    c = CC[fb, q].copy()
    s = CS[fb, q].copy()
    rem = nl - fb * bs
    for t in range(bs):
        j = fb * bs + t
        m = (t < rem) & (j < P)
        if not m.any():
            break
        jj = np.minimum(j, P - 1)
        cov = m & (lo[jj] <= q) & (q < hi[jj])
        c += cov
        s += np.where(cov, val[jj], 0.0)
    return CC[nb, q], c, s


def stage(O, a_opt, q, a):
    """Expected value-to-go from states (q, keep value a) when one run is drawn and the best action
    is taken; q = nU means no incumbent yet."""
    sm = O.sufmin(a_opt)
    q = np.asarray(q)
    a = np.asarray(a, float)
    out = np.empty(len(q))
    init = q >= O.nU
    qq = q[~init]
    if len(qq):
        F = O.forced_sum(sm)
        val = sm[O.pc_opt]
        cover, cnt, s = piece_query(O.pc_lo, O.pc_hi, val, qq, a[~init], O.nU)
        nkeep = O.n_inf[qq] + cover - cnt
        out[~init] = (F[qq] + a[~init] * nkeep + s) / O.N
    if init.any():
        out[init] = sm[O.p0].sum() / O.N
    return out


class BestOnAverage:
    """V[beta, q]: smallest expected incumbent error with beta epochs left and incumbent error U[q]
    (q = nU: no run yet), when the next runs are independent draws from the population."""

    def __init__(self, acc, budget, t_min=2):
        err = 1.0 - np.asarray(acc, np.float64)
        N, T = err.shape
        self.T, self.budget, self.t_min = T, budget, t_min
        self.O = O = Options(err, t_min)
        e1 = err[:, 0]
        V = np.empty((budget + 1, O.nU + 1))
        V[0, :O.nU] = O.U
        V[0, O.nU] = 2.0
        V[1, :O.nU] = np.minimum(O.U[:, None], e1[None, :]).mean(axis=1)
        V[1, O.nU] = e1.mean()
        q = np.arange(O.nU + 1)
        for beta in range(2, budget + 1):
            a_opt = np.where(O.k <= min(T, beta), V[np.maximum(beta - O.k, 0), O.iv], np.inf)
            V[beta] = stage(O, a_opt, q, V[beta - t_min])
        self.V = V

    def expected_regret(self, beta, best_err):
        return self.V[beta, self.O.nU] - best_err

    def horizon(self, target, horizons, best_acc):
        """Smallest horizon at which the model's expected regret reaches the target (else the last)."""
        ok = [int(h) for h in horizons if self.expected_regret(int(h), 1.0 - best_acc) <= target + 1e-12]
        return ok[0] if ok else int(horizons[-1])

    def policy(self, horizon):
        return _DPPolicy(self, horizon)


class _DPPolicy(Policy):
    """Epochs for the current run: optimal for the incumbent at `horizon` epochs, then for the full budget."""

    def __init__(self, dp, horizon):
        self.dp, self.h = dp, int(horizon)

    def stop_epoch(self, s, c):
        dp, O = self.dp, self.dp.O
        beta = self.h - s.used
        beta = beta if beta >= dp.t_min else dp.budget - s.used
        beta = max(beta, dp.t_min)
        if np.isfinite(s.prev):
            q = int(np.searchsorted(O.U, 1.0 - s.prev))
            assert q >= O.nU or O.U[q] == 1.0 - s.prev, "incumbent off the state grid"
        else:
            q = O.nU
        oc = O.optc[c]
        kk = O.k[oc]
        ok = O.valid[c] & (kk <= min(dp.T, beta)) & (O.iv[oc] < q)
        av = np.where(ok, dp.V[np.maximum(beta - kk, 0), O.iv[oc]], np.inf)
        j = int(av.argmin())
        forced = q >= O.nU or O.iv0[c] < q
        if forced or av[j] < dp.V[beta - dp.t_min, q]:
            return int(kk[j])
        return dp.t_min


def default_horizons(T):
    return np.unique(np.round(np.geomspace(20, 1000, 50) * (T / 50)).astype(int))


def nostop_target(acc, budget):
    """Expected regret of the search without early stopping over random orders of a pool: it trains
    the first budget / T configurations to T (exact expectation of the best of a random subset)."""
    N, T = acc.shape
    m = budget // T
    v = np.sort(acc.max(axis=1))
    w = np.array([math.comb(i, m - 1) for i in range(N)], float) / math.comb(N, m)
    return float(v[-1] - (w * v).sum())

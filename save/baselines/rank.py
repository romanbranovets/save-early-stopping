"""Rank-based schedulers and simple rules.

ASHA and Hyperband are re-implementations of Optuna's SuccessiveHalvingPruner and HyperbandPruner
(Optuna 4.5) replayed on recorded curves: a trial reports y(t) after every epoch and stops at the
first epoch where the pruner prunes it.  Successive Halving is synchronous: for budget b it runs the
largest bracket of the next configurations of the order whose cost fits b."""
from __future__ import annotations

import binascii
import bisect
import math
import warnings

import numpy as np

from ..search import Policy, Result, resume_stopped
from ..seeds import hyperband_label


class NoStop(Policy):
    def stop_epoch(self, s, c):
        return s.acc.shape[1]


class FixedBudget(Policy):
    def __init__(self, epochs):
        self.epochs = int(epochs)

    def stop_epoch(self, s, c):
        return self.epochs


def patience_epoch(err, n):
    """(N,) first epoch after which the run has not improved for n epochs (T if never)."""
    err = np.asarray(err, np.float64)
    T = err.shape[1]
    R = np.minimum.accumulate(err, axis=1)
    t = np.arange(T)
    imp = np.zeros_like(err, dtype=bool)
    imp[:, 0] = True
    imp[:, 1:] = err[:, 1:] < R[:, :-1]
    last = np.maximum.accumulate(np.where(imp, t[None, :], 0), axis=1)
    hit = (t[None, :] - last) >= n
    return np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, T)


class MedianStopping(Policy):
    """Stop at the first epoch t >= grace at which the run's best accuracy is below the median,
    over earlier runs, of their running-average accuracy at t; also stop after `patience` epochs
    without improvement.  The running averages of earlier runs are kept in float32."""

    def __init__(self, grace, patience):
        self.grace, self.patience = int(grace), patience

    def start(self, acc, order):
        err = 1.0 - acc
        T = acc.shape[1]
        self.R = np.minimum.accumulate(err, axis=1)
        self.RA = np.cumsum(err, axis=1) / np.arange(1, T + 1)
        self.cap = None if self.patience is None else patience_epoch(err, int(self.patience))
        self.hist = np.full((1, len(order), T), np.nan, np.float32)

    def stop_epoch(self, s, c):
        T = s.acc.shape[1]
        n = len(s.runs)
        if n:
            k_last = s.runs[-1][1]
            self.hist[0, n - 1] = np.where(np.arange(T) < k_last, self.RA[s.runs[-1][0]], np.nan)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                med = np.nanmedian(self.hist[:, :n, :], axis=1)[0]
            hit = (self.R[c] > med) & (np.arange(T) >= self.grace - 1) & np.isfinite(med)
            k = int(hit.argmax()) + 1 if hit.any() else T
        else:
            k = T
        return k if self.cap is None else min(k, int(self.cap[c]))


class ASHAPruner:
    """optuna.pruners.SuccessiveHalvingPruner(min_resource=r, reduction_factor=eta,
    min_early_stopping_rate=s), direction maximize."""

    def __init__(self, r, eta, s=0, bootstrap_count=0):
        self.r, self.eta, self.s, self.bc = int(r), int(eta), int(s), int(bootstrap_count)
        self.rungs = []                      # sorted values recorded per rung

    def run_trial(self, y):
        """Epoch at which the trial with reports y(1..T) is pruned, or T."""
        T = len(y)
        k = 0
        while True:
            step = self.r * self.eta ** (self.s + k)
            if step > T:
                return T
            v = float(y[step - 1])
            if np.isnan(v):
                return step
            while len(self.rungs) <= k:
                self.rungs.append([])
            comp = self.rungs[k]
            n = len(comp) + 1
            idx = max(n // self.eta - 1, 0)
            if n <= self.bc:
                bisect.insort(comp, v)
                return step
            promote = len(comp) - bisect.bisect_right(comp, v) <= idx      # ties promote
            bisect.insort(comp, v)
            if not promote:
                return step
            k += 1


class HyperbandPruner:
    """optuna.pruners.HyperbandPruner(min_resource=r, max_resource=R, reduction_factor=eta); the
    bracket of trial `number` is crc32("{label}_{number}") modulo the bracket budgets."""

    def __init__(self, r, R, eta, label=None):
        self.r, self.R, self.eta = int(r), int(R), int(eta)
        self.nb = math.floor(math.log(self.R / self.r, self.eta)) + 1
        self.budgets = [math.ceil(self.nb * self.eta ** (self.nb - 1 - b) / (self.nb - b)) for b in range(self.nb)]
        self.total = sum(self.budgets)
        self.sh = [ASHAPruner(self.r, self.eta, s=b) for b in range(self.nb)]
        self.label = hyperband_label() if label is None else label
        self.number = 0

    def bracket(self, number):
        n = binascii.crc32(f"{self.label}_{number}".encode()) % self.total
        for b in range(self.nb):
            n -= self.budgets[b]
            if n < 0:
                return b
        raise AssertionError

    def run_trial(self, y):
        k = self.sh[self.bracket(self.number)].run_trial(y)
        self.number += 1
        return k


class PrunerPolicy(Policy):
    def __init__(self, make):
        self.make = make

    def start(self, acc, order):
        self.pruner = self.make()

    def stop_epoch(self, s, c):
        return self.pruner.run_trial(s.acc[c])


def ASHA(r, eta):
    return PrunerPolicy(lambda: ASHAPruner(r, eta))


def Hyperband(r, eta, R):
    return PrunerPolicy(lambda: HyperbandPruner(r, R, eta))


def _bracket_stops(acc, cfgs, r, eta, t_min=2):
    """Epochs trained by every configuration of one bracket, and the best b(T) of those trained to T."""
    T = acc.shape[1]
    K = np.zeros(len(cfgs), np.int64)
    alive = np.arange(len(cfgs))
    ri = min(max(int(r), t_min), T)
    while True:
        K[alive] = ri
        if ri >= T:
            return K, float(acc[np.asarray(cfgs)[alive]].max())
        keep = max(1, len(alive) // eta)
        v = acc[np.asarray(cfgs)[alive], ri - 1]
        alive = alive[np.sort(np.argsort(-v, kind="stable")[:keep])]       # ties: earlier position
        ri = min(max(ri * eta, t_min), T)


def successive_halving(acc, order, r, eta, budget, t_min=2, final_phase=False):
    """Incumbent (budget,) of synchronous SH for one order; before any bracket fits, the running
    best of the first configuration.  Returns (best_e, K of the largest bracket, its cost)."""
    acc = np.asarray(acc, np.float64)
    T = acc.shape[1]
    costs, vals, stops = [], [], []
    n = 1
    while n <= len(order):
        K, v = _bracket_stops(acc, order[:n], r, eta, t_min)
        if K.sum() > budget:
            break
        costs.append(int(K.sum()))
        vals.append(v)
        stops.append(K)
        n += 1
    b = np.arange(1, budget + 1)
    out = np.full(budget, -np.inf)
    first = np.maximum.accumulate(acc[order[0]])
    early = b < (costs[0] if costs else budget + 1)
    out[early] = first[np.minimum(b[early], T) - 1]
    idx = np.full(budget, -1)
    for j, c in enumerate(costs):                 # cost(n) is not always monotone in n
        idx[c - 1:] = j
    ok = idx >= 0
    out[ok] = np.asarray(vals)[idx[ok]]
    K = np.zeros(len(order), np.int64)
    if stops:
        K[:len(stops[-1])] = stops[-1]
    used = costs[-1] if costs else 0
    if final_phase and stops:
        res = resume_stopped(acc, order, Result(K=K, best_e=out, used=used), budget, prev=vals[-1])
        return res.best_e, res.K, res.used
    return out, K, used

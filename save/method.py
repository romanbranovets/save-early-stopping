"""SAVE as a stopping policy, and the same rule with the true future of the run."""
from __future__ import annotations

import numpy as np

from .forecast import banks, forecast_paths
from .rule import threshold_continue, value_continue
from .search import Policy

_BANKS = {}


def cached_banks(M, T):
    if (M, T) not in _BANKS:
        _BANKS[(M, T)] = banks(M, T)
    return _BANKS[(M, T)]


class SAVE(Policy):
    """price: per epoch (cT / T).  floor: lower bound of the path noise (half the recording quantum).
    rule "value" or "threshold" (then delta).  completed_only: donors are only runs trained to T."""

    def __init__(self, price, floor, M=400, rule="value", delta=None, completed_only=False,
                 continuation="add", weighted=True):
        self.price, self.floor, self.M = float(price), float(floor), int(M)
        self.rule, self.delta = rule, delta
        self.completed_only = completed_only
        self.continuation, self.weighted = continuation, weighted

    def start(self, acc, order):
        self.T = acc.shape[1]
        self.Z, self.U, self.Us = cached_banks(self.M, self.T)
        self.runmax = np.maximum.accumulate(acc, axis=1)

    def paths(self, s, c, t):
        runs = [(j, k) for j, k in s.runs if k >= self.T or not self.completed_only]
        DY = s.acc[[j for j, _ in runs]]
        De = np.array([k for _, k in runs])
        return forecast_paths(s.acc[c, :t], DY, De, self.Z, self.U, self.Us, self.floor,
                              self.continuation, self.weighted)

    def decide(self, s, c, t, paths):
        bt = self.runmax[c, t - 1]
        u = max(s.prev, bt)
        if self.rule == "value":
            return value_continue(paths, bt, u, self.price)
        return threshold_continue(paths, u, self.delta)

    def stop_epoch(self, s, c):
        T = self.T
        if s.pos == 0:                       # nothing to compare the first run with
            return T
        for t in range(s.t_min, min(T - 1, s.t_hi) + 1):
            if t >= s.room:
                return T
            if not self.decide(s, c, t, self.paths(s, c, t)):
                return t
        return T


class PerfectForecast(SAVE):
    """The value rule with the true future of the current run in place of the paths."""

    def __init__(self, price):
        self.price = float(price)

    def start(self, acc, order):
        self.T = acc.shape[1]
        self.runmax = np.maximum.accumulate(acc, axis=1)

    def stop_epoch(self, s, c):
        T = self.T
        if s.pos == 0:
            return T
        b = self.runmax[c]
        for t in range(s.t_min, min(T - 1, s.t_hi) + 1):
            if t >= s.room:
                return T
            ks = np.arange(t + 1, T + 1)
            u_star = (b[ks - 1] - (ks - t).astype(np.float64) * self.price).max()
            if not max(s.prev, b[t - 1]) < u_star:
                return t
        return T

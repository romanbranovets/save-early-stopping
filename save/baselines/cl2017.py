"""Chandrashekaran & Lane (2017): the current curve is extrapolated from affine fits to completed
earlier runs; their conservative threshold rule, or the value rule on Gaussian paths."""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..rule import value_continue
from ..search import Policy


def fits(y_prefix, R, th1=1.0, th2=1.0):
    y = np.asarray(y_prefix, np.float64)
    R = np.asarray(R, np.float64)
    t = len(y)
    X = R[:, :t]
    E = np.exp(th2 * t)
    Srr, Sr, Syr, Sy = (X * X).sum(1), X.sum(1), X @ y, y.sum()
    a11, a12, a21, a22 = 2 * Srr + th1 * E, 2 * Sr, Sr, float(t)
    r1, r2 = 2 * Syr + th1 * E, Sy
    det = a11 * a22 - a12 * a21
    a = (r1 * a22 - a12 * r2) / det
    b = (a11 * r2 - a21 * r1) / det
    res = y[None, :] - a[:, None] * X - b[:, None]
    obj = (res * res).sum(1) + 0.5 * th1 * (1 - a) ** 2 * E
    return a * R[:, -1] + b, obj


def forecast(y_prefix, R, S_max=100):
    yh, obj = fits(y_prefix, R)
    keep = np.argsort(obj, kind="stable")[:min(S_max, len(R))]
    return float(yh[keep].mean()), float(yh[keep].std())


def p_beat(mu, sigma, u):
    if sigma <= 0:
        return float(mu >= u)
    return float(1.0 - stats.norm.cdf(u, mu, sigma))


class ChandrashekaranLane(Policy):
    def __init__(self, delta=None, rule="threshold", price=None, Z=None):
        self.delta, self.rule, self.price, self.Z = delta, rule, price, Z

    def start(self, acc, order):
        self.runmax = np.maximum.accumulate(acc, axis=1)

    def stop_epoch(self, s, c):
        acc = s.acc
        T = acc.shape[1]
        done = [j for j, k in s.runs if k == T]
        if s.pos == 0 or not done:
            return T
        R = acc[done]
        for t in range(s.t_min, T):
            if t > s.room:
                return T
            mu, sd = forecast(acc[c, :t], R)
            bt = self.runmax[c, t - 1]
            u = max(s.prev, bt)
            if self.rule == "threshold":
                cont = acc[c, t - 1] > mu or p_beat(mu, sd, u) >= self.delta
            else:
                F = mu + sd * self.Z
                frac = (np.arange(t + 1, T + 1) - t) / (T - t)
                paths = acc[c, t - 1] + frac[None, :] * (F[:, None] - acc[c, t - 1])
                cont = value_continue(paths, bt, u, self.price)
            if not cont:
                return t
        return T

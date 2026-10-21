"""The parametric forecast: e(t) = L_inf + a exp(-b t) + eps_t for the error of the current run, with
AR(1) noise eps_t = rho (s_t / s_{t-1}) eps_{t-1} + s_t sqrt(1 - rho^2) xi_t, s_t = sigma lr_t / lr_1 + q / 2,
fitted by IBIS with adaptive tempering.  Particles are [L_inf, log a, log b, log sigma, rho]."""
from __future__ import annotations

import numpy as np

from ..rule import prob_thresholds_from_paths, value_thresholds_from_paths

LOG_A = (np.log(0.3 / 100), np.log(2.0 * 100))
LOG_B = (np.log(0.005 / 100), np.log(0.1 * 100))
LOG_SIGMA = (np.log(1e-4), np.log(1.0))
RHO = (-0.99, 0.99)
N_PARTICLES, N_MOVES, N_DRAWS = 4000, 5, 400


def mean_error(x, t):
    a = np.exp(np.clip(x[:, 1], -700, 700))[:, None]
    b = np.exp(np.clip(x[:, 2], -700, 700))[:, None]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return x[:, 0][:, None] + a * np.exp(-b * np.asarray(t, float)[None, :])


class Model:
    def __init__(self, noise_scale, quantum):
        self.noise_scale = np.asarray(noise_scale, float)      # lr_t / lr_1, t = 1, 2, ...
        self.floor = 0.5 * quantum

    def sample_prior(self, n, rng):
        x = np.empty((n, 5))
        x[:, 0] = rng.uniform(0.0, 1.0, n)
        x[:, 1] = rng.uniform(*LOG_A, n)
        x[:, 2] = rng.uniform(*LOG_B, n)
        x[:, 3] = rng.uniform(*LOG_SIGMA, n)
        x[:, 4] = rng.uniform(*RHO, n)
        return x

    def log_prior(self, x):
        ok = (x[:, 0] >= 0.0) & (x[:, 0] <= 1.0)
        ok &= (x[:, 1] >= LOG_A[0]) & (x[:, 1] <= LOG_A[1])
        ok &= (x[:, 2] >= LOG_B[0]) & (x[:, 2] <= LOG_B[1])
        lp = np.where(ok, 0.0, -np.inf)
        lp = np.where((x[:, 3] >= LOG_SIGMA[0]) & (x[:, 3] <= LOG_SIGMA[1]), lp, -np.inf)
        return np.where((x[:, 4] >= RHO[0]) & (x[:, 4] <= RHO[1]), lp, -np.inf)

    def loglik_terms(self, x, t, y):
        """(N, len(t)) conditional log densities; column 0 is scored from the marginal, so a window
        must include the observation before the one being added."""
        mu = mean_error(x, t)
        s = np.exp(np.clip(x[:, 3], -700, 700))[:, None]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            r = y[None, :] - mu
            rho = x[:, 4][:, None]
            sdt = s * self.noise_scale[np.asarray(t, int) - 1][None, :] + self.floor
            e = r.copy()
            e[:, 1:] = r[:, 1:] - rho * (sdt[:, 1:] / sdt[:, :-1]) * r[:, :-1]
            sd = sdt * np.sqrt(np.maximum(1.0 - rho ** 2, 1e-12))
            sd[:, 0] = sdt[:, 0]
            z = e / sd
            out = -0.5 * (np.log(2 * np.pi) + 2 * np.log(sd) + z * z)
        return np.nan_to_num(out, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)


def _systematic(w, rng):
    n = len(w)
    u = (rng.random() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(w), u).clip(0, n - 1)


def _norm_w(logw):
    m = np.max(logw)
    if not np.isfinite(m):
        return np.full(len(logw), 1.0 / len(logw))
    w = np.exp(logw - m)
    s = w.sum()
    return w / s if s > 0 else np.full(len(logw), 1.0 / len(logw))


def _move(model, x, t, y, rng, n_moves, power=1.0):
    """Random-walk MH targeting the (partially tempered) posterior; scale from the particle cloud."""
    N, d = x.shape
    cov = np.cov(x.T) + 1e-12 * np.eye(d)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.maximum(np.diag(cov), 1e-24)))
    scale = 2.38 / np.sqrt(d)

    def total(terms):
        return terms.sum(axis=1) if power >= 1.0 else terms[:, :-1].sum(axis=1) + power * terms[:, -1]

    cur_ll = total(model.loglik_terms(x, t, y))
    cur_lp = model.log_prior(x)
    for _ in range(n_moves):
        prop = x + scale * (rng.standard_normal((N, d)) @ L.T)
        prop_lp = model.log_prior(prop)
        alive = np.isfinite(prop_lp)
        prop_ll = np.full(N, -np.inf)
        if alive.any():
            prop_ll[alive] = total(model.loglik_terms(prop[alive], t, y))
        acc = np.log(rng.random(N)) < (prop_ll + prop_lp) - (cur_ll + cur_lp)
        x = np.where(acc[:, None], prop, x)
        cur_ll = np.where(acc, prop_ll, cur_ll)
        cur_lp = np.where(acc, prop_lp, cur_lp)
    return x


def ibis(model, y, seed, checkpoints, n_particles=N_PARTICLES, n_moves=N_MOVES, ess_frac=0.5, max_temper=60):
    """Posterior particles and weights after the first k observations, for every k in checkpoints."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, len(y) + 1, dtype=float)
    y = np.asarray(y, float)
    N = n_particles
    x = model.sample_prior(N, rng)
    logw = np.zeros(N)
    out = {}
    for k in range(len(y)):
        lo = max(0, k - 1)
        inc = model.loglik_terms(x, t[lo:k + 1], y[lo:k + 1])[:, -1]
        phi, n_t = 0.0, 0
        while True:                                   # tempering: keep the ESS at the threshold
            def ess_at(dl, logw=logw, inc=inc):
                ww = _norm_w(logw + dl * inc)
                return 1.0 / np.sum(ww ** 2)
            dl = 1.0 - phi
            if ess_at(dl) < ess_frac * N and n_t < max_temper:
                a, b = 0.0, dl
                for _ in range(30):
                    m_ = 0.5 * (a + b)
                    a, b = (m_, b) if ess_at(m_) >= ess_frac * N else (a, m_)
                dl = max(a, 1e-12)
            logw = logw + dl * inc
            phi += dl
            n_t += 1
            if phi >= 1.0 - 1e-12:
                break
            x = x[_systematic(_norm_w(logw), rng)]
            logw = np.zeros(N)
            x = _move(model, x, t[:k + 1], y[:k + 1], rng, n_moves, power=phi)
            inc = model.loglik_terms(x, t[lo:k + 1], y[lo:k + 1])[:, -1]
        w = _norm_w(logw)
        if k + 1 in checkpoints:
            out[k + 1] = (x.copy(), w.copy())
        if 1.0 / np.sum(w ** 2) < ess_frac * N and k < len(y) - 1:
            x = x[_systematic(w, rng)]
            logw = np.zeros(N)
            x = _move(model, x, t[:k + 1], y[:k + 1], rng, n_moves)
    return out


def future_errors(model, draws, y, t, T_all, rng):
    """(n, T_all - t) error paths at epochs t+1..T_all from posterior draws after t observations."""
    mu = mean_error(draws, np.arange(1, T_all + 1, dtype=float))
    sig = np.exp(draws[:, 3])
    rho = draws[:, 4]
    sd = sig[:, None] * model.noise_scale[None, :] + model.floor
    r = y[t - 1] - mu[:, t - 1]
    out = np.empty((len(draws), T_all - t))
    for j, k in enumerate(range(t + 1, T_all + 1)):
        r = rho * (sd[:, k - 1] / sd[:, k - 2]) * r + sd[:, k - 1] * np.sqrt(np.maximum(1 - rho ** 2, 1e-12)) \
            * rng.standard_normal(len(draws))
        out[:, j] = mu[:, k - 1] + r
    return out


def tables_for_curve(err, acc, noise_scale, quantum, seeds, decisions, T, prices=(), deltas=(), T_all=None):
    """Decision tables of one curve.

    err          observed error series fed to the model (may be longer than T)
    acc          (T,) the accuracy the search sees (for b(t))
    noise_scale  lr_t / lr_1 for t = 1..T_all
    T_all        length of the simulated paths (paths are cut to epochs t+1..T)
    Returns val (T + 1, len(prices)) and thr (T + 1, len(deltas)), NaN outside `decisions`."""
    T_all = len(err) if T_all is None else T_all
    model = Model(noise_scale, quantum)
    post = ibis(model, err, seeds[0], set(decisions))
    rng = np.random.default_rng(seeds[1])
    bt = np.maximum.accumulate(acc)
    val = np.full((T + 1, len(prices)), np.nan)
    thr = np.full((T + 1, len(deltas)), np.nan)
    for t in decisions:
        x, w = post[t]
        draws = x[rng.choice(len(x), size=N_DRAWS, replace=True, p=w)]
        paths = 1.0 - future_errors(model, draws, np.asarray(err, float), t, T_all, rng)[:, :T - t]
        if len(deltas):
            thr[t] = prob_thresholds_from_paths(paths, deltas)
        if len(prices):
            val[t] = value_thresholds_from_paths(paths, bt[t - 1], prices)
    return val.astype(np.float32), thr.astype(np.float32)

"""FT-PFN (ifBO, Rakotoarison et al. 2024) as the forecast of a stopping rule.  Needs torch, ifbo 0.4.1
and the checkpoint bopfn_broken_unisep_1000curves_10params_2M.pt.  The context is every observed epoch
of the search, a point is (position + 1, epoch / T, hyperparameters in [0, 1]) with its accuracy."""
from __future__ import annotations

import numpy as np

from ..search import Policy
from .lcpfn import _torch, value_thresholds_bars


def load(checkpoint, device="cuda"):
    torch = _torch()
    from ifbo.surrogate import FTPFN
    m = FTPFN.__new__(FTPFN)
    torch.nn.Module.__init__(m)
    m.model = torch.load(str(checkpoint), map_location=device).eval()
    m.device = torch.device(device)
    m.version = "0.0.1"
    return m


def forward_batch(m, xtr, ytr, pad, xte):
    """The network on a batch of padded contexts.  xtr (Ntr, B, F), ytr (Ntr, B), pad (B, Ntr) True =
    padding, xte (Nte, B, F).  Returns logits (Nte, B, nb)."""
    torch = _torch()
    net = m.model
    with torch.no_grad():
        ntr = xtr.shape[0]
        x = net.encoder(torch.cat([xtr, xte], 0))
        y = net.y_encoder(ytr.unsqueeze(-1))
        src = net.pos_encoder(torch.cat([x[:ntr] + y, x[ntr:]], 0))
        for L in net.transformer_encoder.layers:
            tr = src[:ntr]
            left = L.self_attn(tr, tr, tr, key_padding_mask=pad)[0]
            right = L.self_attn(src[ntr:], tr, tr, key_padding_mask=pad)[0]
            src = L.norm1(src + L.dropout1(torch.cat([left, right], 0)))
            src = L.norm2(src + L.dropout2(L.linear2(L.dropout(L.activation(L.linear1(src))))))
        return net.decoder_dict["standard"](src[ntr:])


def prob_exceed(m, logits, y_best):
    """P(y > y_best) per query from the bar distribution (BarDistribution.pi)."""
    torch = _torch()
    b = m.model.criterion.borders.to(logits.device)
    p = torch.softmax(logits, -1)
    w = b[1:] - b[:-1]
    f = 1.0 - ((y_best[None, :, None] - b[:-1]) / w).clamp(0.0, 1.0)
    return (p * f).sum(-1)


def context(acc, hp, runs, pos, c, t, T, queries):
    """Context points of the earlier runs and of the current run's prefix, and the query points."""
    xs, ys = [], []
    for j, (cj, k) in enumerate(runs):
        e = np.arange(1, k + 1)
        xs.append(np.c_[np.full(k, j + 1.0), e / T, np.repeat(hp[cj][None], k, 0)])
        ys.append(acc[cj, :k])
    e = np.arange(1, t + 1)
    xs.append(np.c_[np.full(t, pos + 1.0), e / T, np.repeat(hp[c][None], t, 0)])
    ys.append(acc[c, :t])
    xq = np.c_[np.full(len(queries), pos + 1.0), np.asarray(queries) / T, np.repeat(hp[c][None], len(queries), 0)]
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32), xq.astype(np.float32)


def logits_for(m, xtr, ytr, xq):
    torch = _torch()
    dev = m.device
    return forward_batch(m, torch.tensor(xtr, device=dev)[:, None], torch.tensor(ytr, device=dev)[:, None],
                         torch.zeros(1, len(xtr), dtype=torch.bool, device=dev), torch.tensor(xq, device=dev)[:, None])


class FTPFNPolicy(Policy):
    def __init__(self, model, hp, rule="threshold", delta=None, price=None):
        self.m, self.hp = model, np.asarray(hp, np.float32)
        self.rule, self.delta, self.price = rule, delta, price

    def start(self, acc, order):
        self.runmax = np.maximum.accumulate(acc, axis=1)

    def stop_epoch(self, s, c):
        torch = _torch()
        T = s.acc.shape[1]
        for t in range(s.t_min, min(T - 1, s.t_hi) + 1):
            if t >= s.room:
                return T
            q = np.minimum(t + 1 + np.arange(T - 2), T)
            xtr, ytr, xq = context(s.acc, self.hp, s.runs, s.pos, c, t, T, q)
            lg = logits_for(self.m, xtr, ytr, xq)
            yb = max(s.prev, self.runmax[c, t - 1])
            if self.rule == "threshold":
                pe = prob_exceed(self.m, lg, torch.tensor([yb], device=lg.device, dtype=torch.float32))
                stop = float(pe.max(0).values[0]) <= self.delta
            else:
                u = value_thresholds_bars(lg, self.m.model.criterion.borders, [self.price]).cpu().numpy()
                stop = yb >= u[0, 0]
            if stop:
                return t
        return T


def final_best_pit(m, acc, hp, runs, pos, c, t, T):
    """PIT and 10/50/90 % quantiles of the final best accuracy of run c under FT-PFN's marginals,
    combined comonotonically (the quantile of the maximum is the maximum of the quantiles)."""
    torch = _torch()
    xtr, ytr, xq = context(acc, hp, runs, pos, c, t, T, np.arange(t + 1, T + 1))
    lg = logits_for(m, xtr, ytr, xq)[:, 0].double()
    b = m.model.criterion.borders.double()
    p = torch.softmax(lg, -1)
    cum = torch.cat([torch.zeros(len(p), 1, dtype=p.dtype, device=p.device), torch.cumsum(p, -1)], -1)

    def cdf(v):
        j = torch.clamp(torch.searchsorted(b, torch.tensor([v], dtype=b.dtype, device=b.device), right=True) - 1,
                        0, len(b) - 2)
        f = ((v - b[j]) / (b[j + 1] - b[j])).clamp(0, 1)
        return (cum[:, j] + p[:, j] * f).squeeze(-1)

    def qmax(u):
        j = torch.clamp(torch.searchsorted(cum, torch.full((len(p), 1), u, dtype=p.dtype, device=p.device)) - 1,
                        0, len(b) - 2)
        pj = p.gather(1, j)
        x = b[j] + (b[j + 1] - b[j]) * ((u - cum.gather(1, j)) / pj.clamp(min=1e-300)).clamp(0, 1)
        return float(x.max())

    rb = float(np.max(acc[c, :t]))
    truth = float(np.max(acc[c]))
    pit = float(cdf(truth).min()) if truth > rb else 0.5 * float(cdf(rb).min())
    lo, med, hi = (max(rb, qmax(u)) for u in (0.1, 0.5, 0.9))
    return pit, lo, med, hi

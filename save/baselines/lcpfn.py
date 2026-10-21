"""LC-PFN (Adriaensen et al. 2023) decision tables for the threshold rule and the value rule.  Needs
torch and lcpfn; TF32 is switched off."""
from __future__ import annotations

import numpy as np


def _torch():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def icdf_many(crit, logits, left):
    """BarDistribution.icdf for several left probabilities at once (same float32 operations)."""
    torch = _torch()
    probs = logits.softmax(-1)
    cum = torch.cumsum(probs, -1)
    lp = torch.tensor(left, dtype=cum.dtype, device=cum.device)
    lpx = lp.expand(*cum.shape[:-1], len(left)).contiguous()
    idx = torch.searchsorted(cum, lpx).clamp(0, cum.shape[-1] - 1)
    cum0 = torch.cat([torch.zeros(*cum.shape[:-1], 1, device=cum.device), cum], -1)
    rest = lpx - cum0.gather(-1, idx)
    lb, rb = crit.borders[idx], crit.borders[idx + 1]
    return lb + (rb - lb) * rest / probs.gather(-1, idx)


def value_thresholds_bars(logits, borders, prices):
    """torch version of save.rule.value_thresholds_from_bars over a block: logits (L, n, nb) -> (n, C)."""
    torch = _torch()
    p = torch.softmax(logits.double(), -1)
    b = borders.double()
    L, n, nb = p.shape
    w = b[1:] - b[:-1]
    mid = 0.5 * (b[:-1] + b[1:])
    z = torch.zeros(L, n, 1, dtype=p.dtype, device=p.device)
    T0 = torch.cat([torch.flip(torch.cumsum(torch.flip(p, [-1]), -1), [-1]), z], -1)
    T1 = torch.cat([torch.flip(torch.cumsum(torch.flip(p * mid, [-1]), -1), [-1]), z], -1)
    Gb = torch.cummin((T1 - b * T0).clamp(min=0.0), -1).values
    cost = torch.arange(1, L + 1, dtype=p.dtype, device=p.device)
    pr = torch.tensor(prices, dtype=p.dtype, device=p.device)
    lev = (cost[:, None] * pr[None, :])[:, None, :].expand(L, n, len(prices)).contiguous()
    cnt = torch.searchsorted((-Gb).contiguous(), (-lev).contiguous(), right=False)
    below = cnt == 0
    j = (cnt - 1).clamp(0, nb - 1)
    d = lev - Gb.gather(-1, j + 1)
    a = p.gather(-1, j) / (2.0 * w[j])
    t0 = T0.gather(-1, j + 1)
    xx = 2.0 * d / (t0 + torch.sqrt(t0 * t0 + 4.0 * a * d))
    xx = torch.where(d > 0, xx, torch.zeros_like(xx))
    u = torch.where(below, b[0] - (lev - Gb[..., :1]), b[j + 1] - xx)
    return u.max(dim=0).values


def load_model(device="cuda"):
    import lcpfn
    net = lcpfn.LCPFN().model.eval().to(device)
    assert type(net.criterion).__name__ == "FullSupportBarDistribution"
    return net


def tables(acc, decisions, deltas=(), prices=(), device="cuda", block=1000, sub=250, net=None):
    """acc (N, T) accuracies.  Returns thr (T + 1, N, len(deltas)) and val (T + 1, N, len(prices)),
    float32, NaN outside `decisions`."""
    torch = _torch()
    net = load_model(device) if net is None else net
    crit = net.criterion
    A = np.asarray(acc, np.float32)
    N, T = A.shape
    thr = np.full((T + 1, N, len(deltas)), np.nan, np.float32)
    val = np.full((T + 1, N, len(prices)), np.nan, np.float32)
    left = [1.0 - d for d in deltas]
    for b0 in range(0, N, block):
        sl = slice(b0, min(b0 + block, N))
        n = sl.stop - sl.start
        Y = torch.tensor(A[sl].T.copy(), device=device)
        x = torch.arange(1, T + 1, dtype=torch.float32)[:, None, None].expand(T, n, 1).to(device)
        for t in decisions:
            with torch.no_grad():
                lg = net((x, Y[:t]), single_eval_pos=t)                     # (T - t, n, nb)
                if len(deltas):
                    thr[t, sl] = icdf_many(crit, lg, left).max(dim=0).values.cpu().numpy()
                if len(prices):
                    for s in range(0, n, sub):
                        val[t, sl.start + s:sl.start + min(s + sub, n)] = value_thresholds_bars(
                            lg[:, s:s + sub], crit.borders, list(prices)).cpu().numpy()
    return thr, val

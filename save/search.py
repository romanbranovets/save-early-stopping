"""Sequential search on recorded learning curves, with the budget and the final phase.

acc (N, T) holds accuracy curves, epoch t in column t - 1; best_e[b - 1] is the incumbent after b epochs."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class State:
    """What a policy may read when it decides about the run at position `pos`."""
    acc: np.ndarray
    order: np.ndarray
    pos: int = 0
    used: int = 0
    prev: float = -np.inf                         # incumbent before the current run
    room: int = 0                                 # budget left when the run starts
    runs: list = field(default_factory=list)      # (config, epochs trained) of the earlier runs
    t_min: int = 2
    t_hi: int = 0


class Policy:
    """Base class.  stop_epoch returns the number of epochs for the current run; the search clips
    it to [t_min, T] and to the budget left.  Decisions at epochs t >= state.room do not matter
    (the run is cut by the budget anyway), so a policy may stop looking there."""

    def start(self, acc, order):
        pass

    def stop_epoch(self, s: State, c: int) -> int:
        raise NotImplementedError


@dataclass
class Result:
    K: np.ndarray            # (slots,) epochs trained per position, 0 = never started
    best_e: np.ndarray       # (budget,) incumbent after every epoch
    used: int
    resumed: int = 0         # epochs spent in the final phase


def run_search(acc, order, policy, budget, slots=None, t_min=2, t_hi=None, final_phase=False):
    acc = np.asarray(acc, np.float64)
    T = acc.shape[1]
    slots = len(order) if slots is None else min(slots, len(order))
    B = int(budget)
    runmax = np.maximum.accumulate(acc, axis=1)
    s = State(acc=acc, order=np.asarray(order), t_min=t_min, t_hi=T - 1 if t_hi is None else t_hi)
    K = np.zeros(slots, np.int64)
    best_e = np.full(B, -np.inf)
    policy.start(acc, s.order)
    for pos in range(slots):
        if s.used >= B:
            break
        c = int(s.order[pos])
        s.pos, s.room = pos, B - s.used
        k = int(policy.stop_epoch(s, c))
        k = min(max(k, t_min), T)
        k = min(k, max(B - s.used, 1))
        e = s.used + np.arange(k)
        m = e < B
        best_e[e[m]] = np.maximum(s.prev, runmax[c, :k][m])
        K[pos] = k
        s.prev = max(s.prev, runmax[c, k - 1])
        s.runs.append((c, k))
        s.used += k
    best_e = np.maximum.accumulate(best_e)
    res = Result(K=K, best_e=best_e, used=s.used)
    if final_phase:
        res = resume_stopped(acc, order, res, B)
    return res


def resume_stopped(acc, order, res, budget, prev=None):
    """The final phase.  prev: incumbent when the phase starts (default: best_e at `used`)."""
    T = acc.shape[1]
    K = res.K.copy()
    used = int(res.used)
    if not ((K > 0).all() and used < budget):
        return res
    order = np.asarray(order)[:len(K)]
    if prev is None:
        prev = res.best_e[used - 1]
    b = np.array([acc[int(order[j]), :K[j]].max() for j in range(len(K))])
    cand = [j for j in sorted(range(len(K)), key=lambda j: (-b[j], j)) if K[j] < T]
    tail = np.full(budget, -np.inf)
    u, best = used, -np.inf
    for j in cand:
        if u >= budget:
            break
        c = int(order[j])
        while K[j] < T and u < budget:
            best = max(best, float(acc[c, K[j]]))
            tail[u] = best
            K[j] += 1
            u += 1
    best_e = res.best_e.copy()
    best_e[used:] = np.maximum(prev, np.maximum.accumulate(tail[used:]))
    return Result(K=K, best_e=best_e, used=u, resumed=u - used)

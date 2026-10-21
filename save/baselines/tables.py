"""Decision tables thr (T + 1, N): a run stops at the first decision epoch t with
max(incumbent before the run, b(t)) >= thr[t, run].  Kept in float32."""
from __future__ import annotations

import pathlib

import numpy as np

from ..rule import table_stop_epoch
from ..search import Policy


class TablePolicy(Policy):
    def __init__(self, thr):
        self.thr = np.asarray(thr)

    def stop_epoch(self, s, c):
        return table_stop_epoch(s.acc[c], s.prev, self.thr[:, c], s.t_min, s.t_hi)


def save_table(path, thr, **meta):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, thr=np.asarray(thr, np.float32), **meta)


def load_table(path):
    return np.load(path)["thr"]

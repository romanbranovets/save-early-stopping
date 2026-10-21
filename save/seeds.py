"""Seeds of every random draw.  The values used for the paper are in configs/seeds.json; anything not
listed there gets a seed from stable_seed."""
from __future__ import annotations

import functools
import hashlib
import json
import pathlib

SEEDS_FILE = pathlib.Path(__file__).resolve().parents[1] / "configs" / "seeds.json"


def stable_seed(*parts) -> int:
    blob = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") % (2**32)


@functools.cache
def table() -> dict:
    return json.loads(SEEDS_FILE.read_text())


def _get(*keys):
    d = table()
    for k in keys:
        if not isinstance(d, dict) or str(k) not in d:
            return None
        d = d[str(k)]
    return d


def _or(value, *fallback):
    return value if value is not None else stable_seed(*fallback)


def bank_seed(M, T):
    return _or(_get("forecast_bank", f"{M}x{T}"), "bank", M, T)


def stage_bank_seed(M, T):
    return _or(_get("forecast_stage_bank", f"{M}x{T}"), "stage-bank", M, T)


def orders_seed(bench, split, unit):
    return _or(_get("orders", bench, split, unit), "orders", bench, split, unit)


def bootstrap_seed(bench, split=None, unit=None):
    s = _get("bootstrap", bench)
    if isinstance(s, dict):
        s = _get("bootstrap", bench, split, unit)
    return _or(s, "bootstrap", bench, split, unit)


def pool_seed(bench, unit, P, s):
    v = _get("pools", bench, unit, P)
    return v[s] if v is not None and s < len(v) else stable_seed("pool", bench, unit, P, s)


def pool_bootstrap_seed(bench, unit, P):
    return _or(_get("pool_bootstrap", bench, unit, P), "pool-bootstrap", bench, unit, P)


def parametric_seeds(bench, unit, config):
    """(particle seed, draw seed) of the parametric forecast of one curve; config is the index of the
    configuration in the benchmark (NAS-Bench-201: architecture index, LCBench: configuration id)."""
    if bench == "db5":
        v = _get("parametric_seeds_db5", unit)
        if v is not None:
            return tuple(v[config])
    lab = _get("parametric_seed_labels", bench)
    if lab is None:
        return stable_seed("particles", bench, unit, config), stable_seed("draws", bench, unit, config)
    key = [unit, *lab["after_unit"], config, *lab["after_config"]]
    return stable_seed(lab["particles"], *key), stable_seed(lab["draws"], *key)


def hyperband_label():
    """label hashed with the trial number to assign Hyperband brackets"""
    return _get("hyperband_bracket_label") or "study"

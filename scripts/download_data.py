"""Download the raw benchmark files and prepare the arrays the experiments read.

    python scripts/download_data.py nb201 lcbench [--raw data/raw] [--workers 8]

NAS-Bench-201: NATS-tss-v1_0-3ffb9-simple.tar (NATS-Bench topology search space, 1.1 GB, from the
link in the NATS-Bench README), unpacked and read once into data/nb201_<task>.npz.
LCBench: data_2k_lw.zip from figshare (340 MB), read once into data/lcbench/<dataset>.npz.
A file that already exists is not downloaded again; checksums are verified.
NinaPro DB5 is not downloaded here: the recorded curves are in data/db5_curves.csv.gz, and
scripts/db5_train.py downloads the raw signals to record them again.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import tarfile
import urllib.request
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from save.data import DATA, lcbench, nb201

SOURCES = {
    "nb201": dict(file="NATS-tss-v1_0-3ffb9-simple.tar",
                  url="https://drive.usercontent.google.com/download?id=17_saCsj_krKjlCBLOJEpNtzPXArMCqxU"
                      "&export=download&confirm=t",
                  sha256="580fd8f3425fed9f495640b0f12ccb7744f89dec7aff263bd75aed37ab0b8bb6"),
    "lcbench": dict(file="data_2k_lw.zip", url="https://ndownloader.figshare.com/files/21188598",
                    md5="26c8493006358a15ce12041c0bfc12a2"),
}


def digest(path, kind):
    h = hashlib.new(kind)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(src, raw):
    path = raw / src["file"]
    if not path.exists():
        raw.mkdir(parents=True, exist_ok=True)
        print(f"downloading {src['url']} -> {path}", flush=True)
        tmp = path.with_name(path.name + ".part")
        req = urllib.request.Request(src["url"], headers={"User-Agent": "save-early-stopping"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            f.writelines(iter(lambda: r.read(1 << 20), b""))
        tmp.replace(path)
    kind = "sha256" if "sha256" in src else "md5"
    if digest(path, kind) != src[kind]:
        raise SystemExit(f"{path}: {kind} does not match; delete the file and run again")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("benchmarks", nargs="+", choices=sorted(SOURCES))
    ap.add_argument("--raw", default=str(DATA / "raw"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    raw = pathlib.Path(a.raw)
    DATA.mkdir(parents=True, exist_ok=True)
    if "nb201" in a.benchmarks:
        tar = fetch(SOURCES["nb201"], raw)
        records = raw / "NATS-tss-v1_0-3ffb9-simple"
        if not records.exists():
            with tarfile.open(tar) as t:
                try:
                    t.extractall(raw, filter="data")
                except TypeError:                   # Python without extraction filters
                    t.extractall(raw)
        nb201.prepare(records, a.workers)
        print("NAS-Bench-201 ready", flush=True)
    if "lcbench" in a.benchmarks:
        z = fetch(SOURCES["lcbench"], raw)
        js = raw / "data_2k_lw.json"
        if not js.exists():
            with zipfile.ZipFile(z) as f:
                name = next(n for n in f.namelist() if n.endswith("data_2k_lw.json"))
                js.write_bytes(f.read(name))
        lcbench.prepare(js)
        print("LCBench ready", flush=True)


if __name__ == "__main__":
    main()

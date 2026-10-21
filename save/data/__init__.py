import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = pathlib.Path(os.environ.get("SAVE_DATA", REPO / "data"))

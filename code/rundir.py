"""
Run folders. Every run gets its own runs/<kind>-<YYYYmmdd-HHMMSS>[-tag]/ so nothing is overwritten.

kinds: prime (primed encoder), train (fine-tuned classifier), baseline (frozen encoder + logistic regression)
"""

import json
import sys
from datetime import datetime

from config import RUNS


def new_run(kind, tag=None):
    name = f"{kind}-{datetime.now():%Y%m%d-%H%M%S}" + (f"-{tag}" if tag else "")
    path = RUNS / name
    path.mkdir(parents=True)
    return path


def latest_run(kind, must_have):
    """Newest runs/<kind>-* folder that contains must_have (e.g. "model.pt")."""
    runs = sorted(p for p in RUNS.glob(f"{kind}-*") if (p / must_have).exists())
    if not runs:
        sys.exit(f"No finished {kind} run in {RUNS}.")
    return runs[-1]


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

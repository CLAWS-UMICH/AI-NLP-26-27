"""Diversity gate (plan §4.4): within-label mean pairwise cosine (ruler = all-mpnet-base-v2, pinned).
The frozen test is never read here. Usage: python -m datagen.diversity"""
import json
from collections import defaultdict
import numpy as np
from .config import GEN, DIVERSITY_TARGET
from .common import read_jsonl, embed, norm

def within_label(rows):
    by = defaultdict(list)
    for r in rows: by[r["label"]].append(norm(r["text"]))
    out = {}
    for lab, ts in by.items():
        ts = sorted(set(ts))
        if len(ts) < 2: continue
        E = embed(ts); S = E @ E.T
        iu = np.triu_indices(len(ts), 1)
        out[lab] = float(S[iu].mean())
    return out

def report(train_rows):
    base = [r for r in train_rows if r.get("source") != "tts_roundtrip"]
    wl = within_label(base)
    mean = float(np.mean(list(wl.values())))
    worst = sorted(wl.items(), key=lambda kv: -kv[1])[:10]
    return {"within_label_mean": round(mean, 4), "target": DIVERSITY_TARGET, "pass": mean < DIVERSITY_TARGET,
            "labels_over_0.55": sum(v > 0.55 for v in wl.values()), "labels_over_0.62": sum(v > 0.62 for v in wl.values()),
            "per_label": {k: round(v, 4) for k, v in sorted(wl.items())},
            "diversity_worst": [[k, round(v, 3)] for k, v in worst]}

if __name__ == "__main__":
    r = report(read_jsonl(GEN / "train.jsonl"))
    print(json.dumps({k: v for k, v in r.items() if k != "per_label"}, indent=1))

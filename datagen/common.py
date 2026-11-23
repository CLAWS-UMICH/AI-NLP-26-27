"""Helpers shared by the loop scripts: IO, normalisation, embeddings, MinHash."""
import json, re, string
from functools import lru_cache
from pathlib import Path
import numpy as np
from .config import FILLER_RE, RULER

def read_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.open()] if p.exists() else []

def write_jsonl(p, rows, mode="w"):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, mode) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

_PUNCT = re.compile(r"[^\w\s']")
def norm(t):
    """Model-side normalisation: lowercase, strip punctuation, collapse spaces."""
    t = _PUNCT.sub(" ", t.lower().replace("’", "'"))
    return re.sub(r"\s+", " ", t).strip()

def strip_filler(t):
    return re.sub(r"\s+", " ", FILLER_RE.sub(" ", norm(t))).strip()

def shingles(t):
    w = strip_filler(t).split()
    return set(w) | {f"{a} {b}" for a, b in zip(w, w[1:])}

def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))

@lru_cache(maxsize=1)
def ruler():
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer(RULER[0], revision=RULER[1], device=dev)

def embed(texts, batch_size=128):
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    return ruler().encode(list(texts), batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

def minhash(sh, num_perm=128):
    from datasketch import MinHash
    m = MinHash(num_perm=num_perm)
    m.update_batch([s.encode() for s in sh] or [b"<empty>"])
    return m

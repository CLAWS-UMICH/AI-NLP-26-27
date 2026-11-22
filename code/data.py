"""
Loading and splitting data. Shared by every script so they all see the same rows and the same split.

Row format (JSONL, one per line):
    {"text": "...", "intent": "label", "slot": "name", "source": "...", "group": "..."}
- intent:  exactly one label from intents.json
- slot:    optional, only for the five name intents (the place or task named in the text)
- group:   written by split_seed.py and inherited by every row generated from a seed row. Rows that share a
           group always land on the same side of every split, otherwise val/test accuracy is inflated.
- style:   optional free tag (e.g. "astronaut", "normal"); train.py reports accuracy per style.
- source:  where the row came from: seed, tmpl, para:bt, asr:piper, clinc, llm, rec:a, rec:b
- parent:  id of the seed row a generated row came from (split_seed.py gives every seed row an id)
"""

import json
import re
import sys

from sklearn.model_selection import StratifiedGroupKFold

from config import INTENTS_FILE, PRIMER_FILE, SEED, TEST_FILE, TRAIN_FILE, UNHANDLED, VAL_SIZE


def norm(text):
    """Lowercase, strip punctuation, collapse spaces. Used for duplicate and leakage checks."""
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def load_labels():
    return [x["intent"] for x in json.loads(INTENTS_FILE.read_text(encoding="utf-8"))]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mix_bucket(row):
    """Which MIX_TARGETS bucket a row counts toward."""
    if row["intent"] == UNHANDLED:
        return "unhandled"
    source = row.get("source", "seed")
    if source in ("seed", "tmpl"):
        return "seed+tmpl"
    return source.split(":")[0]  # para:bt -> para, asr:piper -> asr, llm -> llm


def load_train():
    if not TRAIN_FILE.exists():
        sys.exit(f"No {TRAIN_FILE}. Run python code/generate.py (or write it by hand).")
    return read_jsonl(TRAIN_FILE)


def load_test(train_rows):
    """Handwritten test rows. Quits if any of them also appear in training (after normalising)."""
    if not TEST_FILE.exists():
        sys.exit(f"No {TEST_FILE}. Write it by hand, same format as train.jsonl.")
    rows = read_jsonl(TEST_FILE)
    seen = {norm(r["text"]) for r in train_rows}
    overlap = [r["text"] for r in rows if norm(r["text"]) in seen]
    if overlap:
        sys.exit("These test.jsonl sentences are also in train.jsonl, remove one copy:\n  " + "\n  ".join(overlap))
    return rows


def split(rows):
    """Stratified train/val split, grouped by row["group"] so augmentations of one seed row don't leak into val.
    Rows without a group are their own group. Val is ~1/round(1/VAL_SIZE) of the rows (1/7 = 14.3% at 0.15)."""
    labels = [r["intent"] for r in rows]
    groups = [r.get("group", f"row{i}") for i, r in enumerate(rows)]
    splitter = StratifiedGroupKFold(n_splits=max(2, round(1 / VAL_SIZE)), shuffle=True, random_state=SEED)
    train_idx, val_idx = next(splitter.split(rows, labels, groups=groups))
    return [rows[i] for i in train_idx], [rows[i] for i in val_idx]


def load_primer():
    """(anchor, positive) pairs from astronaut_primer.json: a JSON list of {"anchor": ..., "positive": ...}."""
    if not PRIMER_FILE.exists():
        sys.exit(f"No {PRIMER_FILE}. Add the astronaut primer first.")
    items = json.loads(PRIMER_FILE.read_text(encoding="utf-8"))
    pairs = [(x["anchor"].strip(), x["positive"].strip()) for x in items if x.get("anchor") and x.get("positive")]
    pairs = list(dict.fromkeys(pairs))  # a repeated pair is a false negative in its own batch and leaks into held-out
    if not pairs:
        sys.exit(f"{PRIMER_FILE.name} has no usable anchor/positive pairs.")
    return pairs

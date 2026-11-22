"""
Check a data file before training: valid JSON, known labels, slot is valid, no duplicates,
enough rows per intent, and (for train files) no overlap with the handwritten test set.

Usage:
    python code/validate.py data/train.jsonl
    python code/validate.py data/test.jsonl --min 5
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from config import SLOT_INTENTS, TEST_FILE
from data import load_labels, norm, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--min", type=int, default=30, help="minimum rows per intent")
    args = parser.parse_args()

    labels = set(load_labels())
    seen, per_intent, problems = {}, Counter(), []

    for n, line in enumerate(open(args.path, encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            text, intent = row["text"].strip(), row["intent"]
        except (ValueError, KeyError, TypeError, AttributeError):
            problems.append(f'line {n}: not a valid row, expected {{"text": "...", "intent": "label"}}')
            continue
        if not text:
            problems.append(f"line {n}: empty text")
        if intent not in labels:
            problems.append(f"line {n}: bad intent {intent!r} (must be one label from intents.json)")
            continue
        slot = row.get("slot")
        if slot is not None:
            if intent not in SLOT_INTENTS:
                problems.append(f"line {n}: {intent} takes no slot but got {slot!r}")
            elif norm(slot) not in norm(text):
                problems.append(f"line {n}: slot {slot!r} not found in text")
        key = norm(text)
        if key in seen:
            problems.append(f"line {n}: duplicate of line {seen[key]} '{text}'")
        seen.setdefault(key, n)
        per_intent[intent] += 1

    for label in sorted(labels):
        if per_intent[label] < args.min:
            problems.append(f"{label}: only {per_intent[label]} rows (need {args.min})")

    if args.path.resolve() != TEST_FILE.resolve() and TEST_FILE.exists():
        test = {norm(r["text"]) for r in read_jsonl(TEST_FILE)}
        leaks = [t for t in seen if t in test]
        if leaks:
            problems.append(f"{len(leaks)} rows also in {TEST_FILE.name} (test leakage): " + "; ".join(leaks[:10]))

    print("\n".join(problems))
    counts = per_intent.most_common()
    if counts:
        print(f"per intent: min {counts[-1][1]} ({counts[-1][0]}), max {counts[0][1]} ({counts[0][0]})")
    print(f"{sum(per_intent.values())} rows, {len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()

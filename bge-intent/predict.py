"""Predict one intent using a trained BGE classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import IntentClassifier, get_device, load_tokenizer, tokenize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="utterance to classify")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("text must not be empty")
    if args.top_k < 1:
        parser.error("top-k must be positive")

    label2id = json.loads((args.runs_dir / "label2id.json").read_text(encoding="utf-8"))
    id2label = {index: label for label, index in label2id.items()}
    device = get_device()
    model = IntentClassifier(len(label2id)).to(device)
    model.load_state_dict(torch.load(args.runs_dir / "model.pt", map_location=device, weights_only=True))
    model.eval()
    tokenizer = load_tokenizer()
    with torch.inference_mode():
        scores = model(**tokenize(tokenizer, [args.text.strip()], device)).softmax(dim=1)[0]
    top_indices = scores.argsort(descending=True)[:min(args.top_k, len(label2id))]
    print(json.dumps({
        "text": args.text,
        "selected_intent": id2label[int(top_indices[0])],
        "confidence": float(scores[top_indices[0]]),
        "top_intents": [
            {"intent": id2label[int(index)], "confidence": float(scores[index])}
            for index in top_indices
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""
Classify a sentence with a trained run.

- Loads runs/<run>/ (newest train run by default): model.pt, label2id.json, encoder_config/, tokenizer/
- Normalizes jargon with the run's own glossary.json copy (same rewrite it was trained with)
- Softmax; top probability under the run's min_conf -> "unhandled"
- For the five name intents, also extracts the name as "slot"
  ("named"/"called" -> what follows; otherwise peel command words off both ends)

Usage:
    python code/predict.py "add a waypoint named Olympus Ridge"
    python code/predict.py --run runs/train-... "where is the lander"
    python code/predict.py                       type sentences, blank line quits
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

from config import SLOT_INTENTS, UNHANDLED
from model import IntentModel, load_tokenizer, tokenize
from normalize import Normalizer
from rundir import latest_run

# Words that can surround a name but are never part of it
FILLER = {"um", "uh", "hey", "please", "now", "thanks", "ok", "okay", "so", "and", "copy", "roger",
          "that", "suits", "houston", "alright", "over", "real", "quick"}  # radio filler, as in generate.py
LEAD = FILLER | {
    "can", "could", "would", "you", "i", "want", "need", "lets", "let's", "let", "us", "me", "my",
    "what", "whats", "what's", "where", "where's", "is", "are", "show", "give", "tell", "find", "get",
    "set", "add", "at", "create", "drop", "make", "put", "place", "new", "save", "pin",
    "delete", "remove", "clear", "erase", "rid", "mark", "complete", "check", "off", "finish", "finished",
    "navigate", "route", "take", "go", "guide", "head", "bring", "start",
    "waypoint", "marker", "point", "task", "navigation", "nav", "target",
    "position", "location", "coordinates", "coords", "destination",
    "the", "a", "an", "to", "for", "of", "towards", "toward", "as",
}
TRAIL = FILLER | {
    "waypoint", "marker", "pin", "point", "task", "location", "position", "coordinates", "coords",
    "complete", "completed", "done", "finished", "as", "is", "for", "me",
}
NAME_CUE = re.compile(r"\b(?:named|called)\b", re.I)


def _bare(word):
    return re.sub(r"[^\w']", "", word).lower()


def extract_slot(text):
    cue = list(NAME_CUE.finditer(text))
    words = (text[cue[-1].end():] if cue else text).split()
    if not cue:
        while words and _bare(words[0]) in LEAD:
            words.pop(0)
    while words and _bare(words[-1]) in TRAIL:
        words.pop()
    while words and _bare(words[0]) in {"the", "a", "an"}:
        words.pop(0)
    return " ".join(words).strip(" .,!?'\"") or None


class Predictor:
    def __init__(self, run=None):
        self.run = Path(run) if run else latest_run("train", "model.pt")
        label2id = json.loads((self.run / "label2id.json").read_text(encoding="utf-8"))
        self.labels = sorted(label2id, key=label2id.get)
        config = json.loads((self.run / "config.json").read_text(encoding="utf-8"))
        self.min_conf = config["min_conf"]
        self.normalize = Normalizer(self.run / "glossary.json" if config.get("normalize") else None)
        self.tokenizer = load_tokenizer(self.run / "tokenizer")
        self.model = IntentModel.load(self.run, "cpu")  # one sentence at a time: CPU beats GPU transfer overhead

    @torch.inference_mode()
    def __call__(self, text):
        probs = self.model(tokenize(self.tokenizer, [self.normalize(text)], "cpu")).softmax(dim=1)[0]
        top_confs, top_idxs = probs.topk(min(5, len(self.labels)))
        conf = top_confs[0].item()
        intent = self.labels[top_idxs[0].item()] if conf >= self.min_conf else UNHANDLED
        result = {
            "text": text,
            "selected_intent": intent,
            "confidence": round(conf, 3),
            "top5": [{"intent": self.labels[i], "confidence": round(c, 3)} for c, i in zip(top_confs.tolist(), top_idxs.tolist())],
        }
        if intent in SLOT_INTENTS:
            result["slot"] = extract_slot(text)
        return result


_predictor = None


def predict(text, run=None):
    global _predictor
    if _predictor is None:
        _predictor = Predictor(run)
    return _predictor(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run")
    parser.add_argument("text", nargs="*")
    args = parser.parse_args()

    if args.text:
        print(json.dumps(predict(" ".join(args.text), args.run)))
    else:
        p = Predictor(args.run)
        print(f"run: {p.run.name}. Type a sentence, blank line quits.")
        for line in sys.stdin:
            if not line.strip():
                break
            print(json.dumps(p(line.strip())))

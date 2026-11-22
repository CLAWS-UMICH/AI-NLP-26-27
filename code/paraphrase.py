"""
Paraphrase seed and template rows by back-translation (English -> German/French -> English) with MarianMT.

Runs on one pool at a time (data/pools/<pool>/seed.jsonl + tmpl.jsonl), so a val paraphrase is never a
variation of a train row. Writes data/pools/<pool>/para.jsonl; run generate.py afterwards to rebuild train/val.

1. Pick sources per intent: seed rows first, then template rows, until the intent's quota is met
   (--target rows for the train pool, split evenly over the command intents; val gets the same share of it
   that its seed is of the train seed). unhandled rows are not paraphrased.
2. Mask the slot, glossary terms and the words that name the intent ("vitals", "storage", "coolant") as
   X1, X2, ... (MarianMT copies these through untouched), so "primary oxygen" can't come back as
   "first oxygen", "vitals" can't become "Vitale", and the slot keeps its exact spelling
3. Translate out with sampling (2 per pivot language) and back with beam search, then put the masked words back
4. Drop a candidate if: a placeholder went missing or doubled, its length is far off from the source,
   its meaning drifted (primed-encoder cosine to the source < ADEQUACY), it's a near-duplicate (rapidfuzz
   ratio >= 90) of a row already in this pool, it's close to a row of another split (ratio >= 85: test,
   the other pool), or the primed encoder's nearest seed row has another intent. The most faithful
   candidates are tried first.
5. Every kept row keeps its source row's group and parent, with "source": "para:bt"

Usage:
    python code/paraphrase.py
    python code/paraphrase.py --target 1000 --per-row 2 --pools train
"""

import argparse
import math
import random
import re
import sys
from collections import Counter, defaultdict

import torch
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer, util
from transformers import MarianMTModel, MarianTokenizer

from config import BASE_MODEL, POOLS, RUNS, SEED, TEST_FILE, UNHANDLED
from data import norm, read_jsonl, write_jsonl
from model import pick_device
from normalize import Normalizer, term_forms

PIVOTS = ("de", "fr")
SAMPLES = 2           # forward translations per pivot language, so up to 4 candidates per source row
BATCH = 32
DUP_RATIO = 90        # near-duplicate inside the pool (same threshold as audit.py)
LEAK_RATIO = 85       # too close to a row of another split
ADEQUACY = 0.80       # min cosine to the source row; below it the meaning drifted ("pull up the vital umbrella")
VERBS = {"get", "set", "open", "close", "add", "delete", "complete", "reroute"}  # free to vary, so not masked
PLACEHOLDER = re.compile(r"(?<!\w)x(\d+)(?!\w)", re.I)


def load_pool(pool):
    rows = []
    for name in ("seed", "tmpl"):
        path = POOLS / pool / f"{name}.jsonl"
        if path.exists():
            rows += read_jsonl(path)
        elif name == "seed":
            sys.exit(f"No {path}. Run python code/split_seed.py first.")
    return rows


def intent_words(intent):
    """Regex for the words in an intent's name that say what it's about: get_coolant_gas_pressure ->
    coolant, gas, pressure (plural allowed)."""
    words = [w for w in intent.lower().split("_") if len(w) > 2 and w not in VERBS]
    return re.compile(r"(?<!\w)(" + "|".join(rf"{re.escape(w)}s?" for w in words) + r")(?!\w)", re.I) if words else None


def mask(row, patterns):
    """Text with the slot and every patterns match replaced by X1, X2, ... and the list of what they replaced."""
    text, spans = row["text"], []
    if row.get("slot"):
        m = re.search(rf"(?<!\w){re.escape(row['slot'])}(?!\w)", text, re.I)
        if m:
            spans.append(m.span())
    for pattern in filter(None, patterns):
        spans += [m.span() for m in pattern.finditer(text)
                  if not any(m.start() < e and s < m.end() for s, e in spans)]
    out, words, last = [], [], 0
    for s, e in sorted(spans):
        words.append(text[s:e])
        out += [text[last:s], f"X{len(words)}"]
        last = e
    return "".join(out + [text[last:]]), words


def unmask(text, words):
    """Put the masked words back, or None if any placeholder is missing, repeated or unknown."""
    found = [int(m.group(1)) for m in PLACEHOLDER.finditer(text)]
    if sorted(found) != list(range(1, len(words) + 1)):
        return None
    return PLACEHOLDER.sub(lambda m: words[int(m.group(1)) - 1], text)


class BackTranslator:
    def __init__(self, device):
        self.device, self.models = device, {}

    def _model(self, name):
        if name not in self.models:
            tok = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{name}")
            self.models[name] = (tok, MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{name}").to(self.device).eval())
        return self.models[name]

    @torch.inference_mode()
    def _translate(self, name, texts, sample):
        tok, model = self._model(name)
        batch = tok(texts, return_tensors="pt", padding=True).to(self.device)
        kw = dict(do_sample=True, top_p=0.9, num_return_sequences=SAMPLES) if sample else dict(num_beams=4)
        return tok.batch_decode(model.generate(**batch, max_new_tokens=48, **kw), skip_special_tokens=True)

    def __call__(self, texts):
        """[(index into texts, pivot, English candidate)] for every pivot and sample."""
        out = []
        for pivot in PIVOTS:
            there = self._translate(f"en-{pivot}", texts, sample=True)
            back = self._translate(f"{pivot}-en", there, sample=False)
            out += [(i // SAMPLES, pivot, t) for i, t in enumerate(back)]
        return out


def encoder_path():
    primed = sorted(p / "encoder" for p in RUNS.glob("prime-*") if (p / "encoder" / "config.json").exists())
    return str(primed[-1]) if primed else BASE_MODEL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1000, help="paraphrases for the train pool")
    parser.add_argument("--per-row", type=int, default=2, help="max paraphrases kept per source row")
    parser.add_argument("--pools", nargs="+", default=["train", "val"])
    args = parser.parse_args()

    rng = random.Random(SEED)
    device = pick_device()
    forms, pattern = term_forms()
    normalizer = Normalizer()
    mt = BackTranslator(device)

    pools = {p: load_pool(p) for p in ("train", "val")}
    test_norms = [norm(r["text"]) for r in read_jsonl(TEST_FILE)] if TEST_FILE.exists() else []

    # nearest-seed intent check, over the seed rows of both pools (the check creates no rows, so no leakage)
    seeds = [r for p in ("train", "val") for r in read_jsonl(POOLS / p / "seed.jsonl")]
    encoder = SentenceTransformer(encoder_path(), device=str(device))
    seed_emb = encoder.encode([normalizer(r["text"]) for r in seeds], normalize_embeddings=True, convert_to_tensor=True)
    seed_intents = [r["intent"] for r in seeds]

    train_seeds = sum(r.get("source") == "seed" for r in pools["train"])
    for pool in args.pools:
        rows = [r for r in pools[pool] if r["intent"] != UNHANDLED]
        intents = sorted({r["intent"] for r in rows})
        if not intents:
            print(f"{pool}: nothing to paraphrase")
            continue
        scale = 1.0 if pool == "train" else sum(r.get("source") == "seed" for r in pools[pool]) / max(1, train_seeds)
        quota = max(1, round(args.target * scale / len(intents)))

        own = [norm(r["text"]) for r in pools[pool]]
        others = test_norms + [norm(r["text"]) for p in pools if p != pool for r in pools[p]]
        kept, drops = [], Counter()

        for intent in intents:
            mine = [r for r in rows if r["intent"] == intent]
            seed_rows = [r for r in mine if r.get("source") == "seed"]
            tmpl_rows = [r for r in mine if r.get("source") != "seed"]
            sources = rng.sample(seed_rows, len(seed_rows)) + rng.sample(tmpl_rows, len(tmpl_rows))
            sources = sources[:math.ceil(3 * quota / args.per_row)]  # stop wasting compute on a hard intent
            own_words = intent_words(intent)
            got = 0
            for start in range(0, len(sources), BATCH):
                chunk = sources[start:start + BATCH]
                masked = [mask(r, (pattern, own_words)) for r in chunk]
                cands, per_src = [], Counter()
                for i, pivot, text in mt([m for m, _ in masked]):
                    text = unmask(text, masked[i][1])
                    if text is None:
                        drops["placeholder"] += 1
                        continue
                    src_words, words = len(chunk[i]["text"].split()), len(text.split())
                    if not (0.5 * src_words <= words <= 2 * src_words + 2):
                        drops["length"] += 1
                        continue
                    cands.append((i, pivot, text))
                if not cands:
                    continue
                emb = encoder.encode([normalizer(t) for _, _, t in cands], normalize_embeddings=True,
                                     convert_to_tensor=True)
                src_emb = encoder.encode([normalizer(r["text"]) for r in chunk], normalize_embeddings=True,
                                         convert_to_tensor=True)
                adequacy = (emb * src_emb[[i for i, _, _ in cands]]).sum(dim=1).tolist()
                nearest = util.cos_sim(emb, seed_emb).argmax(dim=1).tolist()
                for (i, pivot, text), near, adq in sorted(zip(cands, nearest, adequacy), key=lambda c: -c[2]):
                    if got >= quota:
                        break
                    if per_src[i] >= args.per_row:
                        continue
                    if adq < ADEQUACY:
                        drops["meaning drifted"] += 1
                        continue
                    key = norm(text)
                    if process.extractOne(key, own, scorer=fuzz.ratio, score_cutoff=DUP_RATIO):
                        drops["near-duplicate"] += 1
                        continue
                    if process.extractOne(key, others, scorer=fuzz.ratio, score_cutoff=LEAK_RATIO):
                        drops["close to another split"] += 1
                        continue
                    if seed_intents[near] != intent:
                        drops["nearest seed has another intent"] += 1
                        continue
                    src = chunk[i]
                    row = {"text": text, "intent": intent, "source": "para:bt", "pivot": pivot,
                           "group": src["group"], "parent": src.get("parent", src.get("id"))}
                    if src.get("slot"):
                        row["slot"] = src["slot"]
                    kept.append(row)
                    own.append(key)
                    per_src[i] += 1
                    got += 1
                if got >= quota:
                    break
            if got < quota:
                drops[f"short of quota ({intent})"] += quota - got

        write_jsonl(POOLS / pool / "para.jsonl", kept)
        print(f"{pool}: {len(kept)} paraphrases ({quota} per intent target) -> {POOLS / pool / 'para.jsonl'}")
        for reason, n in sorted(drops.items(), key=lambda kv: -kv[1]):
            print(f"  {n:5}  {reason}")


if __name__ == "__main__":
    main()

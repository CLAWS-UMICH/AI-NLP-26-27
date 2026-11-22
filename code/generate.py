"""
Build data/train.jsonl and data/val.jsonl from the two pools that split_seed.py made.

1. Templates: expand data/templates/<intent>.txt into data/pools/<pool>/tmpl.jsonl. Each line is
       <seed id>: {take|bring|get} me to <SLOT> [please]
   {a|b|c} picks one ({|please} can pick nothing), [words] shows up half the time, both can nest, and
   <SLOT> gets a place name from data/slot_names.txt. A line goes to the pool its seed row is in and keeps
   that row's group. Every line of an intent gets an equal share of --tmpl-per-intent rows.
2. Load every *.jsonl in each pool: seed, tmpl, and whatever later steps add (para, clinc, asr, llm)
3. Noise, in place, on --noise of the rows (seed, recorded and Whisper round-trip rows are left as they are):
   - slot swap:  replace the slot name with another place name
   - filler:     radio/discourse words around the command ("copy,", "uh", "suits,", "please")
   - jargon:     swap a term for another form from data/suits_glossary.json ("primary oxygen" -> "pri o2",
                 "uia" -> "u i a", "heart rate" -> "hr"), so the model sees astronaut and Whisper spellings
   - asr:        generic Whisper mishearings ("add" -> "at", "what's" -> "what is")
   - style:      lowercase/no punctuation (what we write) vs. Whisper casing + punctuation (what it outputs)
   The slot name is protected while the text is noised, so "slot" stays correct.
4. Drop exact duplicates, rows that match test.jsonl, and val rows that match a train row
5. Cap llm rows at LLM_MAX_SHARE of the pool and every intent at --cap rows (seed rows are kept first),
   then print the mix next to MIX_TARGETS

Run it again after paraphrase.py (or any other step) adds a file to the pools.

Usage:
    python code/generate.py
    python code/generate.py --tmpl-per-intent 85 --noise 0.5 --cap 170
"""

import argparse
import json
import math
import random
import re
import sys
from collections import Counter

from config import (GLOSSARY_FILE, LLM_MAX_SHARE, MIX_TARGETS, POOLS, SEED, SLOT_INTENTS, SLOT_NAMES_FILE,
                    TEMPLATES_DIR, TEST_FILE, TRAIN_FILE, VAL_FILE)
from data import load_labels, mix_bucket, norm, read_jsonl, write_jsonl

DEFAULT_SLOT_NAMES = [
    "crater rim", "boulder field", "lander", "poi a", "poi b", "poi c", "alpha", "bravo", "charlie",
    "ridge two", "shadow edge", "rock garden", "sample site one", "comm tower", "rover", "basalt outcrop",
    "olympus ridge", "station delta", "egress", "ingress", "geology sampling", "photo survey",
]
PREFIXES = ["", "", "", "uh ", "um ", "okay ", "copy, ", "copy that, ", "hey suits, ", "suits, ", "roger, ",
            "quick one, ", "alright ", "so ", "houston, "]
SUFFIXES = ["", "", "", " please", " real quick", " over", " thanks", " now", " for me"]
ASR = [  # (pattern, Whisper-style mishearing); term-specific ones ("o two", "uya") live in the glossary aliases
    (r"\badd\b", "at"), (r"\brpm\b", "r p m"), (r"\bwhat's\b", "what is"), (r"\bwhat is\b", "what's"),
]


def load_jargon():
    """(pattern, replacements) from the glossary. Any form of a term can become any other form, but only
    the expansion or a safe-to-normalize form is ever a *source*: "secondary" -> "sec" is fine,
    "give me a sec" -> "give me a secondary" is not. Radio words are left to the filler lists."""
    if not GLOSSARY_FILE.exists():
        return []
    swaps = []
    for row in json.loads(GLOSSARY_FILE.read_text(encoding="utf-8")):
        if row["category"] == "radio":
            continue
        forms = list(dict.fromkeys(f.lower() for f in [row["expansion"], row["term"], *row.get("aliases", [])]))
        sources = forms if row.get("normalize") else forms[:1]
        for src in sources:
            targets = [f for f in forms if f != src]
            if targets:
                swaps.append((re.compile(rf"(?<!\w){re.escape(src)}(?!\w)", re.I), targets))
    return swaps


JARGON = load_jargon()


SLOT_MARK = "\x00slot\x00"


def protect(text, slot):
    """Swap the slot name for a placeholder so noise can't edit it."""
    return re.sub(re.escape(slot), SLOT_MARK, text, count=1, flags=re.I) if slot else text


def restore(text, slot):
    return text.replace(SLOT_MARK, slot) if slot else text


def jargon(text, rng):
    """Swap one glossary term that actually occurs in text (a random pick would usually match nothing)."""
    options = [(p, targets) for p, targets in JARGON if p.search(text)]
    if not options:
        return text
    pattern, targets = rng.choice(options)
    new = rng.choice(targets)
    return pattern.sub(lambda m: new, text, count=1)


def asr(text, rng):
    options = [(p, heard) for p, heard in ASR if re.search(p, text, re.I)]
    if not options:
        return text
    pattern, heard = rng.choice(options)
    return re.sub(pattern, heard, text, count=1, flags=re.I)


def style(text, rng):
    text = " ".join(text.split())
    if rng.random() < 0.6:
        return " ".join(re.sub(r"[^\w\s'\x00]", " ", text).lower().split())
    text = text[0].upper() + text[1:] if text else text
    return text if text[-1:] in ".?!" else text + rng.choice([".", "?", ""])


def variant(row, names, rng):
    """row with noised text (or None if it came out unchanged). Keeps source, group and parent, and
    records what was done in "noise"."""
    text, slot = row["text"], row.get("slot")
    ops = []

    if slot and rng.random() < 0.7:
        new = rng.choice(names)
        text, found = re.subn(re.escape(slot), lambda m: new, text, count=1, flags=re.I)
        if found:  # otherwise slot would name something the text doesn't say
            slot = new
            ops.append("slot")

    text = protect(text, slot)
    if rng.random() < 0.5:
        text = rng.choice(PREFIXES) + text + rng.choice(SUFFIXES)
        ops.append("filler")
    for name, fn, p in (("jargon", jargon, 0.5), ("asr", asr, 0.3)):
        if rng.random() < p and (new := fn(text, rng)) != text:
            text = new
            ops.append(name)
    text = restore(style(text, rng), slot)

    if norm(text) == norm(row["text"]) or not text:
        return None
    out = {**row, "text": text, "noise": "+".join(ops or ["style"])}
    if slot:
        out["slot"] = slot
    return out


# TEMPLATES ------------------------------------------------------

INNER = re.compile(r"\{([^{}\[\]]*)\}|\[([^{}\[\]]*)\]")  # innermost {a|b} or [opt]


def sample_template(template, rng):
    """One random sentence from a template; innermost groups are resolved first, so nesting works."""
    def pick(m):
        if m.group(1) is not None:
            return rng.choice(m.group(1).split("|"))
        return m.group(2) if rng.random() < 0.5 else ""
    while INNER.search(template):
        template = INNER.sub(pick, template)
    return " ".join(template.split())


def read_templates(seeds):
    """{intent: [(seed row, template)]} from data/templates/*.txt. seeds maps id -> (pool, seed row)."""
    labels, out, problems = set(load_labels()), {}, []
    for path in sorted(TEMPLATES_DIR.glob("*.txt")):
        intent = path.stem
        if intent not in labels:
            problems.append(f"{path.name}: {intent!r} is not an intent")
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            where = f"{path.name}:{n}"
            parent, sep, template = line.partition(":")
            parent, template = parent.strip(), template.strip()
            if not sep or not template:
                problems.append(f"{where}: expected '<seed id>: template'")
            elif parent not in seeds:
                problems.append(f"{where}: {parent!r} is not a train or val pool seed id (test rows can't have templates)")
            elif seeds[parent][1]["intent"] != intent:
                problems.append(f"{where}: seed {parent} is {seeds[parent][1]['intent']}, not {intent}")
            elif ("<SLOT>" in template) != (intent in SLOT_INTENTS):
                problems.append(f"{where}: <SLOT> belongs in every {intent} template" if intent in SLOT_INTENTS
                                else f"{where}: {intent} takes no <SLOT>")
            elif template.count("{") != template.count("}") or template.count("[") != template.count("]"):
                problems.append(f"{where}: unbalanced brackets")
            else:
                out.setdefault(intent, []).append((seeds[parent], template))
    if problems:
        sys.exit("Fix the templates first:\n  " + "\n  ".join(problems))
    return out


def expand_templates(seeds, names, per_intent, rng):
    """Write data/pools/<pool>/tmpl.jsonl for both pools. Returns rows per pool."""
    rows = {"train": [], "val": []}
    for intent, lines in read_templates(seeds).items():
        in_train = sum(pool == "train" for (pool, _), _ in lines)
        per_line = math.ceil(per_intent / max(1, in_train))
        for (pool, seed), template in lines:
            seen = set()
            for _ in range(per_line * 10):
                if len(seen) >= per_line:
                    break
                slot = rng.choice(names) if "<SLOT>" in template else None
                text = sample_template(template, rng).replace("<SLOT>", slot or "")
                if not text or norm(text) in seen:
                    continue
                seen.add(norm(text))
                row = {"text": text, "intent": intent, "source": "tmpl", "group": seed["group"], "parent": seed["id"]}
                if slot:
                    row["slot"] = slot
                rows[pool].append(row)
    for pool, pool_rows in rows.items():
        write_jsonl(POOLS / pool / "tmpl.jsonl", pool_rows)
    return rows


# ASSEMBLY -------------------------------------------------------

KEEP_AS_IS = ("seed", "asr:", "rec:")  # handwritten, or real Whisper output: never noised


def load_pool(pool):
    """Every *.jsonl in the pool, seed rows first so they win duplicate checks."""
    files = sorted((POOLS / pool).glob("*.jsonl"), key=lambda p: (p.stem != "seed", p.stem))
    rows = []
    for path in files:
        rows += [{"source": path.stem, **r} for r in read_jsonl(path)]
    return rows


def cap_llm(rows, rng):
    llm = [r for r in rows if r["source"] == "llm"]
    allowed = math.floor(LLM_MAX_SHARE * (len(rows) - len(llm)) / (1 - LLM_MAX_SHARE))
    if len(llm) <= allowed:
        return rows, 0
    drop = {id(r) for r in rng.sample(llm, len(llm) - allowed)}
    return [r for r in rows if id(r) not in drop], len(drop)


def cap_intents(rows, cap, rng):
    if not cap:
        return rows
    rows = sorted(rng.sample(rows, len(rows)), key=lambda r: r["source"] != "seed")  # seed first, rest shuffled
    counts, out = Counter(), []
    for r in rows:
        if counts[r["intent"]] < cap:
            counts[r["intent"]] += 1
            out.append(r)
    return out


def report(name, rows, dropped):
    per = Counter(r["intent"] for r in rows).most_common()
    dropped = ", ".join(f"{n} {why}" for why, n in dropped.items() if n)
    print(f"\n{name}: {len(rows)} rows" + (f" (dropped {dropped})" if dropped else ""))
    if per:
        print(f"  per intent: min {per[-1][1]} ({per[-1][0]}), max {per[0][1]} ({per[0][0]})")
    mix = Counter(mix_bucket(r) for r in rows)
    for bucket in sorted(set(MIX_TARGETS) | set(mix)):
        share = mix[bucket] / max(1, len(rows))
        target = f"target ~{MIX_TARGETS[bucket]:.0%}" if bucket in MIX_TARGETS else ""
        print(f"  {bucket:>10}: {mix[bucket]:5} ({share:4.0%})  {target}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmpl-per-intent", type=int, default=85, help="template rows per intent in the train pool")
    parser.add_argument("--noise", type=float, default=0.5, help="share of rows that get noised")
    parser.add_argument("--cap", type=int, default=0, help="max train rows per intent (0 = no cap); val gets the same share")
    args = parser.parse_args()

    if not (POOLS / "train" / "seed.jsonl").exists():
        sys.exit(f"No {POOLS / 'train' / 'seed.jsonl'}. Run python code/split_seed.py first.")
    rng = random.Random(SEED)
    names = DEFAULT_SLOT_NAMES
    if SLOT_NAMES_FILE.exists():
        names = [n.strip() for n in SLOT_NAMES_FILE.read_text(encoding="utf-8").splitlines() if n.strip()]
    seeds = {r["id"]: (pool, r) for pool in ("train", "val") for r in read_jsonl(POOLS / pool / "seed.jsonl")}
    tmpl = expand_templates(seeds, names, args.tmpl_per_intent, rng)
    print(f"templates: {len(tmpl['train'])} train-pool rows, {len(tmpl['val'])} val-pool rows")

    test = {norm(r["text"]) for r in read_jsonl(TEST_FILE)} if TEST_FILE.exists() else set()
    train_seeds = sum(1 for _, (pool, _) in seeds.items() if pool == "train")
    train_keys = set()
    for pool, out_file in (("train", TRAIN_FILE), ("val", VAL_FILE)):
        rows, seen, dropped = [], set(), Counter()
        for r in load_pool(pool):
            if not r["source"].startswith(KEEP_AS_IS) and rng.random() < args.noise:
                r = variant(r, names, rng) or r
            key = norm(r["text"])
            if key in test:
                dropped["matched test"] += 1
            elif key in seen:
                dropped["duplicates"] += 1
            elif key in train_keys:
                dropped["matched train"] += 1
            else:
                seen.add(key)
                rows.append(r)
        rows, dropped["llm over cap"] = cap_llm(rows, rng)
        scale = 1.0 if pool == "train" else (len(seeds) - train_seeds) / max(1, train_seeds)
        before = len(rows)
        rows = cap_intents(rows, round(args.cap * scale), rng)
        dropped["over --cap"] = before - len(rows)
        if pool == "train":
            train_keys = seen

        rng.shuffle(rows)
        write_jsonl(out_file, rows)
        report(f"{pool} -> {out_file}", rows, dropped)


if __name__ == "__main__":
    main()

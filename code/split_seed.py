"""
Split the handwritten seed into test and the two training pools, once, before anything is generated.

1. Every seed row gets a stable id (a hash of its text), so templates can name the row they were written from
2. Skeleton: the text with its slot masked as <SLOT> and glossary terms masked by their expansion
   ("what's my pri o2" -> "what's my <PRIMARY> <OXYGEN>"). Rows of one intent with the same skeleton are one group
3. Groups of one intent whose skeletons are close (MiniLM cosine >= GROUP_SIM) are joined, so
   "take me to <SLOT>" and "bring me to <SLOT>" can't land on opposite sides. The comparison is on phrasing
   only: the slot, glossary terms and the words that name the intent are masked to one neutral word first
4. Per intent, by group: ~TEST_SHARE of the rows -> data/test.jsonl, the rest is the train side.
   ~VAL_POOL_SHARE of the train side -> data/pools/val/seed.jsonl, the rest -> data/pools/train/seed.jsonl

Templates, paraphrases, TTS and LLM rows are made from one pool at a time and keep their seed row's group,
so a val or test row is never a variation of a train row.

Test is frozen once written: an existing test.jsonl is never touched unless --force. With --force only the
seed rows in it are replaced; CLINC and recorded rows already in it are kept.

Usage:
    python code/split_seed.py
    python code/split_seed.py --force
"""

import argparse
import hashlib
import random
import re
import sys
from collections import Counter, defaultdict

from sentence_transformers import SentenceTransformer, util

from config import (BASE_MODEL, GROUP_SIM, POOLS, SEED, SEED_FILE, SLOT_INTENTS, TEST_FILE, TEST_SHARE,
                    VAL_POOL_SHARE)
from data import load_labels, norm, read_jsonl, write_jsonl
from normalize import intent_words, term_forms


def row_id(text):
    return "s" + hashlib.sha1(norm(text).encode()).hexdigest()[:6]


def skeleton(row, forms, pattern):
    text = norm(row["text"])
    if row.get("slot"):
        text = re.sub(rf"(?<!\w){re.escape(norm(row['slot']))}(?!\w)", "<SLOT>", text, count=1)
    if pattern:
        text = pattern.sub(lambda m: "<" + forms[m.group(0).lower()].upper().replace(" ", "_") + ">", text)
    return text


def readable(skel, intent):
    """Skeleton for embedding, so similarity measures how a command is asked, not what it's about.
    Skeletons are only compared within one intent, where every row names the same subject ("helmet co2"),
    and those shared words pushed short rows over GROUP_SIM whatever their phrasing. So the subject becomes
    one neutral word: glossary placeholders and the words that name the intent -> "something", runs of it
    collapsed ("suit <OXYGEN> pressure" -> "something"). The bare <SLOT> token makes every short slot command
    look alike to MiniLM (~0.85 for "navigate to <SLOT>" vs "point me toward the <SLOT>"), so it becomes
    "somewhere"."""
    text = re.sub(r"<(?!SLOT>)[A-Z_]+>", "something", skel).replace("<SLOT>", "somewhere")
    own = intent_words(intent)
    if own:
        text = own.sub("something", text)
    return re.sub(r"\bsomething(?: something)+\b", "something", text)


def components(n, pairs):
    """Connected components (union-find) of n items joined by pairs. Transitive on purpose: if A~B and B~C,
    A and C must stay on one side too."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        parent[find(i)] = find(j)
    return [find(i) for i in range(n)]


def take(groups, target, rng):
    """Pick whole groups (in random order) whose row count lands as close to target as possible."""
    picked, count = [], 0
    for g in rng.sample(groups, len(groups)):
        if abs(count + len(g) - target) < abs(count - target):
            picked.append(g)
            count += len(g)
    return picked


def check(rows, labels):
    problems, seen = [], {}
    for n, r in enumerate(rows, 1):
        if r.get("intent") not in labels:
            problems.append(f"row {n}: bad intent {r.get('intent')!r}")
        if r.get("slot") and r["intent"] not in SLOT_INTENTS:
            problems.append(f"row {n}: {r['intent']} takes no slot")
        if r.get("slot") and norm(r["slot"]) not in norm(r["text"]):
            problems.append(f"row {n}: slot {r['slot']!r} not in text")
        key = norm(r["text"])
        if key in seen:
            problems.append(f"row {n}: duplicate of row {seen[key]} '{r['text']}'")
        seen.setdefault(key, n)
    if problems:
        sys.exit(f"Fix {SEED_FILE.name} first:\n  " + "\n  ".join(problems))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-split even though test.jsonl exists")
    args = parser.parse_args()

    if not SEED_FILE.exists():
        sys.exit(f"No {SEED_FILE}. Write the seed first (see PLAN.md, Pipeline step 1).")
    kept_test = []
    if TEST_FILE.exists():
        if not args.force:
            sys.exit(f"{TEST_FILE} exists and test is frozen. Pass --force to re-split (only if nothing has been "
                     "generated or recorded against this split yet).")
        kept_test = [r for r in read_jsonl(TEST_FILE) if r.get("source", "seed") != "seed"]

    rows = read_jsonl(SEED_FILE)
    check(rows, set(load_labels()))
    forms, pattern = term_forms()
    for r in rows:
        r["id"] = row_id(r["text"])
        r["source"] = "seed"
        r["skeleton"] = skeleton(r, forms, pattern)

    # 2-3: one node per (intent, skeleton), joined by embedding similarity within an intent
    nodes = sorted({(r["intent"], r["skeleton"]) for r in rows})
    model = SentenceTransformer(BASE_MODEL)
    emb = model.encode([readable(s, intent) for intent, s in nodes], normalize_embeddings=True, convert_to_tensor=True)
    sim = util.cos_sim(emb, emb)
    pairs = [(i, j) for i, j in (sim >= GROUP_SIM).nonzero().tolist()
             if i < j and nodes[i][0] == nodes[j][0]]
    comp = dict(zip(nodes, components(len(nodes), pairs)))

    by_comp = defaultdict(list)
    for r in rows:
        by_comp[comp[(r["intent"], r["skeleton"])]].append(r)
    by_intent = defaultdict(list)
    for members in by_comp.values():
        group = "g" + members[0]["id"][1:]
        for r in members:
            r["group"] = group
        by_intent[members[0]["intent"]].append(members)

    # 4: per intent, whole groups to test, then val pool, then train pool
    rng = random.Random(SEED)
    test, val, train, warnings = [], [], [], []
    for intent in sorted(by_intent):
        groups = sorted(by_intent[intent], key=lambda g: g[0]["id"])
        n = sum(len(g) for g in groups)
        to_test = take(groups, n * TEST_SHARE, rng) if len(groups) > 1 else []
        rest = [g for g in groups if g not in to_test]
        side = sum(len(g) for g in rest)
        to_val = take(rest, max(1, side * VAL_POOL_SHARE), rng) if len(rest) > 1 else []
        to_train = [g for g in rest if g not in to_val]
        if not to_test or not to_val:
            warnings.append(f"{intent}: only {len(groups)} phrasing groups, so "
                            f"{'test' if not to_test else 'val'} gets none. Write more varied seed rows.")
        for dest, picked in ((test, to_test), (val, to_val), (train, to_train)):
            dest.extend(r for g in picked for r in g)

    TEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(TEST_FILE, test + kept_test)
    for pool, pool_rows in (("train", train), ("val", val)):
        (POOLS / pool).mkdir(parents=True, exist_ok=True)
        write_jsonl(POOLS / pool / "seed.jsonl", pool_rows)

    print(f"{len(rows)} seed rows, {len({r['skeleton'] for r in rows})} skeletons, "
          f"{len(by_comp)} groups after joining {len(pairs)} close skeleton pairs (cosine >= {GROUP_SIM})")
    for name, part in (("test", test), ("val pool", val), ("train pool", train)):
        per = Counter(r["intent"] for r in part)
        print(f"{name:>10}: {len(part):4} rows, {len({r['group'] for r in part}):3} groups, "
              f"per intent {min(per.values(), default=0)}-{max(per.values(), default=0)}")
    if kept_test:
        print(f"kept {len(kept_test)} non-seed rows already in {TEST_FILE.name}")
    if warnings:
        print("\n" + "\n".join(warnings))


if __name__ == "__main__":
    main()

"""Apply the real Whisper confusion table to train text (DATA_GENERATION.md §4: 'The first run also yields a confusion
table that you can apply to text cheaply in later rounds'). Only confusions seen >= 3 times are used; slot names are never
touched; confusions that change meaning (e.g. co2 -> o2) or delete a deciding word are skipped; every variant must still
pass the deciding-word rules. Variants (round-trip + confusion) are capped at --cap x the label's base rows.
Usage: python -m datagen.confusion_aug [--cap 0.75]"""
import argparse, json, random, re
from collections import Counter
from .config import GEN, REQUIRE, SLOT_LABELS
from .common import read_jsonl, write_jsonl, norm, strip_filler

BLOCK = {("co2", "o2"), ("ppco2", "co2"), ("helmet", ""), ("primary", ""), ("secondary", ""), ("b", ""), ("a", ""),
         ("gas", ""), ("liquid", ""), ("suit", ""), ("scrubber", "")}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cap", type=float, default=0.75); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = random.Random(a.seed)
    conf = json.load(open(GEN / "asr_confusions.json"))
    subs = []
    for k, c in conf.items():
        if c < 3 or " => " not in k: continue
        x, y = k.split(" => ")
        if (x, y) in BLOCK or not x: continue
        subs.append((x, y, c))
    base = [r for r in read_jsonl(GEN / "train.jsonl")]
    variants = read_jsonl(GEN / "train_asr.jsonl")
    have = Counter(v["label"] for v in variants)
    nbase = Counter(r["label"] for r in base)
    has_variant = {v["parent_id"] for v in variants if v.get("source") == "asr_confusion"}
    seen = {(r["label"], strip_filler(r["text"])) for r in base + variants}
    rng.shuffle(base); out, stats = [], Counter()
    for r in base:
        lab = r["label"]
        if r["id"] in has_variant or have[lab] >= a.cap * nbase[lab]: continue
        t = norm(r["text"]); slot = norm(r["slot_text"]) if r.get("slot_text") else None
        protected = set(slot.split()) if slot else set()
        cands = [(x, y, c) for x, y, c in subs if re.search(rf"\b{re.escape(x)}\b", t) and not (set(x.split()) & protected)]
        if not cands: stats["no_candidate"] += 1; continue
        rng.shuffle(cands); new = t; n = 0
        for x, y, c in sorted(cands, key=lambda z: -z[2] * rng.random()):
            new2 = re.sub(rf"\b{re.escape(x)}\b", y, new, count=1); new2 = re.sub(r"\s+", " ", new2).strip()
            if new2 != new: new = new2; n += 1
            if n >= 2: break
        if new == t or not new: continue
        req = REQUIRE.get(lab)
        if req and ((req[0] and not re.search(req[0], new)) or (req[1] and re.search(req[1], new))):
            stats["rule_fail"] += 1; continue
        if slot and not re.search(rf"\b{re.escape(slot)}\b", new): stats["slot_fail"] += 1; continue
        key = (lab, strip_filler(new))
        if key in seen: stats["dup"] += 1; continue
        seen.add(key); have[lab] += 1
        out.append({"id": r["id"] + "-conf", "parent_id": r["id"], "group": r.get("group"), "label": lab, "kind": r.get("kind"),
                    "fam": r.get("fam"), "text": new, "slot_text": slot, "split": "train", "source": "asr_confusion", "src_text": r["text"]})
    write_jsonl(GEN / "train_asr.jsonl", out, mode="a")
    stats["added"] = len(out)
    print(json.dumps(dict(stats))); print("examples:", [(o["src_text"], o["text"]) for o in out[:6]])

if __name__ == "__main__":
    main()

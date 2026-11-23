"""Use the team-written sentences in data/suits_glossary.json (= the 'usage' pairs in data/astronaut_primer.json) as a
source. Labels come from the glossary's intent hints, confirmed by the blind relabellers:
- one hinted intent: blind two-step check against it (same as agent rows);
- several hinted intents: kept only if qwen3:8b and qwen3:14b agree on one of them;
- no hint: kept as `unhandled` only if both models say unhandled.
Slots: a train-pool name in the text, else the glossary term itself (added to the train pool if it is not a dev/test name).
Each anchor/paraphrase pair shares one group. Then the normal rule checks, dedup gate.
Usage: python -m datagen.ingest_glossary --round K"""
import argparse, json, re
from collections import Counter
from .config import GEN, ROOT, DATA, SLOT_LABELS, SLOT_TYPE
from .common import read_jsonl, write_jsonl, norm
from .relabel import label_many, MODEL, SECOND
from .ingest import rule_check, gate, SLOTS, pool

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--round", type=int, required=True); a = ap.parse_args()
    g = json.load(open(DATA / "suits_glossary.json"))
    heldout = {n for t in ("place", "waypoint", "task") for sp in ("dev", "test") for n in SLOTS[t][sp]}
    rows = []
    for i, x in enumerate(g):
        for j, e in enumerate(x["example"]):
            t = re.sub(r"\s+", " ", e.lower().strip())
            rows.append({"id": f"r{a.round}-gloss{i:03d}-{j}", "group": f"r{a.round}-gloss{i:03d}", "round": a.round, "fam": None,
                         "cell": None, "kind": None, "text": t, "slot_text": None, "src": "glossary", "source": "glossary",
                         "term": x["term"].lower(), "hints": x["intents"]})
    p8 = label_many([r["text"] for r in rows], workers=4, model=MODEL)
    p14 = label_many([r["text"] for r in rows], workers=2, model=SECOND)
    stats, keep, rejected = Counter(proposed=len(rows)), [], []
    added_names = Counter()
    for r, a8, a14 in zip(rows, p8, p14):
        r["relabel_8b"], r["relabel_14b"] = a8, a14
        hints = r["hints"]
        if len(hints) == 1: lab = hints[0] if hints[0] in (a8, a14) else None
        elif hints: lab = a8 if (a8 == a14 and a8 in hints) else None
        else: lab = "unhandled" if a8 == a14 == "unhandled" else None
        if lab is None:
            r["reject"] = f"relabel:{a8}/{a14} hints={hints}"; rejected.append(r); stats["no_label_agreement"] += 1; continue
        r["label"] = lab
        if lab in SLOT_LABELS:
            names = sorted(pool("train", SLOT_TYPE[lab]), key=len, reverse=True)
            s = next((n for n in names if re.search(rf"\b{re.escape(n)}\b", r["text"])), None)
            term = r["term"]
            if not s and re.search(rf"\b{re.escape(term)}\b", r["text"]) and term not in heldout:
                ty = SLOT_TYPE[lab][-1]
                SLOTS[ty]["train"].append(term); added_names[term] += 1; s = term
            r["slot_text"] = s
        why = rule_check(r)
        if why: r["reject"] = why; rejected.append(r); stats["rule:" + why] += 1; continue
        keep.append(r)
    if added_names:
        json.dump(SLOTS, open(ROOT / "datagen/slots.json", "w"), indent=1)
    final = gate(keep, stats, rejected)
    for r in final: r["split"] = "train"
    write_jsonl(GEN / "train.jsonl", final, mode="a")
    write_jsonl(GEN / f"rejected_r{a.round}_glossary.jsonl", rejected)
    stats["accepted_train"] = len(final)
    print(json.dumps(dict(stats), indent=1))
    print("labels:", Counter(r["label"] for r in final).most_common())
    print("new train slot names:", dict(added_names))

if __name__ == "__main__":
    main()

"""Turn ASR outputs into labelled variants (DATA_GENERATION.md §4) and build the confusion table.
A variant keeps its row's label. It is dropped if the deciding word was deleted or turned into the sibling's word,
or if the slot name was lost. Misheard-but-present deciding words ("scrubber be") are kept: those are the real errors.
Usage: python -m datagen.asr_post --rows data/gen/train.jsonl --asr data/gen/asr/train_r0.jsonl --out data/gen/train_asr.jsonl"""
import argparse, json, re
from collections import Counter
from difflib import SequenceMatcher
from .config import GEN, REQUIRE, SLOT_LABELS
from .common import read_jsonl, write_jsonl, norm, strip_filler

def align(src, hyp):
    s, h = norm(src).split(), norm(hyp).split()
    return s, h, SequenceMatcher(a=s, b=h, autojunk=False).get_opcodes()

def map_span(s_idx, ops, h):
    """Map a source token span [i0,i1) through the alignment to hypothesis tokens."""
    i0, i1 = s_idx; out = []
    for tag, a0, a1, b0, b1 in ops:
        if a1 <= i0 or a0 >= i1: continue
        if tag in ("equal",):
            out += list(range(b0 + max(0, i0 - a0), b0 + min(a1, i1) - a0))
        elif tag == "replace":
            out += list(range(b0, b1))
    return (min(out), max(out) + 1) if out else None

def find_tokens(tokens, phrase):
    p = phrase.split()
    for i in range(len(tokens) - len(p) + 1):
        if tokens[i:i + len(p)] == p: return (i, i + len(p))
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True); ap.add_argument("--asr", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--cap", type=float, default=0.5)
    a = ap.parse_args()
    rows = {r["id"]: r for f in a.rows for r in read_jsonl(f)}
    outs = [o for o in read_jsonl(a.asr) if "asr" in o and o["id"] in rows]
    existing = read_jsonl(a.out)
    have = {e["parent_id"] for e in existing}
    seen = {(e["label"], strip_filler(e["text"])) for e in existing} | {(r["label"], strip_filler(r["text"])) for r in rows.values()}
    conf, stats, keep = Counter(), Counter(), []
    for o in outs:
        r = rows[o["id"]]; hyp = o["asr"].strip()
        if r["id"] in have: continue
        s, h, ops = align(r["text"], hyp)
        for tag, a0, a1, b0, b1 in ops:
            if tag == "replace": conf[(" ".join(s[a0:a1]), " ".join(h[b0:b1]))] += 1
            elif tag == "delete": conf[(" ".join(s[a0:a1]), "")] += 1
        if not h: stats["empty"] += 1; continue
        if norm(hyp) == norm(r["text"]): stats["unchanged"] += 1; continue
        lab = r["label"]; t = norm(hyp)
        req = REQUIRE.get(lab)
        if req:
            must, mustnot = req
            if mustnot and re.search(mustnot, t) and not re.search(mustnot, norm(r["text"])):
                stats["sibling_word_appeared"] += 1; continue
            if must and not re.search(must, t):
                m = re.search(must, norm(r["text"]))
                span = find_tokens(s, m.group(0)) if m else None
                mapped = map_span(span, ops, h) if span else None
                if not mapped: stats["deciding_word_deleted"] += 1; continue
                stats["deciding_word_misheard_kept"] += 1
        slot = None
        if lab in SLOT_LABELS and r.get("slot_text"):   # unnamed pins / tasks keep an empty slot
            span = find_tokens(s, norm(r["slot_text"]))
            mapped = map_span(span, ops, h) if span else None
            if not mapped: stats["slot_lost"] += 1; continue
            slot = " ".join(h[mapped[0]:mapped[1]])
        key = (lab, strip_filler(t))
        if key in seen: stats["dup"] += 1; continue
        seen.add(key)
        keep.append({"id": r["id"] + "-asr", "parent_id": r["id"], "group": r.get("group", r["id"]), "label": lab,
                     "kind": r.get("kind"), "fam": r.get("fam"), "text": t, "slot_text": slot, "split": r.get("split"),
                     "source": "tts_roundtrip", "src_text": r["text"], **{k: o[k] for k in ("voice", "link", "snr", "avg_logprob", "no_speech_prob")}})
    # cap: round-trip rows <= cap x the label's other rows (DATA_GENERATION.md / plan §3.8)
    base = Counter(r["label"] for r in rows.values())
    have_lab = Counter(e["label"] for e in existing)
    final = []
    for k in keep:
        if have_lab[k["label"]] < a.cap * base[k["label"]]:
            final.append(k); have_lab[k["label"]] += 1
        else: stats["capped"] += 1
    write_jsonl(a.out, final, mode="a")
    old = json.load(open(GEN / "asr_confusions.json")) if (GEN / "asr_confusions.json").exists() else {}
    for (x, y), c in conf.items():
        if x and x != y: old[f"{x} => {y}"] = old.get(f"{x} => {y}", 0) + c
    json.dump(dict(sorted(old.items(), key=lambda kv: -kv[1])), open(GEN / "asr_confusions.json", "w"), indent=0)
    stats["kept"] = len(final); stats["asr_outputs"] = len(outs)
    print(json.dumps(dict(stats), indent=1))

if __name__ == "__main__":
    main()

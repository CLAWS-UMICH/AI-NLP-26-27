"""Build the harder dev set (data/dev_hard.jsonl) from agent proposals in the stage.py format.
Rule checks (dev slot pool, no test names) -> far-from-train gate -> two-step blind relabel -> dedup vs dev.
Guardrails: the frozen test is never read; rows are never kept or dropped by a trained intent model's predictions.
A set tagged "= contrastive" marks rows that also name a sibling ("not the map, the tasks"); those need both labellers.
Usage: python -m datagen.dev_hard --batch K data/gen/staged/devhard_K.jsonl
       python -m datagen.dev_hard --report      (recompute train distance and the summary for the whole set)"""
import argparse, json, re
from collections import Counter, defaultdict
import numpy as np
from .config import (GEN, DEV_HAND, DEV_HARD, REPORTS, LABELS, SLOT_LABELS, SLOT_TYPE, REQUIRE, REQUEST_VERBS,
                     AMBIGUOUS_DROP, OPTIONAL_SLOT, UNHANDLED_KINDS, ROOT)
from .common import read_jsonl, write_jsonl, norm, strip_filler, shingles, jaccard, embed
from .relabel import blind_check, label_many, SECOND

OUT = DEV_HARD
REPORT = REPORTS / "dev_hard_build.json"
MAX_TRAIN_COS = 0.85     # same-label ruler cosine to the nearest train row must stay below this
MAX_TRAIN_JAC = 0.6      # same-label shingle Jaccard to the nearest train row
MAX_DEV_COS = 0.9        # near-duplicates of other dev rows are dropped
SLOTS = json.load(open(ROOT / "datagen/slots.json"))
def pool(split, types): return {n for t in types for n in SLOTS[t][split]}
TRAIN_NAMES = pool("train", ("place", "waypoint", "task"))
TEST_NAMES = pool("test", ("place", "waypoint", "task"))
DEV_NAMES = pool("dev", ("place", "waypoint", "task"))

# guide and card examples: exact (normalised) matches, plus {place}/{waypoint}/{task} templates as regexes
_banned = json.load(open(ROOT / "datagen/banned.json"))
BANNED = {norm(b) for b in _banned if "{" not in b}
BANNED_BARE = {strip_filler(b) for b in _banned if "{" not in b} - {""}   # same, with filler words removed
_any_name = "(" + "|".join(re.escape(n) for n in sorted(DEV_NAMES | TRAIN_NAMES, key=len, reverse=True)) + ")"
BANNED_RE = [re.compile("^" + re.escape(norm(re.sub(r"\{(place|waypoint|task)\}", " slotx ", b))).replace("slotx", _any_name) + "$")
             for b in _banned if "{" in b]   # a template filled with any pool name

def names_in(t, names): return [n for n in names if re.search(rf"\b{re.escape(n)}\b", t)]

def parse(files, batch):
    rows, k = [], 0
    for f in files:
        for line in open(f):
            if not line.strip(): continue
            s = json.loads(line); k += 1
            gid = f"dh{batch}-g{k:04d}"
            for r in s["rows"]:
                lab, kind = r[0], None
                if lab.startswith("unhandled:"): lab, kind = "unhandled", lab.split(":", 1)[1]
                rows.append({"id": f"{gid}-{len(rows)}", "group": gid, "batch": batch, "fam": s["fam"], "kind": kind,
                             "label": lab, "text": r[1].strip(), "slot_text": r[2] if len(r) > 2 else None,
                             "tagged_contrastive": s.get("src") == "contrastive",
                             "source": "agent_dev_hard", "split": "dev_hard"})
    return rows

def rule_check(r):
    """Returns (reason or None, contrastive)."""
    t, lab = r["text"], r["label"]
    if t != t.lower(): return "not_lowercase", False
    if lab not in LABELS: return "bad_label", False
    if lab == "unhandled" and r["kind"] not in UNHANDLED_KINDS: return "unhandled_without_kind", False
    nt = norm(t)
    if nt in BANNED or strip_filler(t) in BANNED_BARE or any(p.match(nt) for p in BANNED_RE): return "banned_example", False
    if names_in(t, TEST_NAMES): return "uses_test_name", False
    if lab in SLOT_LABELS:
        s = r["slot_text"]
        # train names are ordinary words too ("uia", "airlock"); they are only banned where a slot is marked
        if names_in(t, TRAIN_NAMES): return "slot_row_uses_train_name", False
        named = names_in(t, DEV_NAMES)
        if not s:
            if lab not in OPTIONAL_SLOT: return "slot_missing", False
            if named: return "name_in_text_but_empty_slot", False
        else:
            if not re.search(rf"\b{re.escape(s)}\b", t): return "slot_not_verbatim", False
            if s not in pool("dev", SLOT_TYPE[lab]): return "slot_not_in_dev_pool", False
            if [n for n in named if n != s and n not in s]: return "more_than_one_name", False
    elif r["slot_text"]:
        return "slot_on_nonslot_label", False
    if re.search(AMBIGUOUS_DROP, t): return "ambiguous_drop", False
    contrastive = r["tagged_contrastive"]
    must, mustnot = REQUIRE.get(lab, (None, None))
    if must and not re.search(must, t): return "missing_deciding_word", False
    if mustnot and re.search(mustnot, t): contrastive = True
    if lab == "unhandled" and r["kind"] == "report" and REQUEST_VERBS.search(t): return "report_has_request_verb", False
    return None, contrastive

class Ref:
    """Same-label reference rows with embeddings computed once; rows can be appended."""
    def __init__(self, rows):
        self.E, self.S = defaultdict(list), defaultdict(list)
        if rows: self.add(rows)
    def add(self, rows, E=None):
        E = embed([strip_filler(x["text"]) for x in rows]) if E is None else E
        for x, e in zip(rows, E):
            self.E[x["label"]].append(e); self.S[x["label"]].append(shingles(x["text"]))
    def nearest(self, rows, E=None):
        E = embed([strip_filler(r["text"]) for r in rows]) if E is None else E
        out = []
        for r, e in zip(rows, E):
            l = r["label"]
            cos = float((np.stack(self.E[l]) @ e).max()) if self.E[l] else 0.0
            sh = shingles(r["text"])
            out.append((cos, max((jaccard(sh, s) for s in self.S[l]), default=0.0)))
        return out

def train_rows(): return read_jsonl(GEN / "train.jsonl") + read_jsonl(GEN / "train_asr.jsonl")
def old_dev_rows(): return read_jsonl(GEN / "dev.jsonl") + read_jsonl(DEV_HAND) + read_jsonl(GEN / "dev_asr.jsonl")

def summary():
    rows = read_jsonl(OUT)
    if not rows: return {}
    near = Ref(train_rows()).nearest(rows)       # recomputed: train may have grown since a batch was built
    cos = np.array([c for c, _ in near]); jac = np.array([j for _, j in near])
    u = [r for r in rows if r["label"] == "unhandled"]
    return {"total": len(rows), "per_label": dict(sorted(Counter(r["label"] for r in rows).items())),
            "unhandled_share": round(len(u) / len(rows), 3), "unhandled_kinds": dict(Counter(r["kind"] for r in u)),
            "contrastive_share": round(float(np.mean([r["contrastive"] for r in rows])), 3),
            "empty_slot_rows": sum(1 for r in rows if r["label"] in OPTIONAL_SLOT and not r["slot_text"]),
            "train_cos": {"p50": round(float(np.median(cos)), 3), "p95": round(float(np.percentile(cos, 95)), 3), "max": round(float(cos.max()), 3),
                          "over_limit": int((cos >= MAX_TRAIN_COS).sum())},
            "train_jac": {"p50": round(float(np.median(jac)), 3), "p95": round(float(np.percentile(jac, 95)), 3), "max": round(float(jac.max()), 3)}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int); ap.add_argument("files", nargs="*")
    ap.add_argument("--check-only", action="store_true"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    rep = json.load(open(REPORT)) if REPORT.exists() else {"batches": {}}
    if a.report:
        rep["summary"] = summary(); json.dump(rep, open(REPORT, "w"), indent=1)
        print(json.dumps(rep["summary"], indent=1)); return
    rows = parse(a.files, a.batch)
    stats, rejected, keep = Counter(proposed=len(rows)), [], []
    for r in rows:
        why, r["contrastive"] = rule_check(r)
        if why: r["reject"] = why; rejected.append(r); stats["rule:" + why] += 1
        else: keep.append(r)
    if a.check_only:
        for r in rejected: print(f"  {r['reject']:28s} {r['label']:28s} {r['text']}")
        print(dict(stats), "ok rows:", len(keep)); return
    if str(a.batch) in rep["batches"]: raise SystemExit(f"batch {a.batch} already built; use a new number")

    # 1. far from train: the model must not have seen anything this close
    E = embed([strip_filler(r["text"]) for r in keep])
    kept, kept_E = [], []
    for r, e, (cos, jac) in zip(keep, E, Ref(train_rows()).nearest(keep, E)):
        r["train_cos"], r["train_jac"] = round(cos, 3), round(jac, 3)
        if cos >= MAX_TRAIN_COS or jac >= MAX_TRAIN_JAC:
            r["reject"] = "too_close_to_train"; rejected.append(r); stats["too_close_to_train"] += 1
        else: kept.append(r); kept_E.append(e)

    # 2. blind relabel; contrastive rows need both labellers to agree
    ok, disputed = blind_check(kept)
    for r in disputed:
        r["reject"] = f"relabel:{r['relabel_8b']}/{r['relabel_14b']}"; rejected.append(r); stats["relabel_disagree"] += 1
    contr = [r for r in ok if r["contrastive"] and "relabel_14b" not in r]
    for r, p in zip(contr, label_many([r["text"] for r in contr], workers=2, model=SECOND) if contr else []): r["relabel_14b"] = p
    E_of = {r["id"]: e for r, e in zip(kept, kept_E)}
    labelled = []
    for r in ok:
        if r["contrastive"] and not (r.get("relabel_8b") == r["label"] == r.get("relabel_14b")):
            r["reject"] = "contrastive_needs_both"; rejected.append(r); stats["contrastive_needs_both"] += 1
        else: labelled.append(r)

    # 3. dedup vs every dev row (old dev, earlier batches, this batch), only among rows that passed the labellers
    ref = Ref(old_dev_rows() + read_jsonl(OUT))
    final = []
    for r in labelled:
        e = E_of[r["id"]]
        cos, _ = ref.nearest([r], e[None])[0]
        if cos >= MAX_DEV_COS: r["reject"] = "dup_of_dev"; rejected.append(r); stats["dup_of_dev"] += 1
        else: final.append(r); ref.add([r], e[None])

    for r in final: r.pop("tagged_contrastive", None)
    write_jsonl(OUT, final, mode="a")
    write_jsonl(GEN / f"dev_hard_rejected_b{a.batch}.jsonl", rejected)
    stats["kept"] = len(final)
    rep["batches"][str(a.batch)] = dict(stats)
    rep["summary"] = summary()
    json.dump(rep, open(REPORT, "w"), indent=1)
    print(dict(stats))
    for r in rejected: print(f"  {r['reject']:28s} {r['label']:28s} {r.get('train_cos', ''):6} {r['text']}")

if __name__ == "__main__":
    main()

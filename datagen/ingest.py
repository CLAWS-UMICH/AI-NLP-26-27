"""Ingest one round of agent proposals: rule checks -> blind relabel -> dedup gate.
The frozen test is never read here: test rows are not compared with train or dev rows.
Usage: python -m datagen.ingest --round 0 data/gen/staged/r0_*.jsonl [--dev-holdout 150]"""
import argparse, json, random, re, sys
from collections import Counter, defaultdict
import numpy as np
from .config import (GEN, DEV_HAND, DEV_HARD, DEV_GAP_COSINE, REPORTS, LABELS, FAMILY_OF, SLOT_LABELS, SLOT_TYPE, GRID,
                     REQUIRE, REQUEST_VERBS, AMBIGUOUS_DROP, OPTIONAL_SLOT, DEDUP_JACCARD, DEDUP_COSINE, ROOT)
from .common import read_jsonl, write_jsonl, norm, strip_filler, shingles, jaccard, embed, minhash
from .relabel import blind_check

SLOTS = json.load(open(ROOT / "datagen/slots.json"))
BANNED = set(json.load(open(ROOT / "datagen/banned.json")))
def pool(split, types): return {n for t in types for n in SLOTS[t][split]}
NONTRAIN_NAMES = pool("dev", ("place", "waypoint", "task")) | pool("test", ("place", "waypoint", "task"))

def parse(files, rnd):
    rows, k = [], 0
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            s = json.loads(line); k += 1
            gid = f"r{rnd}-g{k:05d}"
            cell = s.get("cell")
            for r in s["rows"]:
                lab, text = r[0], r[1]
                kind = None
                if lab.startswith("unhandled:"):
                    lab, kind = "unhandled", lab.split(":", 1)[1]
                rows.append({"id": f"{gid}-{len(rows)}", "group": gid if s["fam"] != "unhandled" else f"{gid}-{len(rows)}", "round": rnd, "fam": s["fam"], "cell": cell,
                             "label": lab, "kind": kind, "text": text.strip(), "slot_text": r[2] if len(r) > 2 else None,
                             "src": s.get("src", "grid"), "source": "agent"})
    return rows

def rule_check(r):
    t, lab = r["text"], r["label"]
    if t != t.lower(): return "not_lowercase"
    if lab not in LABELS: return "bad_label"
    if r["src"] != "appendix_a" and (norm(t) in BANNED or t in BANNED): return "banned_example"
    if lab in SLOT_LABELS:
        s = r["slot_text"]
        if not s: return None if lab in OPTIONAL_SLOT else "slot_missing"
        if not re.search(rf"\b{re.escape(s)}\b", t): return "slot_not_verbatim"
        if s not in pool("train", SLOT_TYPE[lab]): return "slot_not_in_train_pool"
    elif r["slot_text"]:
        return "slot_on_nonslot_label"
    if any(re.search(rf"\b{re.escape(n)}\b", t) for n in NONTRAIN_NAMES): return "uses_heldout_name"
    if re.search(AMBIGUOUS_DROP, t): return "ambiguous_drop"
    req = REQUIRE.get(lab)
    if req:
        must, mustnot = req
        if must and not re.search(must, t): return "missing_deciding_word"
        if mustnot and re.search(mustnot, t): return "conflicting_deciding_word"
    if lab == "unhandled" and r["kind"] == "report" and REQUEST_VERBS.search(t): return "report_has_request_verb"
    if r["cell"] and r["src"] == "grid" and lab != "unhandled":
        n = len(norm(t).split()); want = r["cell"][4]
        if want == "1-3" and n > 5: return "length_mismatch"
        if want == "8+" and n < 6: return "length_mismatch"
    return None

def gate(keep, stats, rejected):
    """Dedup gate (DATA_GENERATION.md §5): same label, vs train + dev + this batch, using datasketch MinHashLSH
    (Jaccard >= 0.8 on filler-stripped shingles, confirmed exactly) and sentence-transformers util.semantic_search
    (ruler cosine >= 0.9 after filler removal)."""
    from datasketch import MinHashLSH
    from sentence_transformers import util
    import torch
    train = read_jsonl(GEN / "train.jsonl"); dev = read_jsonl(GEN / "dev.jsonl") + read_jsonl(DEV_HAND)
    dev_hard = read_jsonl(DEV_HARD)
    def index(rows, tag):
        lsh, sh, emb = defaultdict(lambda: MinHashLSH(threshold=DEDUP_JACCARD, num_perm=128)), {}, defaultdict(list)
        E = embed([strip_filler(x["text"]) for x in rows]) if rows else np.zeros((0, 768), dtype=np.float32)
        for i, x in enumerate(rows):
            key = f"{tag}{i}"; sh[key] = shingles(x["text"])
            lsh[x["label"]].insert(key, minhash(sh[key])); emb[x["label"]].append(E[i])
        return lsh, sh, emb
    lsh_ref, sh_ref, emb_ref = index(train + dev, "r")
    lsh_hard, sh_hard, emb_hard = index(dev_hard, "h")
    def near(lab, s_new, m_new, e_new, lsh, sh, emb):
        jac = max((jaccard(s_new, sh[k]) for k in lsh[lab].query(m_new)), default=0.0) if lab in lsh else 0.0
        cos = 0.0
        if emb.get(lab):
            hit = util.semantic_search(torch.tensor(e_new[None]), torch.tensor(np.stack(emb[lab])), top_k=1)[0]
            cos = hit[0]["score"] if hit else 0.0
        return jac, cos
    stripped_label = {strip_filler(x["text"]): x["label"] for x in train + dev}
    E_new = embed([strip_filler(r["text"]) for r in keep])
    final = []
    for j, r in enumerate(keep):
        lab, e, sh_new = r["label"], E_new[j], shingles(r["text"])
        m_new = minhash(sh_new); st = strip_filler(r["text"])
        if st in stripped_label and stripped_label[st] != lab:
            r["reject"] = "cross_label_conflict"; rejected.append(r); stats["dedup:conflict"] += 1; continue
        jac, cos = near(lab, sh_new, m_new, e, lsh_ref, sh_ref, emb_ref)
        if cos >= DEDUP_COSINE or jac >= DEDUP_JACCARD:
            r["reject"] = "dedup"; rejected.append(r); stats["dedup:near_dup"] += 1; continue
        _, cos_h = near(lab, sh_new, m_new, e, lsh_hard, sh_hard, emb_hard)
        if cos_h >= DEV_GAP_COSINE:   # keeps train from drifting onto dev_hard phrasings
            r["reject"] = "too_close_to_dev_hard"; rejected.append(r); stats["dedup:near_dev_hard"] += 1; continue
        key = f"n{j}"; sh_ref[key] = sh_new; lsh_ref[lab].insert(key, m_new); emb_ref[lab].append(e)
        stripped_label[st] = lab
        final.append(r)
    return final

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--dev-holdout", type=int, default=0)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--reingest", help="rejected_rK.jsonl: re-run relabel-rejected rows, keeping their groups and splits")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    rnd = a.round
    if a.reingest:
        return reingest(a.reingest, rnd)
    rows = parse(a.files, rnd)
    stats = Counter(proposed=len(rows))
    rejected = []

    # 1. rule checks
    keep = []
    for r in rows:
        why = rule_check(r)
        if why: r["reject"] = why; rejected.append(r); stats["rule:" + why] += 1
        else: keep.append(r)
    if a.check_only:
        for r in rejected: print(f"  {r['reject']:28s} {r['label']:28s} {r['text']}")
        print(dict(stats), "ok rows:", len(keep)); return

    # 2. blind relabel (the labellers never see r["label"])
    total = Counter(r["label"] for r in keep)
    keep, disputed = blind_check(keep)
    agree = Counter(r["label"] for r in keep)
    for r in disputed:
        r["reject"] = f"relabel:{r['relabel_8b']}/{r['relabel_14b']}"; rejected.append(r); stats["relabel_disagree"] += 1
    stats["kept_by_second_look"] = sum(1 for r in keep if "relabel_14b" in r)

    final = gate(keep, stats, rejected)

    # Optional: hold out whole groups for dev (round 0)
    dev_rows = []
    if a.dev_holdout:
        random.seed(rnd)
        groups = defaultdict(list)
        for r in final: groups[r["group"]].append(r)
        need = Counter({l: max(1, a.dev_holdout // len(LABELS)) for l in LABELS})
        order = list(groups); random.shuffle(order)
        chosen = set()
        for g in order:
            if sum(len(groups[c]) for c in chosen) >= a.dev_holdout: break
            if any(need[r["label"]] > 0 for r in groups[g]):
                chosen.add(g)
                for r in groups[g]: need[r["label"]] -= 1
        dev_rows = [dict(r, split="dev") for g in chosen for r in groups[g]]
        final = [r for r in final if r["group"] not in chosen]
    for r in final: r["split"] = "train"

    write_jsonl(GEN / "train.jsonl", final, mode="a")
    if dev_rows: write_jsonl(GEN / "dev.jsonl", dev_rows, mode="a")
    write_jsonl(GEN / f"rejected_r{rnd}.jsonl", rejected)
    # coverage: family x cell
    cov = json.load(open(GEN / "coverage.json")) if (GEN / "coverage.json").exists() else {}
    for r in final + dev_rows:
        if r["cell"] and r["label"] != "unhandled":
            key = r["fam"] + "|" + "|".join(r["cell"])
            cov[key] = cov.get(key, 0) + 1
    json.dump(cov, open(GEN / "coverage.json", "w"))
    stats.update(accepted_train=len(final), accepted_dev=len(dev_rows))
    out = {"round": rnd, "stats": dict(stats),
           "relabel_agreement": {l: f"{agree[l]}/{total[l]}" for l in sorted(total)},
           "accepted_per_label": dict(Counter(r["label"] for r in final + dev_rows))}
    REPORTS.mkdir(exist_ok=True)
    json.dump(out, open(REPORTS / f"round{rnd}_ingest.json", "w"), indent=1)
    print(json.dumps(out["stats"], indent=1))
    low = {l: v for l, v in out["relabel_agreement"].items() if int(v.split('/')[0]) < 0.7 * int(v.split('/')[1])}
    print("labels with relabel agreement < 70%:", low)

def reingest(path, rnd):
    rows = [r for r in read_jsonl(path) if r.get("reject", "").startswith("relabel:")]
    for r in rows:
        for k in ("reject", "relabel_8b", "relabel_14b"): r.pop(k, None)
    stats, rejected = Counter(proposed=len(rows)), []
    keep, disputed = blind_check(rows)
    for r in disputed:
        r["reject"] = f"relabel:{r['relabel_8b']}/{r['relabel_14b']}"; rejected.append(r); stats["relabel_disagree"] += 1
    final = gate(keep, stats, rejected)
    dev_groups = {r["group"] for r in read_jsonl(GEN / "dev.jsonl")}
    tr = [dict(r, split="train") for r in final if r["group"] not in dev_groups]
    dv = [dict(r, split="dev") for r in final if r["group"] in dev_groups]
    write_jsonl(GEN / "train.jsonl", tr, mode="a"); write_jsonl(GEN / "dev.jsonl", dv, mode="a")
    others = [r for r in read_jsonl(path) if not r.get("reject", "").startswith("relabel:")]
    write_jsonl(path, others + rejected)
    cov = json.load(open(GEN / "coverage.json"))
    for r in tr + dv:
        if r["cell"] and r["label"] != "unhandled":
            key = r["fam"] + "|" + "|".join(r["cell"]); cov[key] = cov.get(key, 0) + 1
    json.dump(cov, open(GEN / "coverage.json", "w"))
    stats.update(accepted_train=len(tr), accepted_dev=len(dv))
    print(json.dumps(dict(stats), indent=1))
    print("still disputed by label:", Counter(r["label"] for r in disputed).most_common(10))

if __name__ == "__main__":
    main()

"""Work orders for the agent (DATA_GENERATION.md §2 and §7).
Round 0: N emptiest grid cells per family.  Later rounds: budget split from the last report:
~60% worst confusion pairs, ~25% hard examples (wrong / low-margin dev rows), ~15% failing CheckList tests + empty cells."""
import argparse, itertools, json, random
from collections import Counter
from .config import GEN, REPORTS, FAMILIES, GRID

CELLS = list(itertools.product(*GRID.values()))

def emptiest(fam, n, cov, rng):
    counts = {c: cov.get(fam + "|" + "|".join(c), 0) for c in CELLS}
    picked, used = [], Counter()
    for _ in range(n):  # greedy: lowest count, then spread each dimension's values
        best = min(CELLS, key=lambda c: (counts[c], sum(used[(i, v)] for i, v in enumerate(c)), rng.random()))
        picked.append(best); counts[best] += 1
        for i, v in enumerate(best): used[(i, v)] += 1
    return picked

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--sets", type=int, default=45, help="aligned sets per family (round 0)")
    ap.add_argument("--budget", type=int, default=900, help="rows to spend (rounds >= 1)")
    a = ap.parse_args()
    rng = random.Random(a.round)
    cov = json.load(open(GEN / "coverage.json")) if (GEN / "coverage.json").exists() else {}
    order = {"round": a.round, "cells": {}}
    fams = [f for f in FAMILIES if f != "unhandled"]
    if a.round == 0:
        for f in fams:
            order["cells"][f] = ["|".join(c) for c in emptiest(f, a.sets, cov, rng)]
    else:
        rep = json.load(open(REPORTS / f"round{a.round - 1}_eval.json"))
        pairs = rep["top_confusions"][:8]                      # [[true, pred, count], ...]
        fam_err = rep["family_error"]                           # {fam: error rate}
        set_size = {f: len(FAMILIES[f]) + 1 for f in fams}
        b_pairs, b_hard, b_rest = int(a.budget * .6), int(a.budget * .25), a.budget - int(a.budget * .6) - int(a.budget * .25)
        # pairs: weight by count; each pair -> its family's aligned sets in emptiest cells
        tot = sum(p[2] for p in pairs) or 1
        alloc = Counter()
        for t, p, c in pairs:
            from .config import FAMILY_OF
            alloc[FAMILY_OF[t]] += b_pairs * c / tot
        # rest: worst families by error + emptiest cells
        worst = sorted(fam_err, key=fam_err.get, reverse=True)
        for f in worst[:4]:
            if f in set_size: alloc[f] += b_rest / 2 / 4
        for f in fams:
            alloc[f] += b_rest / 2 / len(fams)
        for f, rows in alloc.items():
            if f in set_size:
                n = max(1, round(rows / set_size[f]))
                order["cells"][f] = ["|".join(c) for c in emptiest(f, n, cov, rng)]
        order["confusion_pairs"] = pairs
        order["hard_examples"] = rep["hard_examples"][:max(1, b_hard // 3)]   # write ~3 variants each
        order["failing_checklist"] = {k: v for k, v in rep.get("checklist", {}).items() if v.get("fail_rate", 0) > 0.02}
        order["diversity_worst"] = rep.get("diversity_worst", [])
    json.dump(order, open(GEN / f"workorder_r{a.round}.json", "w"), indent=1)
    for f, cs in order["cells"].items():
        print(f"## {f} ({len(cs)} sets)")
        for i in range(0, len(cs), 6): print("  " + "  ".join(cs[i:i + 6]))
    for k in ("confusion_pairs", "hard_examples", "failing_checklist", "diversity_worst"):
        if k in order: print(f"## {k}:", json.dumps(order[k])[:3000])

if __name__ == "__main__":
    main()

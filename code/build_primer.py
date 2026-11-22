"""
Build data/astronaut_primer.json (the pairs prime.py trains on) from:

- data/suits_glossary.json, two pairs per row:
    term:  "UIA"                        -> "umbilical interface assembly: the airlock switch panel that ..."
    usage: "EV1, BATT to UMB"           -> "first crew member, switch the battery to umbilical power"
- data/nasa_acronyms_eva.json (fetch_acronyms.py), one pair per row:
    acronym: "LCUG"                     -> "Liquid Cooling Undergarment"

Each anchor and each positive appears once. prime.py uses the rest of the batch as negatives, so a repeated
positive (or an acronym with two expansions) would be scored as both right and wrong in the same batch.
Apollo transcripts are deliberately NOT used here; they're a style reference for the handwritten test set.

Usage:
    python code/build_primer.py
"""

import json
from collections import Counter

from config import ACRONYMS_FILE, GLOSSARY_FILE, PRIMER_FILE


def main():
    glossary = json.loads(GLOSSARY_FILE.read_text(encoding="utf-8"))
    acronyms = json.loads(ACRONYMS_FILE.read_text(encoding="utf-8")) if ACRONYMS_FILE.exists() else []
    if not acronyms:
        print(f"note: no {ACRONYMS_FILE.name}, run python code/fetch_acronyms.py to add NASA acronyms")

    candidates = []
    for row in glossary:
        topic = f"glossary/{row['category']}"
        candidates.append(("term", topic, row["term"], f"{row['expansion']}: {row['definition']}", row["source"]))
        if row.get("example"):
            jargon, plain = row["example"]
            candidates.append(("usage", topic, jargon, plain, row["source"]))
    for a in acronyms:
        candidates.append(("acronym", "nasa-acronyms", a["abbreviation"], a["expansion"], f"NASA-Acronyms/{a['source']}"))

    out, anchors, positives = [], set(), set()
    for kind, topic, anchor, positive, source in candidates:
        a, p = anchor.strip().lower(), positive.strip().lower()
        if not a or not p or a == p or a in anchors or p in positives:
            continue
        anchors.add(a)
        positives.add(p)
        out.append({"id": f"p{len(out):04d}", "type": kind, "topic": topic,
                    "anchor": anchor.strip(), "positive": positive.strip(), "source": source})

    PRIMER_FILE.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(out)} pairs in {PRIMER_FILE} ({dict(Counter(x['type'] for x in out))}), "
          f"{len(candidates) - len(out)} dropped as repeats")


if __name__ == "__main__":
    main()

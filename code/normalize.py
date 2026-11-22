"""
Glossary normalization: rewrite known jargon and Whisper spellings to one plain form before classifying.

- Uses rows of data/suits_glossary.json with "normalize": true. Their term and every alias map to the expansion:
  "UIA" / "u i a" / "uya" -> "umbilical interface assembly", "o two" -> "oxygen", "scrubber eh" -> "scrubber a"
- Rows with "normalize": false are ambiguous in everyday speech ("sec", "met", "hr") and are left alone
- Longest match wins, whole words only, one pass (expansions are never re-scanned)
- train.py copies the glossary into the run folder and predict.py loads that copy, so a run always
  normalizes exactly the way it was trained even if the glossary is edited later

Usage:
    python code/normalize.py "check the UIA and o two"
"""

import json
import re
import sys

from config import GLOSSARY_FILE


class Normalizer:
    def __init__(self, glossary_path=GLOSSARY_FILE):
        self.mapping = {}
        if glossary_path and glossary_path.exists():
            for row in json.loads(glossary_path.read_text(encoding="utf-8")):
                if not row.get("normalize"):
                    continue
                for form in [row["term"], *row.get("aliases", [])]:
                    form = form.lower().strip()
                    if form and form != row["expansion"].lower():
                        self.mapping[form] = row["expansion"].lower()
        forms = sorted(self.mapping, key=len, reverse=True)
        self.pattern = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, forms)) + r")(?!\w)", re.I) if forms else None

    def __bool__(self):
        return self.pattern is not None

    def __call__(self, text):
        if self.pattern is None:
            return text
        return self.pattern.sub(lambda m: self.mapping[m.group(0).lower()], text)


def term_forms(glossary_path=GLOSSARY_FILE):
    """{form: expansion} for every glossary form that names a term, plus one regex over them (longest first).
    Radio words are skipped, and ambiguous short forms ("sec", "hr") only count when "normalize" is true,
    the same rule generate.py uses for jargon swaps. split_seed.py and paraphrase.py mask these."""
    forms = {}
    if glossary_path and glossary_path.exists():
        for row in json.loads(glossary_path.read_text(encoding="utf-8")):
            if row["category"] == "radio":
                continue
            names = [row["term"], *row.get("aliases", [])] if row.get("normalize") else []
            for form in [row["expansion"], *names]:
                forms.setdefault(form.lower().strip(), row["expansion"].lower())
    forms.pop("", None)
    ordered = sorted(forms, key=len, reverse=True)
    pattern = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, ordered)) + r")(?!\w)", re.I) if ordered else None
    return forms, pattern


if __name__ == "__main__":
    print(Normalizer()(" ".join(sys.argv[1:])))

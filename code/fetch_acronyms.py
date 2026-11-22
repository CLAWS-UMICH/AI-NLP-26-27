"""
Pull the EVA / suit / life-support slice of nasa/NASA-Acronyms (https://github.com/nasa/NASA-Acronyms).

- Downloads lists/acronyms.json (~49k entries) and keeps only human-spaceflight sources (HSF, AES, LaRC, buffer)
- Keeps an entry when its expansion mentions suit/EVA/life-support words (KEEP) and none of the
  launch/ground/program words that make most matches noise (DROP)
- Drops terms already in data/suits_glossary.json (the hand-written definitions win)
- Writes data/nasa_acronyms_eva.json: [{"abbreviation", "expansion", "source"}], a few hundred rows

Usage:
    python code/fetch_acronyms.py
"""

import json
import re
import urllib.request

from config import ACRONYMS_FILE, GLOSSARY_FILE

URL = "https://raw.githubusercontent.com/nasa/NASA-Acronyms/master/lists/acronyms.json"
SOURCES = {"HSF", "AES", "LaRC", "buffer"}
KEEP = re.compile(
    r"\b(eva|extravehicular|space ?suits?|emu|xemu|life support|plss|portable life|oxygen|carbon dioxide|co2|"
    r"scrubber|coolant|cooling|helmet|glove|airlock|umbilical|tether|lunar surface|lunar terrain|rover|lander|"
    r"ventilation|metabolic|heart rate|caution and warning|spacewalk|egress|ingress|depress\w*|regolith|"
    r"sublimator|display and control|crewmember)\b", re.I)
DROP = re.compile(
    r"\b(launch|propellant|fuel|engine|rocket|booster|ground|facility|facilities|contract|budget|procurement|"
    r"tunnel|aircraft|ozone|atmospher\w*|satellite|orbiter|shuttle|payload bay|liquid oxygen|lox|hydrogen|panel|"
    r"office|program|project|working group|team|demonstration|experiment|board|review|plan|chamber|test|mars|"
    r"network|mapping|kerosene|methane|turbine|combust\w*|transporter|mission control|requirements?|"
    r"specification|standards?|handbook|lab\w*|center|vapor|corrosive|toxic|support equipment|loading|"
    r"white glove|cardiac|columbus|dragonfly|rotor|materials)\b", re.I)


def main():
    with urllib.request.urlopen(URL, timeout=60) as resp:
        entries = json.load(resp)

    glossary = json.loads(GLOSSARY_FILE.read_text(encoding="utf-8")) if GLOSSARY_FILE.exists() else []
    known = {f.lower() for row in glossary for f in [row["term"], *row.get("aliases", [])]}

    out, seen = [], set()
    for e in entries:
        abbr = e["abbreviation"].strip()
        expansion = re.sub(r"\s*\(as in.*?\)", "", e["expansion"], flags=re.I).strip()
        if e["source"] not in SOURCES or not KEEP.search(expansion) or DROP.search(expansion):
            continue
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9/&\-]{1,9}", abbr) or abbr.lower() in known:
            continue
        key = (abbr.lower(), expansion.lower())
        if key not in seen:
            seen.add(key)
            out.append({"abbreviation": abbr, "expansion": expansion, "source": e["source"]})

    out.sort(key=lambda x: (x["abbreviation"].lower(), x["expansion"].lower()))
    ACRONYMS_FILE.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(entries)} NASA acronyms -> {len(out)} EVA/suit/life-support rows in {ACRONYMS_FILE}")


if __name__ == "__main__":
    main()

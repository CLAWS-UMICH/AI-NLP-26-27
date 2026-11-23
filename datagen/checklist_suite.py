"""CheckList suite built from DEV rows only (never test). Runs in .venv-checklist.
  build: python datagen/checklist_suite.py build --round K   -> data/gen/checklist/rK/{suite.pkl,raw.txt}
  score: python datagen/checklist_suite.py score --round K   -> reports/roundK_checklist.json
Tests (DATA_GENERATION.md §6): word-swap, ASR-noise, report -> unhandled, plus negation, filler, opener, contractions, slot swap."""
import argparse, json, random, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from datagen.config import GEN, DEV_HAND, DEV_HARD, REPORTS, SLOT_LABELS, SLOT_TYPE, PRI, SEC, GAS, LIQ

SWAP = {  # label -> (sibling label, [(from_regex, to_word), ...])
    "get_oxy_pri_storage": ("get_oxy_sec_storage", [(r"\bprimary\b", "secondary"), (r"\bpri\b", "sec"), (r"\bmain\b", "backup"), (r"\bfirst\b", "second")]),
    "get_oxy_sec_storage": ("get_oxy_pri_storage", [(r"\bsecondary\b", "primary"), (r"\bsec\b", "pri"), (r"\bbackup\b", "main"), (r"\bsecond\b", "first")]),
    "get_oxy_pri_pressure": ("get_oxy_sec_pressure", [(r"\bprimary\b", "secondary"), (r"\bpri\b", "sec"), (r"\bmain\b", "backup")]),
    "get_oxy_sec_pressure": ("get_oxy_pri_pressure", [(r"\bsecondary\b", "primary"), (r"\bsec\b", "pri"), (r"\bbackup\b", "main")]),
    "get_fan_pri_rpm": ("get_fan_sec_rpm", [(r"\bprimary\b", "secondary"), (r"\bpri\b", "sec"), (r"\bmain\b", "backup"), (r"\bfan one\b", "fan two")]),
    "get_fan_sec_rpm": ("get_fan_pri_rpm", [(r"\bsecondary\b", "primary"), (r"\bsec\b", "pri"), (r"\bbackup\b", "main"), (r"\bfan two\b", "fan one")]),
    "get_primary_battery_level": ("get_secondary_battery_level", [(r"\bprimary\b", "secondary"), (r"\bpri\b", "sec"), (r"\bmain\b", "backup"), (r"\bbattery one\b", "battery two")]),
    "get_secondary_battery_level": ("get_primary_battery_level", [(r"\bsecondary\b", "primary"), (r"\bsec\b", "pri"), (r"\bbackup\b", "main"), (r"\bbattery two\b", "battery one")]),
    "get_scrubber_a_co2_storage": ("get_scrubber_b_co2_storage", [(r"\bscrubber a\b", "scrubber b"), (r"\balpha\b", "bravo"), (r"\ba scrubber\b", "b scrubber")]),
    "get_scrubber_b_co2_storage": ("get_scrubber_a_co2_storage", [(r"\bscrubber b\b", "scrubber a"), (r"\bbravo\b", "alpha"), (r"\bb scrubber\b", "a scrubber")]),
    "get_coolant_gas_pressure": ("get_coolant_liquid_pressure", [(r"\bgas\b", "liquid"), (r"\bvapor\b", "liquid")]),
    "get_coolant_liquid_pressure": ("get_coolant_gas_pressure", [(r"\bliquid\b", "gas"), (r"\bwater\b", "gas")]),
    "get_suit_pressure_co2": ("get_helmet_pressure_co2", [(r"\bsuit\b", "helmet")]),
    "get_helmet_pressure_co2": ("get_suit_pressure_co2", [(r"\bhelmet\b", "suit")]),
}
OPENER = re.compile(r"^(okay|roger|houston)\b[,]?\s*")
VERB_START = re.compile(r"^(show|open|bring|pull|display|read|check|take|navigate|add|delete|remove|mark|close|put|set|give|tell|report|list|route|locate|find|recalculate)\b")
DECIDING = re.compile("|".join([PRI, SEC, GAS, LIQ, r"\b(scrubber|alpha|bravo|helmet|suit|a|b)\b"]))

def jl(p): return [json.loads(l) for l in open(p)] if Path(p).exists() else []

def build(rnd):
    from checklist.test_types import MFT, INV
    from checklist.test_suite import TestSuite
    from checklist.perturb import Perturb
    rng = random.Random(rnd)
    dev = jl(GEN / "dev.jsonl") + jl(DEV_HAND) + jl(DEV_HARD)
    conf = json.load(open(GEN / "asr_confusions.json")) if (GEN / "asr_confusions.json").exists() else {}
    subs = [(k.split(" => ")[0], k.split(" => ")[1]) for k, c in conf.items() if c >= 2 and " => " in k and k.split(" => ")[1]]
    subs = [(a, b) for a, b in subs if not DECIDING.search(a)]
    slots = json.load(open(ROOT / "datagen/slots.json"))
    cmd = [r for r in dev if r["label"] != "unhandled"]
    suite = TestSuite()
    _add = suite.add
    def add(test, **kw):
        if len(test.data): _add(test, **kw)
    suite.add = add
    # report -> unhandled
    unh = [r["text"] for r in dev if r["label"] == "unhandled"]
    suite.add(MFT(unh, labels="unhandled", name="report_unhandled", capability="unhandled"))
    # negation -> unhandled
    neg = []
    for r in cmd:
        core = OPENER.sub("", r["text"])
        if VERB_START.match(core): neg.append("don't " + core)
    suite.add(MFT(neg, labels="unhandled", name="negation_unhandled", capability="unhandled"))
    # word swap -> sibling label
    sw_data, sw_labels = [], []
    for r in cmd:
        if r["label"] not in SWAP: continue
        sib, pats = SWAP[r["label"]]
        for pat, rep in pats:
            if re.search(pat, r["text"]):
                sw_data.append(re.sub(pat, rep, r["text"])); sw_labels.append(sib); break
    suite.add(MFT(sw_data, labels=sw_labels, name="word_swap", capability="which-one"))
    # INV: filler, opener swap, contractions, asr noise, slot swap
    def filler(t):
        core = OPENER.sub("", t); m = OPENER.match(t)
        f = rng.choice(["uh, ", "um, ", "okay so, ", "uh, uh, "])
        return [(m.group(0) if m else "") + f + core]
    def opener(t):
        core = OPENER.sub("", t)
        return [o + core for o in ("okay, ", "roger, ", "houston, ", "") if o + core != t][:2]
    def asr_noise(t):
        out = t; n = 0
        for a, b in rng.sample(subs, min(len(subs), 400)):
            if re.search(rf"\b{re.escape(a)}\b", out):
                out = re.sub(rf"\b{re.escape(a)}\b", b, out, count=1); n += 1
            if n >= 2: break
        return [out] if out != t else []
    def slot_swap(r):
        types = SLOT_TYPE[r["label"]]
        pool = [n for ty in types for n in slots[ty]["dev"] if n != r["slot_text"]]
        return [re.sub(rf"\b{re.escape(r['slot_text'])}\b", rng.choice(pool), r["text"])] if pool and r["slot_text"] else []
    texts = [r["text"] for r in cmd]
    for name, fn in [("inv_filler", filler), ("inv_opener", opener), ("inv_asr_noise", asr_noise)]:
        suite.add(INV(**Perturb.perturb(texts, fn, keep_original=True)), name=name, capability="robustness")
    suite.add(INV(**Perturb.perturb(texts, Perturb.contractions, keep_original=True)), name="inv_contractions", capability="robustness")
    sl = [r for r in cmd if r["label"] in SLOT_LABELS]
    data = [[r["text"]] + slot_swap(r) for r in sl]
    suite.add(INV([d for d in data if len(d) > 1]), name="inv_slot_swap", capability="slots")
    out = GEN / "checklist" / f"r{rnd}"; out.mkdir(parents=True, exist_ok=True)
    suite.to_raw_file(str(out / "raw.txt")); suite.save(str(out / "suite.pkl"))
    print({n: len(t.data) for n, t in suite.tests.items()})

def score(rnd):
    from checklist.test_suite import TestSuite
    d = GEN / "checklist" / f"r{rnd}"
    suite = TestSuite.from_file(str(d / "suite.pkl"))
    suite.run_from_file(str(d / "preds.txt"), file_format="pred_and_conf", overwrite=True)
    res = {}
    for name, t in suite.tests.items():
        st = t.get_stats()
        fails = []
        for i in list(t.fail_idxs())[:12]:
            x = t.data[i]; fails.append(x if isinstance(x, str) else list(x))
        res[name] = {"testcases": st.testcases, "fail_rate": round(st.get("fail_rate", 0) / 100, 4), "fails": st.get("fails", 0), "examples": fails}
    REPORTS.mkdir(exist_ok=True)
    json.dump(res, open(REPORTS / f"round{rnd}_checklist.json", "w"), indent=1)
    for n, v in res.items(): print(f"{n:22s} cases={v['testcases']:4d} fail={v['fail_rate']:.1%}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["build", "score"]); ap.add_argument("--round", type=int, required=True)
    a = ap.parse_args()
    build(a.round) if a.mode == "build" else score(a.round)

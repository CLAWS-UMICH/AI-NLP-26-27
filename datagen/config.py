"""Shared config for the agent data-generation loop (data/DATA_GENERATION.md)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
GEN = DATA / "gen"            # all loop outputs
FROZEN_TEST = DATA / "frozen" / "test.jsonl"
DEV_HAND = DATA / "dev_hand.jsonl"
DEV_HARD = DATA / "dev_hard.jsonl"       # the dev set loop decisions use (far from train by construction)
DEV_HARD_ASR = GEN / "dev_hard_asr.jsonl"
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs"

RULER = ("sentence-transformers/all-mpnet-base-v2", "e8c3b32edf5434bc2275fc9bab85f82640a19130")
STUDENT = ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")

FAMILIES = {
    "oxygen": ["get_oxy_pri_storage", "get_oxy_sec_storage", "get_oxy_pri_pressure", "get_oxy_sec_pressure", "get_oxy_consumption"],
    "suit_atmos": ["get_suit_pressure_oxy", "get_suit_pressure_co2", "get_suit_pressure_other", "get_suit_pressure_total", "get_helmet_pressure_co2"],
    "co2_scrub": ["get_scrubber_a_co2_storage", "get_scrubber_b_co2_storage", "get_co2_production"],
    "coolant": ["get_coolant_storage", "get_coolant_gas_pressure", "get_coolant_liquid_pressure"],
    "fans": ["get_fan_pri_rpm", "get_fan_sec_rpm"],
    "power": ["get_primary_battery_level", "get_secondary_battery_level"],
    "body": ["get_heart_rate", "get_temperature", "get_eva_elapsed_time", "get_warnings"],
    "menus": ["open_menu_vitals", "open_menu_navigation", "open_menu_tasks", "open_menu_uia", "close_menu"],
    "nav": ["Get_coordinates", "Set_navigation_target", "reroute_navigation", "Add_waypoint", "Delete_waypoint"],
    "tasks": ["Complete_task"],
    "unhandled": ["unhandled"],
}
LABELS = [l for f in FAMILIES.values() for l in f]
FAMILY_OF = {l: f for f, ls in FAMILIES.items() for l in ls}
SLOT_LABELS = {"Get_coordinates", "Set_navigation_target", "Add_waypoint", "Delete_waypoint", "Complete_task"}
SLOT_TYPE = {"Get_coordinates": ("place", "waypoint"), "Set_navigation_target": ("place", "waypoint"),
             "Add_waypoint": ("waypoint",), "Delete_waypoint": ("waypoint",), "Complete_task": ("task",)}

# Shape grid (DATA_GENERATION.md §2)
GRID = {
    "opener": ["none", "okay", "roger", "houston"],
    "form": ["question", "command", "clipped", "need"],
    "pos": ["early", "late", "abbrev"],
    "disfl": ["none", "filler", "false_start", "repeat"],
    "len": ["1-3", "4-7", "8+"],
}
UNHANDLED_KINDS = ["report", "chatter", "out_of_scope", "fragment", "negation", "callout", "jargon_trap", "noise",
                   "two_commands", "followup"]

# Filler removed before the dedup comparison (DATA_GENERATION.md §5)
FILLER = ["okay", "ok", "roger", "houston", "uh", "um", "give me", "read", "check", "so", "copy", "please", "hey"]
FILLER_RE = re.compile(r"\b(" + "|".join(re.escape(f) for f in sorted(FILLER, key=len, reverse=True)) + r")\b")

# Deciding-word patterns for the rule checks (DATA_GENERATION.md §3)
PRI = r"\b(primary|pri|pre|main|first|(?<!ev )one|(?<!ev )1)\b"   # not the callsign "ev one"
SEC = r"\b(secondary|sec|sex|second|(?<!o )(?<!oh )two|(?<!o )2|backup)\b"
SCRUB_A = r"\bscrubber (a|alpha|one|1)\b|\b(a|alpha) scrubber\b|\balpha\b"
SCRUB_B = r"\bscrubber (b|be|bee|bravo|two|2)\b|\b(b|bravo) scrubber\b|\bbravo\b"
GAS = r"\b(gas|vapor)\b"
LIQ = r"\b(liquid|water)\b"
HELMET = r"\bhelmet\b"
PRESS = r"\b(pressure|psi)\b"
AMBIGUOUS_DROP = r"\bdrop (the|that|this)\b"  # guide P9: "drop the X waypoint" could mean add or delete
OPTIONAL_SLOT = {"Add_waypoint", "Complete_task"}  # guide P2, P3: no name -> empty slot
REQUIRE = {  # label -> pattern that must match / must not match
    "get_oxy_sec_storage": (SEC, PRI), "get_oxy_sec_pressure": (SEC, PRI), "get_fan_sec_rpm": (SEC, PRI),
    "get_secondary_battery_level": (SEC, PRI),
    "get_oxy_pri_storage": (None, SEC), "get_oxy_pri_pressure": (None, SEC), "get_fan_pri_rpm": (None, SEC),
    "get_primary_battery_level": (None, SEC),
    "get_scrubber_b_co2_storage": (SCRUB_B, SCRUB_A), "get_scrubber_a_co2_storage": (None, SCRUB_B),
    "get_coolant_gas_pressure": (GAS, LIQ), "get_coolant_liquid_pressure": (LIQ + "|" + PRESS, GAS),  # guide P5
    "get_coolant_storage": (None, GAS + "|" + LIQ + "|" + PRESS),
    "get_helmet_pressure_co2": (HELMET, None), "get_suit_pressure_co2": (None, HELMET),
}
REQUEST_VERBS = re.compile(r"\b(what|what's|whats|how|how's|show|give|open|tell|need|take|navigate|set|add|drop|delete|remove|mark|bring|pull|display|can you|could you|where|where's)\b")

ASR_HINT = "Suit telemetry: O2, CO2, scrubber, coolant, UIA, waypoint, primary, secondary."
VOICES = ["Samantha", "Daniel", "Karen", "Moira", "Rishi", "Tessa", "Fred", "Ralph", "Kathy", "Junior"]

# Gates
DEDUP_JACCARD = 0.8
DEDUP_COSINE = 0.9
DEV_GAP_COSINE = 0.85   # new train rows must stay this far from same-label dev_hard rows, so dev stays hard
DIVERSITY_TARGET = 0.5

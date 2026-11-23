"""Convert the agent's compact proposal format to staged JSONL.
  @ family            -> family for following sets
  = cell | hard | misc  -> starts a new aligned set (cell = opener|form|pos|disfl|len)
  code | text [| slot]  -> one row; code is a short label code (see CODES) or a full label; u / u:kind = unhandled
Usage: python -m datagen.stage in.txt out.jsonl"""
import json, sys
from .config import LABELS
CODES = {"ps": "get_oxy_pri_storage", "ss": "get_oxy_sec_storage", "pp": "get_oxy_pri_pressure", "sp": "get_oxy_sec_pressure",
         "oc": "get_oxy_consumption", "so": "get_suit_pressure_oxy", "sc": "get_suit_pressure_co2", "sx": "get_suit_pressure_other",
         "st": "get_suit_pressure_total", "hc": "get_helmet_pressure_co2", "sa": "get_scrubber_a_co2_storage",
         "sb": "get_scrubber_b_co2_storage", "cp": "get_co2_production", "cs": "get_coolant_storage", "cg": "get_coolant_gas_pressure",
         "cl": "get_coolant_liquid_pressure", "fp": "get_fan_pri_rpm", "fs": "get_fan_sec_rpm", "bp": "get_primary_battery_level",
         "bs": "get_secondary_battery_level", "hr": "get_heart_rate", "tp": "get_temperature", "et": "get_eva_elapsed_time",
         "wn": "get_warnings", "mv": "open_menu_vitals", "mn": "open_menu_navigation", "mt": "open_menu_tasks", "mu": "open_menu_uia",
         "mc": "close_menu", "gc": "Get_coordinates", "sn": "Set_navigation_target", "rr": "reroute_navigation", "aw": "Add_waypoint",
         "dw": "Delete_waypoint", "ct": "Complete_task", "u": "unhandled:report"}

def convert(lines):
    fam, sets, cur = None, [], None
    for n, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"): continue
        if line.startswith("@"):
            fam = line[1:].strip(); continue
        if line.startswith("="):
            tag = line[1:].strip()
            cur = {"fam": fam, "cell": tag.split("|") if tag.count("|") == 4 else None,
                   "src": "grid" if tag.count("|") == 4 else tag, "rows": []}
            sets.append(cur); continue
        parts = [p.strip() for p in line.split(" | ")]
        code = parts[0]
        lab = CODES.get(code) or (("unhandled:" + code[2:]) if code.startswith("u:") else code)
        if lab.split(":")[0] not in LABELS: raise SystemExit(f"line {n}: bad code {code!r}")
        cur["rows"].append([lab, parts[1].lower()] + ([parts[2].lower()] if len(parts) > 2 and parts[2] else []))
    return sets

if __name__ == "__main__":
    sets = convert(open(sys.argv[1]))
    with open(sys.argv[2], "w") as f:
        for s in sets: f.write(json.dumps(s) + "\n")
    print(f"{len(sets)} sets, {sum(len(s['rows']) for s in sets)} rows -> {sys.argv[2]}")

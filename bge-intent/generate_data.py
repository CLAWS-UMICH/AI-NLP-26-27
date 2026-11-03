"""Build a balanced 3,600-row intent dataset from existing normal examples and EVA radio phrasing."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "trainingData.json"
INTENTS_PATH = ROOT / "intents.json"
NORMAL_PATH = ROOT / "normal_examples.json"
SEED = 20261025

# Five intent-specific ways to name the requested telemetry/action. Combining
# these with the ten radio frames below produces 50 distinct astronaut-style
# examples per label.
TELEMETRY = {
    "get_oxy_pri_storage": ["primary oxygen bottle fill level", "primary O2 tank quantity", "main oxygen reserve percentage", "primary O2 supply remaining", "oxygen storage in tank one"],
    "get_oxy_sec_storage": ["secondary oxygen bottle fill level", "backup O2 tank quantity", "secondary oxygen reserve percentage", "secondary O2 supply remaining", "oxygen storage in tank two"],
    "get_oxy_pri_pressure": ["pressure in the primary oxygen bottle", "primary O2 tank pressure", "main oxygen cylinder pressure", "pressure on oxygen tank one", "primary O2 bottle pressure reading"],
    "get_oxy_sec_pressure": ["pressure in the secondary oxygen bottle", "backup O2 tank pressure", "secondary oxygen cylinder pressure", "pressure on oxygen tank two", "secondary O2 bottle pressure reading"],
    "get_suit_pressure_oxy": ["suit oxygen partial pressure", "O2 partial pressure in the suit", "suit ppO2 reading", "oxygen partial pressure on the suit loop", "suit-side oxygen pressure"],
    "get_suit_pressure_co2": ["suit carbon dioxide partial pressure", "CO2 partial pressure in the suit", "suit ppCO2 reading", "carbon dioxide pressure on the suit loop", "suit-side CO2 pressure"],
    "get_suit_pressure_other": ["suit trace-gas partial pressure", "partial pressure of other gases in the suit", "trace contaminant pressure in the suit", "suit other-gas pressure reading", "trace gas level on the suit loop"],
    "get_suit_pressure_total": ["total suit pressure", "overall pressure inside the suit", "suit pressure total reading", "combined pressure on the suit loop", "full suit pressure value"],
    "get_helmet_pressure_co2": ["carbon dioxide pressure in the helmet", "helmet CO2 partial pressure", "CO2 reading at the helmet", "helmet-side carbon dioxide pressure", "helmet ppCO2 value"],
    "get_fan_pri_rpm": ["primary circulation fan RPM", "speed of fan one", "main suit fan revolutions per minute", "primary fan tachometer reading", "RPM on the primary ventilation fan"],
    "get_fan_sec_rpm": ["secondary circulation fan RPM", "speed of fan two", "backup suit fan revolutions per minute", "secondary fan tachometer reading", "RPM on the secondary ventilation fan"],
    "get_scrubber_a_co2_storage": ["CO2 capacity remaining in scrubber A", "scrubber A carbon dioxide storage level", "fill status of the A-side CO2 scrubber", "scrubber A absorbent capacity reading", "carbon dioxide load on scrubber A"],
    "get_scrubber_b_co2_storage": ["CO2 capacity remaining in scrubber B", "scrubber B carbon dioxide storage level", "fill status of the B-side CO2 scrubber", "scrubber B absorbent capacity reading", "carbon dioxide load on scrubber B"],
    "get_temperature": ["temperature inside my suit", "suit internal temperature", "current EVA suit temperature", "temperature on my suit loop", "my suit thermal reading"],
    "get_coolant_storage": ["coolant remaining in the suit", "coolant reservoir level", "coolant supply percentage", "liquid coolant storage status", "coolant quantity on the suit"],
    "get_coolant_gas_pressure": ["gas-side coolant loop pressure", "coolant gas line pressure", "pressure in the coolant gas circuit", "gas coolant pressure reading", "coolant loop pressure on the gas side"],
    "get_coolant_liquid_pressure": ["liquid-side coolant loop pressure", "coolant liquid line pressure", "pressure in the liquid coolant circuit", "liquid coolant pressure reading", "coolant loop pressure on the liquid side"],
    "get_heart_rate": ["my heart rate", "current pulse in beats per minute", "crew heart-rate telemetry", "my cardiac rate reading", "pulse rate from the biomedical feed"],
    "get_oxy_consumption": ["my oxygen consumption rate", "current O2 use rate", "oxygen flow consumption per minute", "rate of oxygen draw from the suit", "my metabolic oxygen usage"],
    "get_co2_production": ["my carbon dioxide production rate", "current CO2 output rate", "carbon dioxide generation per minute", "my metabolic CO2 production", "rate of CO2 entering the loop"],
    "get_eva_elapsed_time": ["elapsed EVA time", "time since start of this spacewalk", "EVA mission timer reading", "duration of the current excursion", "time on the spacewalk clock"],
    "get_primary_battery_level": ["primary suit battery charge", "main battery percentage", "charge remaining on battery one", "primary power pack level", "main EVA battery status"],
    "get_secondary_battery_level": ["secondary suit battery charge", "backup battery percentage", "charge remaining on battery two", "secondary power pack level", "backup EVA battery status"],
    "get_warnings": ["active suit warnings", "current EVA alerts", "any caution or warning flags", "open suit fault notifications", "active alarms on my suit systems"],
}

RADIO_TELEMETRY_FRAMES = [
    "Houston, request the {item}, over.",
    "Flight, send the {item} to my HUD.",
    "CapCom, need a live {item} from the suit bus, copy?",
    "Telemetry check: {item}. Read back when ready, Houston.",
    "{call} to Flight: pull the {item} reading.",
    "Ground, call out the {item}; standing by.",
    "Can you uplink the {item} to my visor, Flight?",
    "Requesting the {item}; keep it short, over.",
    "Houston, confirm the {item} before I proceed.",
    "EVA control, status on the {item}, please acknowledge.",
]

OPERATIONS = {
    "open_menu_vitals": ["open the vitals panel", "bring up suit vitals", "display the biomedical page", "put the life-support readouts on my HUD", "switch my visor to vitals"],
    "open_menu_navigation": ["open the navigation map", "bring up the nav display", "show my route map on the visor", "switch the HUD to navigation", "display the EVA map panel"],
    "open_menu_tasks": ["open the mission task list", "bring up the EVA checklist", "display my task panel", "put the mission checklist on the HUD", "switch the visor to tasks"],
    "open_menu_uia": ["open the airlock interface", "bring up the UIA panel", "display the suit umbilical interface", "switch the HUD to the airlock controls", "show the UIA status page"],
    "close_menu": ["close the active HUD panel", "dismiss the open window", "clear the visor menu", "close the current display", "get the panel off my HUD"],
    "Get_coordinates": ["send coordinates for Shackleton Crater", "give me the position of the tool cache", "read out the coordinates for waypoint Alpha", "locate the antenna worksite for me", "send the grid position of the rover"],
    "Set_navigation_target": ["set my nav target to Shackleton Crater", "make the tool cache my destination", "navigate to waypoint Alpha", "set course for the antenna worksite", "target the rover position on my map"],
    "reroute_navigation": ["recalculate my route to Shackleton Crater", "plot a new path around the crater rim", "reroute me to waypoint Alpha", "find a new route to the antenna worksite", "update my course to the rover"],
    "Add_waypoint": ["add Shackleton Crater as a map waypoint", "mark the tool cache as a waypoint", "drop waypoint Alpha at my current position", "add the antenna worksite to the map", "save the rover location as a waypoint"],
    "Delete_waypoint": ["delete the Shackleton Crater waypoint", "remove the tool cache marker", "erase waypoint Alpha from the map", "delete the antenna worksite waypoint", "remove the rover location from my saved points"],
    "Complete_task": ["mark the tether inspection task complete", "check off the sample collection task", "close out the antenna repair checklist item", "mark the handrail inspection done", "complete the tool stowage task"],
}

RADIO_OPERATION_FRAMES = [
    "Houston, {action}, over.",
    "Flight, {action} on my visor.",
    "Request: {action}. Copy?",
    "CapCom, please {action}.",
    "{call} to Houston: {action}.",
    "Ground, {action} for this EVA.",
    "Can you {action} from your side, Flight?",
    "Need you to {action} now, Houston.",
    "Please {action}; standing by.",
    "Flight, go ahead and {action}; confirm when set.",
]

OFF_TOPIC = [
    "tell me a joke", "give me a quick weather report for Houston", "what time is it back on Earth",
    "recommend a song for the ride home", "how is the baseball game going", "tell me what is for dinner",
    "share a fun fact about the Moon", "remind me what movie we watched last night", "say hello to the crew",
    "tell me a short story about Mars",
]
RADIO_OFF_TOPIC_FRAMES = [
    "Houston, off-topic question while we have a minute: {topic}, over.",
    "Flight, no suit request here; just {topic}?",
    "CapCom, for morale, could you {topic}?",
    "{call} to Houston: can you {topic} when you get a chance?",
    "Ground, quick personal question: {topic}.",
]

CALLSIGNS = ["EVA One", "Suit Two", "Commander", "EV One", "Crew One"]


def astronaut_examples(intent: str) -> list[str]:
    if intent in TELEMETRY:
        items, frames = TELEMETRY[intent], RADIO_TELEMETRY_FRAMES
        return [frame.format(item=item, call=CALLSIGNS[j % len(CALLSIGNS)])
                for i, item in enumerate(items) for j, frame in enumerate(frames)]
    if intent in OPERATIONS:
        actions, frames = OPERATIONS[intent], RADIO_OPERATION_FRAMES
        return [frame.format(action=action, call=CALLSIGNS[j % len(CALLSIGNS)])
                for i, action in enumerate(actions) for j, frame in enumerate(frames)]
    if intent == "unhandled":
        return [frame.format(topic=topic, call=CALLSIGNS[j % len(CALLSIGNS)])
                for topic in OFF_TOPIC for j, frame in enumerate(RADIO_OFF_TOPIC_FRAMES)]
    raise ValueError(f"No astronaut phrasing defined for {intent}")


def main() -> None:
    old_rows = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    intents = [item["intent"] for item in json.loads(INTENTS_PATH.read_text(encoding="utf-8"))]
    by_intent: dict[str, list[dict]] = defaultdict(list)
    for row in old_rows:
        by_intent[row["intent"]].append(row)

    rng = random.Random(SEED)
    normal_rows = []
    astronaut_rows = []
    for intent in intents:
        candidates = list({row["text"]: row for row in by_intent[intent]}.values())
        if len(candidates) < 50:
            raise ValueError(f"{intent} has only {len(candidates)} unique existing phrases")
        selected = rng.sample(candidates, 50)
        for row in selected:
            normal_rows.append({**row, "style": "normal"})

        talk = astronaut_examples(intent)
        talk = list(dict.fromkeys(talk))
        if len(talk) != 50:
            raise ValueError(f"{intent} produced {len(talk)} astronaut examples, expected 50")
        for index, text in enumerate(talk):
            astronaut_rows.append({
                "text": text,
                "intent": intent,
                "source": "generated_astronaut_radio",
                "style": "astronaut_talk",
                # Generated variants are training-only so evaluation stays on
                # phrase families from the original data rather than siblings.
                "family": f"training_only_astronaut_{intent}_{index:02d}",
            })

    combined = normal_rows + astronaut_rows
    texts = [row["text"] for row in combined]
    if len(combined) != 3600 or len(set(texts)) != 3600:
        raise ValueError(f"expected 3600 unique rows; got {len(combined)} rows and {len(set(texts))} unique texts")
    counts = Counter(row["intent"] for row in combined)
    styles = Counter((row["intent"], row["style"]) for row in combined)
    if set(counts.values()) != {100} or any(styles[(label, style)] != 50 for label in intents for style in ("normal", "astronaut_talk")):
        raise ValueError("output is not balanced at 50 examples per style per intent")

    rng.shuffle(combined)
    NORMAL_PATH.write_text(json.dumps(normal_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DATA_PATH.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(combined)} unique rows: 36 intents × 100 (50 normal + 50 astronaut talk).")


if __name__ == "__main__":
    main()

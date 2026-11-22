# Labelling Guide

Draft, 2026-11-22. For everyone who writes or checks data rows. The Ollama label checker gets the same rules.

One sentence in, one label out. If two careful people could pick different labels for a row, the row is the problem: rewrite it or add a rule here.

---

## 1. Decide the label in this order

1. **Is the astronaut asking the HUD to do something?**
   If not, it's `unhandled`. That covers read-backs, radio chatter, switch callouts, self-talk, small talk, cancellations, negated commands, and Whisper's made-up "thank you".
2. **Is it one of our 35 commands?**
   Requests we don't support are `unhandled`, for example camera, spectrometer, comms, lights, timers and remaining time.
3. **A panel or a single value?**
   A panel ("show my vitals") is `open_menu_*`. A single value ("what's my heart rate") is `get_*`.
4. **Which one?**
   Use the qualifier words in §2. If there is no qualifier, use the default in §3.
5. **Does it name a place, waypoint or task?**
   The verb picks the label (§4.7, §4.8). Mark the name as the slot (§5).
6. **Does it have Whisper errors?**
   Label what the speaker clearly meant ("at a waypoint called crater rim" is `Add_waypoint`). If the deciding word is lost ("what's my... level"), the row is `unhandled`.

---

## 2. Qualifier words

These words decide the "which one" pairs. Only use them inside their own block.

| Side | Words that pick it |
|---|---|
| Primary | primary, pri, main, one, number one, first |
| Secondary | secondary, sec, second, two, number two, backup |
| Scrubber A | A, alpha, one |
| Scrubber B | B, bravo, two |
| Coolant gas | gas, gas loop, gas side, vapor |
| Coolant liquid | liquid, liquid loop, liquid side, water loop |
| Helmet (CO2) | helmet |
| Suit (CO2) | suit, or no qualifier |

---

## 3. Unqualified requests (decided)

No qualifier means primary, or total for pressure. This matches the nominal DCU setup in NASA's procedures ("OXY – PRI", "FAN – PRI", "CO2 – A").

| Request without a qualifier | Label |
|---|---|
| "oxygen", "O2 level", "how much air" | `get_oxy_pri_storage` |
| "oxygen pressure", "tank pressure" | `get_oxy_pri_pressure` |
| "battery", "power level" | `get_primary_battery_level` |
| "fan", "fan speed" | `get_fan_pri_rpm` |
| "pressure", "suit pressure" | `get_suit_pressure_total` |
| "scrubber" | `get_scrubber_a_co2_storage` |
| "CO2", "CO2 level" | `get_suit_pressure_co2` |
| "coolant" | `get_coolant_storage` |

---

## 4. Every label

Examples show the idea. **Never copy them into the data.** They are banned from every split, and the leakage check looks for them.

### 4.1 Oxygen

| Label | Means | Examples | Not this |
|---|---|---|---|
| `get_oxy_pri_storage` | % left in the primary tank | "how much O2 have I got left" / "primary oxygen percent" | psi → `get_oxy_pri_pressure`; "suit O2 pressure" → `get_suit_pressure_oxy` |
| `get_oxy_sec_storage` | % left in the secondary tank | "what's the backup tank at" / "sec O2 level" | no qualifier → primary |
| `get_oxy_pri_pressure` | primary tank pressure (psi) | "primary tank pressure" / "O2 psi on tank one" | % left → storage |
| `get_oxy_sec_pressure` | secondary tank pressure (psi) | "secondary oxygen tank psi" / "pressure in tank two" | |
| `get_oxy_consumption` | how fast oxygen is being used | "what's my O2 burn rate" / "how quickly am I going through oxygen" | how much is left → storage |

### 4.2 Suit atmosphere and CO2

| Label | Means | Examples | Not this |
|---|---|---|---|
| `get_suit_pressure_oxy` | oxygen partial pressure in the suit | "suit ppO2" / "partial pressure of oxygen in the suit" | tank pressure → `get_oxy_*_pressure` |
| `get_suit_pressure_co2` | CO2 partial pressure in the suit | "CO2 level" / "how much carbon dioxide is in my suit" | "helmet CO2" → helmet; how much I'm producing → `get_co2_production` |
| `get_suit_pressure_other` | trace gas pressure in the suit | "trace gas pressure" / "other gases reading" | |
| `get_suit_pressure_total` | total suit pressure | "what's my suit pressure" / "pressure check" | "oxygen pressure" → `get_oxy_pri_pressure` |
| `get_helmet_pressure_co2` | CO2 pressure in the helmet | "helmet CO2" / "carbon dioxide in my helmet" | no qualifier → suit |
| `get_co2_production` | how fast the astronaut makes CO2 | "CO2 production rate" / "how much CO2 am I putting out" | level or pressure → `get_suit_pressure_co2` |
| `get_scrubber_a_co2_storage` | how full scrubber A is | "scrubber alpha capacity" / "how saturated is the scrubber" | B, bravo → scrubber B |
| `get_scrubber_b_co2_storage` | how full scrubber B is | "scrubber bravo level" / "check the B scrubber" | |

### 4.3 Thermal and coolant

| Label | Means | Examples | Not this |
|---|---|---|---|
| `get_temperature` | suit internal temperature | "how hot is it in my suit" / "suit temp" | outside or surface temperature → `unhandled` |
| `get_coolant_storage` | coolant left | "coolant level" / "how much coolant do I have" | coolant pressure → gas or liquid |
| `get_coolant_gas_pressure` | coolant gas loop pressure | "coolant gas pressure" / "pressure on the gas loop" | |
| `get_coolant_liquid_pressure` | coolant liquid loop pressure | "liquid coolant pressure" / "water loop pressure" | |

### 4.4 Fans and power

| Label | Means | Examples | Not this |
|---|---|---|---|
| `get_fan_pri_rpm` | primary fan speed | "fan speed" / "how fast is fan one spinning" | |
| `get_fan_sec_rpm` | secondary fan speed | "secondary fan RPM" / "backup fan speed" | |
| `get_primary_battery_level` | primary battery level | "battery" / "how much power do I have" | "BATT UMB" (switch callout) → `unhandled` |
| `get_secondary_battery_level` | secondary battery level | "secondary battery percent" / "backup battery level" | |

### 4.5 Body, time and warnings

| Label | Means | Examples | Not this |
|---|---|---|---|
| `get_heart_rate` | heart rate | "what's my pulse" / "BPM" | "vitals" → `open_menu_vitals` |
| `get_eva_elapsed_time` | time since the EVA started | "how long have I been out here" / "EVA clock" | "what time is it", "time remaining" → `unhandled` |
| `get_warnings` | active warnings and alerts | "any warnings" / "what's that alarm" / "is anything off nominal" | the vitals panel → `open_menu_vitals` |

### 4.6 Panels

Open words: open, show, pull up, bring up, display, switch to, go to.

| Label | Means | Examples | Not this |
|---|---|---|---|
| `open_menu_vitals` | vitals panel | "show my vitals" / "pull up suit status" | one value → its `get_*` |
| `open_menu_navigation` | map / navigation panel | "open the map" / "bring up nav" | "take me to X" → `Set_navigation_target` |
| `open_menu_tasks` | tasks / checklist panel | "show the checklist" / "open procedures" | "mark X done" → `Complete_task` |
| `open_menu_uia` | UIA (airlock) panel | "open the UIA panel" / "bring up the airlock interface" | "open the O2 vent" (a switch) → `unhandled` |
| `close_menu` | close the open panel | "close this" / "hide the panel" / "dismiss" | "close the vent / valve / hatch" → `unhandled` |

### 4.7 Navigation and waypoints (slot labels except reroute)

| Label | Means | Examples | Not this |
|---|---|---|---|
| `Get_coordinates` | where a named target is | "where's POI 2" / "coordinates for the lander" | "take me there" → `Set_navigation_target` |
| `Set_navigation_target` | navigate to a named place | "take me to crater rim" / "guide me back to the lander" | no place named → `reroute_navigation` |
| `reroute_navigation` | new route to the current target | "reroute" / "this path is blocked, find another way" | a new place named → `Set_navigation_target` |
| `Add_waypoint` | add a named waypoint / drop a pin | "drop a pin, call it boulder field" / "add waypoint sample site four" | Whisper's "at a waypoint called X" is still `Add_waypoint` |
| `Delete_waypoint` | delete a named waypoint | "delete the boulder field pin" / "remove waypoint alpha" | |

### 4.8 Tasks (slot label)

| Label | Means | Examples | Not this |
|---|---|---|---|
| `Complete_task` | mark a named task complete | "mark PLCD deployment complete" / "seismometer's done, check it off" | "is the seismometer done?" → `unhandled`; "open tasks" → `open_menu_tasks` |

### 4.9 `unhandled`

| Kind | Examples |
|---|---|
| Keyword trap | "this rock reads high in oxygen" / "close the O2 vent" |
| Read-back of a value | "copy, primary O2 forty-seven" / "secondary battery at ninety, copy" |
| Switch callout from procedures | "OXY PRI" / "depress pump power on" |
| Cancel or negation (decided) | "never mind" / "don't open the map" / "no, not the map" |
| Not supported | "take a picture" / "start the spectrometer scan" / "how much time do I have left" |
| Radio chatter, small talk, self-talk | "EV2, how's it going over there" / "okay, this one's heavy" |
| Fragment or false start | "open the" / "what's my" |
| Whisper noise text | "Thank you." / "you" |

---

## 5. Marking the slot

- `slot_text` is copied exactly from the sentence: same spelling, same case.
- Mark the whole name ("crater rim", "POI 2", "PLCD deployment"). Leave out "the", "waypoint", "pin" and "task" unless they're part of the name.
- In Whisper-style rows, mark the misspelt name as written ("Greater Rim").
- Use names from the list you were given. Train names and held-out names are separate lists, and held-out rows go to val and test only.
- `Add_waypoint` keeps the name verbatim, with no HUD confirmation (decided).
- One name per row.

---

## 6. Writing rules (short version)

- Every row is a genuinely different sentence. Don't make a row by swapping one word in another row.
- Cover the variety axes in PLAN.md §3.4: questions, commands, fragments, long rows, filler, jargon, indirect requests, Whisper-style spelling, keyword not first.
- Whisper-style rows use real errors from the confusion phrasebook, built from the Whisper round-trips. Until it exists, use the ones already seen: "add" → "at", "O2" → "o two", "CO2" → "see oh two", "crater" → "greater".
- If you're unsure of a label, don't write the row. Ask, then add the answer here.

---

## 7. Proposed rules: need team OK

Until each one is confirmed, **don't write rows of that kind.**

| # | Case | Proposal |
|---|---|---|
| P1 | "how far is X" | `Get_coordinates` (the HUD shows the target's location and distance) |
| P2 | Pin with no name: "drop a pin here" | `Add_waypoint`, empty slot; the HUD names it automatically |
| P3 | Task with no name: "done", "task complete" | `Complete_task`, empty slot; the HUD confirms the current task |
| P4 | "my temperature", "body temp" | `get_temperature` |
| P5 | "coolant pressure" with no gas / liquid | no default yet; pick gas or liquid |
| P6 | DCU battery names "local" / "UMB" | map to primary / secondary, or `unhandled`? |
| P7 | Two commands in one sentence | `unhandled`, so the astronaut repeats one at a time |
| P8 | "what's next", "next step" | `open_menu_tasks` |
| P9 | "drop the X waypoint" (could mean delete) | `Delete_waypoint`, or avoid the phrase |

---

## 8. Checking labels

- **Blind double label:** two people label the same 100 pilot rows without seeing each other's labels. Target agreement is 95% or more.
- **Disagreements:** each one becomes a rule here, or a rewritten row.
- **Ollama checker:** gets §1–§4 as its prompt. Its disagreements go to a person, never straight into the data.
- **Changes:** log every rule change below, so older rows can be re-checked.

## Changelog

- 2026-11-22: first draft. Decided: unqualified requests → primary / total; `Add_waypoint` slot verbatim; cancel and negation → `unhandled`.

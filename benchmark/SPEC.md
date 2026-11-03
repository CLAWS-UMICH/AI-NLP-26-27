# Shared EVA intent benchmark

## Data contract
Use the exact case-sensitive labels in `onboarding/intents.json`. Each row is one JSON object per UTF-8 line: `{"text":"...","intents":["<label>"],"source":"llm"}`. Exactly one label is allowed. The five name slots (`Get_coordinates`, `Set_navigation_target`, `Add_waypoint`, `Delete_waypoint`, `Complete_task`) also require `slot` with a verbatim name from `names.json`, present verbatim in `text`.

The target is 3,600 rows: 100 per each of 36 labels. Utterances should sound like mid-EVA speech or plausible speech recognition output. Do not infer an unspecified primary/secondary, A/B scrubber, gas/liquid, or suit/helmet distinction. Keep the exact requested measurement or action clear.

Within each intent vary terse fragments, direct commands, questions, indirect questions, polite/long requests, urgent speech, fillers, ASR-like lowercase/no punctuation, homophones and dropped words, spoken numbers/abbreviations, and lengths from 1–3 to 13+ words. No repeated or templated sentences; openings, style, and length should not identify labels. `unhandled` includes non-request status comments that mention suit terms, small talk, radio acknowledgments, off-topic requests and noise/fragments.

Use the shared name pool. Every name should appear in at least two different name intents and no more than three times within any one intent. Mix multiword, numbered and lowercase names.

## Confusion pairs and contrast examples
The requested `claws_intent_onboarding.tex` was not found in the workspace. The following high-risk pairs are inferred from neighboring labels in `onboarding/intents.json`; examples are guidance, not benchmark rows.

### Primary/secondary oxygen storage
Good: “primary oxygen tank level”; “secondary O2 reserve percentage”; “how much oxygen is left in the backup tank?”
Bad: “oxygen level?” (tank unspecified); “primary tank pressure” (pressure, not storage); “secondary oxygen burn rate” (consumption).

### Primary/secondary oxygen pressure
Good: “primary O2 tank pressure”; “read the backup oxygen pressure”; “what pressure is left in tank two?”
Bad: “primary oxygen remaining” (storage); “oxygen pressure?” (tank unspecified); “secondary tank percentage” (measurement unclear).

### Suit vs helmet CO2 pressure
Good: “suit CO2 partial pressure”; “helmet carbon dioxide pressure”; “read the pressure inside my helmet for CO2.”
Bad: “CO2 pressure” (location unspecified); “scrubber B CO2 storage” (storage); “total suit pressure” (total pressure).

### Suit gas, total, and oxygen pressure
Good: “suit oxygen partial pressure”; “suit trace-gas pressure”; “total pressure in the suit.”
Bad: “suit pressure” (component unspecified); “oxygen tank pressure” (tank); “helmet CO2 pressure” (helmet reading).

### Scrubber A/B CO2 storage
Good: “scrubber A CO2 canister level”; “how full is scrubber B?”; “check the B-side carbon dioxide storage.”
Bad: “scrubber storage” (A/B unspecified); “scrubber A CO2 pressure” (pressure vs storage); “helmet CO2 pressure” (different subsystem).

### Primary/secondary fans and batteries
Good: “primary circulation fan RPM”; “secondary fan speed in revolutions per minute”; “secondary battery charge percentage.”
Bad: “fan speed” (primary/secondary unspecified); “primary battery voltage” (not battery level); “secondary fan status” (not RPM).

### Coolant storage vs gas/liquid pressure
Good: “coolant reservoir level”; “coolant gas-loop pressure”; “liquid coolant loop pressure.”
Bad: “coolant pressure” (phase unspecified); “gas coolant remaining” (storage); “oxygen gas pressure” (different subsystem).

### CO2 production vs oxygen consumption
Good: “my CO2 production rate”; “oxygen use per minute”; “am I consuming oxygen quickly?”
Bad: “CO2 level” (storage/pressure); “oxygen remaining” (storage); “gas rate” (substance unspecified).

### EVA elapsed time vs heart rate
Good: “time since EVA start”; “what's my heart rate”; “pulse in beats per minute.”
Bad: “time?” (ambiguous); “heart status” (not a rate); “mission duration remaining” (not elapsed time).

### Open panels and close menu
Good: “open the vitals panel”; “bring up the UIA interface”; “dismiss this HUD window.”
Bad: “show me the map and vitals” (multiple panels); “close the navigation panel and open tasks” (two actions); “where am I on the map?” (coordinate lookup).

### Navigation actions
Good: “coordinates for Shackleton Crater”; “set Artemis Ridge as my destination”; “reroute around the ridge”; “add Alpha 3 as a waypoint”; “delete Alpha 3 waypoint”; “mark soil sample complete.”
Bad: “go to a crater” (no named destination); “add a waypoint and reroute” (multiple actions); “is Alpha 3 nearby?” (not an action); “finish” (task unspecified).

### Warnings vs unhandled status chatter
Good: “list active suit warnings”; “any current EVA alerts?”; “oxygen looks fine, moving on” is unhandled.
Bad: “oxygen looks fine” labeled as a request for oxygen level; “take a photo” labeled as an EVA panel request; “copy that” labeled as a warning query.

## Acceptance gates
`audit.py` checks counts and schema; duplicates after normalization; character trigram Jaccard duplicates/contradictions; opening diversity; length buckets and mean; per-label style rates; style-only five-fold CV accuracy; TF-IDF/logistic five-fold CV score and top confusions; slot/name constraints; split counts and train-to-held-out similarity. A final split uses a stratified 80/10/10 allocation per label with seed 42. Any held-out row too similar to same-label training text must be rewritten before release.

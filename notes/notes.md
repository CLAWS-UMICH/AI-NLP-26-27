# RAIANA: onboarding notes (MPNet redo)

Same tasks as `../onboarding/submissions/raiana/notes.md` (MiniLM), redone with **MPNet** so the two encoders can be compared on the same data.
Run everything from this folder (`NLP/`):
```bash
python ../onboarding/code/validate.py data/train.jsonl   # 1258 rows, 0 problems
python code/baseline.py      # -> baseline.txt
python code/train.py         # -> runs/model.pt, label2id.json, metrics.json, history.json (~1 min on M5 MPS)
python code/plot_curves.py   # -> runs/learning_curves.png
python code/predict.py "add a waypoint named Olympus Ridge"
python code/voice.py         # optional, mic or .wav files
```

## Results
- Data: synthetic (LLM-generated) + handwritten. Same `train.jsonl` / `test.jsonl` as the MiniLM run, plus 10 handwritten `unhandled` near-misses (weather / ambient heat) added after the first round of failures
- Model (see MODELS.md): MPNet (`sentence-transformers/all-mpnet-base-v2`, 110M params, 768-dim)
- Baseline accuracy (baseline.txt): 89.8% / macro-F1 91.0%
- Fine-tuned accuracy / macro-F1 (runs/metrics.json): 96.8% / 97.1% (val 100% / 100%)

### MPNet vs MiniLM (same test.jsonl, 186 handwritten sentences)
| | MiniLM baseline | MiniLM fine-tuned | MPNet baseline | MPNet fine-tuned |
|---|---|---|---|---|
| Test accuracy | 93.0% | 96.2% | 89.8% | **96.8%** |
| Test macro-F1 | 93.6% | 96.5% | 91.0% | **97.1%** |

- Frozen MPNet is *worse* than frozen MiniLM here (89.8 vs 93.0), but it gains much more from fine-tuning (+7.0 vs +3.2 points), so it ends up slightly ahead
- Confidence is a lot higher on correct answers: most unseen sentences below are 0.93-0.97 vs 0.83-0.95 with MiniLM

### Hyperparameters (changed from the MiniLM run)
| | MiniLM | MPNet | Why |
|---|---|---|---|
| epochs | 4 | 8 | val loss still dropping at 6, flat by 8 (see `runs/learning_curves.png`) |
| batch | 8 | 16 | |
| encoder lr | 1e-4 | 3e-5 | 5x bigger model, smaller steps so it doesn't forget its pretraining |
| head lr | (same as encoder) | 1e-3 | see "What failed" #1 |
| warmup | 0.4 | 0.1 | |

## Unseen sentences (10+)
| Sentence | Predicted | Confidence | Correct? |
|---|---|---|---|
| how much oxygen do I have | get_oxy_pri_storage | 0.965 | Yes |
| navigate to Ithaca | Set_navigation_target (slot: Ithaca) | 0.934 | Yes |
| drop a pin called rock cluster | Add_waypoint (slot: rock cluster) | 0.949 | Yes |
| get rid of the crater waypoint | Delete_waypoint (slot: crater) | 0.957 | Yes |
| where is the lander | Get_coordinates (slot: lander) | 0.947 | Yes |
| I finished the sample collection | Complete_task (slot: sample collection) | 0.865 | Yes |
| bring out the map | open_menu_navigation | 0.970 | Yes |
| find a new path around the crater | reroute_navigation | 0.950 | Yes |
| how's my heart | get_heart_rate | 0.955 | Yes (MiniLM said unhandled 0.41) |
| am I running low on oxygen | get_oxy_pri_storage | 0.961 | Yes (MiniLM said get_oxy_consumption) |
| scrubber ACO2 storage | get_scrubber_a_co2_storage | 0.941 | Yes (the Whisper merge that dropped MiniLM to 0.67) |
| what's the second fan spinning at | get_fan_sec_rpm | 0.954 | Yes |
| what's the backup battery at | get_secondary_battery_level | 0.966 | Yes |
| tank two pressure | get_oxy_sec_pressure | 0.960 | Yes |
| mark the egress task complete | Complete_task (slot: egress) | 0.952 | Yes |
| erase the pebble field waypoint | Delete_waypoint (slot: pebble field) | 0.957 | Yes |
| set a course for the north ridge | Set_navigation_target (slot: north ridge) | 0.916 | Yes (after fix #3) |
| plot a path to the drill site | Set_navigation_target (slot: drill site) | 0.819 | Yes (after fix #3) |
| what's the weather on mars | unhandled | 0.951 | Yes (after fix #2, was get_temperature 0.69) |
| I'm feeling dizzy | unhandled | 0.804 | Yes |
| pressure | unhandled | 0.326 | Yes (ambiguous, so good it refused) |
| is it chilly outside | get_temperature | 0.811 | **No**, outside temp isn't suit temp |

## What failed and why
1. **First fine-tune predicted `unhandled` for everything (5% accuracy)**
   - Used one learning rate (3e-5) for the whole model. The classification head starts random, and at 3e-5 it barely learned in 6 epochs (loss stuck at 1.8), so no prediction ever got above `MIN_CONF` = 0.5 and they all became unhandled
   - MiniLM got away with this because its LR was 1e-4
   - Fix: two param groups in AdamW, 3e-5 for the encoder and 1e-3 for the head. Loss went down to 0.06 and test went 5% -> 96.8%
2. **Anything about heat/weather -> `get_temperature`**
   - "what's the weather on mars" gave get_temperature (0.69). The data only had "temperature" meaning suit temp, so the model learned "hot/cold/weather" = suit temperature
   - Fix: added 10 handwritten unhandled near-misses ("is it going to rain", "how hot is the surface of the moon", "what's the temperature back in houston", ...), retrained. Weather on mars is now unhandled at 0.95
   - Still fails on "is it chilly outside" (0.81). Need more examples about outside / ambient temperature
3. **Slot extraction kept command words**
   - "set a course for the north ridge" -> slot "course for the north ridge", "plot a path to the drill site" -> the whole sentence
   - Fix: added "course", "path", "plot" to `LEAD` in predict.py
4. **Code bugs carried over from the MiniLM version (fixed here)**
   - Imports were `from onboarding.submissions.raiana.code.model import ...`, which breaks when running `python code/train.py`. Changed to `from model import ...`
   - predict.py imported model.py *before* setting `INTENT_MODEL` from metrics.json, so the env var did nothing (MODEL_NAME is read at import time). Moved the import below it
5. Note: "can you make it less hot" looked like an unseen failure, but it's actually in train.jsonl and landed in the 15% val split. Not unseen, so it's not in the table

## Stretch (optional)
Voice pipeline working?
- Yes, same `voice.py` as the MiniLM run (faster-whisper base.en + domain prompt + VAD, then MPNet, then spoken reply)
- Tested in .wav file mode with clips made by macOS `say` (synthetic voice, **not** my own mic, so this is a pipeline check, not a real-speech test):
  | Said | Whisper heard | Intent | Conf |
  |---|---|---|---|
  | "what is the scrubber A CO2 storage" | "What is the scrubber ACO2 storage?" | get_scrubber_a_co2_storage | 0.90 (MiniLM: 0.67) |
  | "add a waypoint named Olympus Ridge" | "at a waypoint named Olympus Ridge." | **Get_coordinates** (slot: Olympus Ridge) | 0.60 (MiniLM got Add_waypoint 0.81) |
  | "open the UIA panel" | "Open the UIA panel" | open_menu_uia | 0.97 |
  | "how's my heart" | "house my heart" | unhandled | 0.91 (Whisper error, refusing is the right call) |
- Notes: Whisper hears "add" as "at" and then MPNet goes for Get_coordinates ("at" sounds like a location). Should add some Whisper-style mishearings ("at a waypoint", "had a waypoint") to the Add_waypoint data

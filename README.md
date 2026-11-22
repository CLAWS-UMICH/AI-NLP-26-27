# AI-NLP-26-27

Intent recognition for the CLAWS (University of Michigan) voice assistant in the **2026–27 NASA SUITS Challenge**.

When the astronaut speaks, Whisper turns the speech into text, and this model works out what they want. It returns **one command per sentence** for the spacesuit display to carry out, such as a telemetry query, menu action, navigation or task update.

```
"take me to crater rim"
        │
        ▼
Set_navigation_target  (slot: "crater rim")
```

## Background

NASA SUITS (Spacesuit User Interface Technologies for Students) is a NASA design challenge in which student teams build spacesuit displays and voice assistants. In 2026–27 the scenario is a lunar EVA in low light. One astronaut leaves the airlock using UIA (Umbilical Interface Assembly) procedures, navigates to a series of Points of Interest, does geology and science-payload tasks at each one, and follows breadcrumbs back to the lander. NASA tests every team's system at Johnson Space Center in **May 2027**.

The voice assistant is a hard requirement: *"The UI shall provide voice assistance using natural language"* (Mission Description, 4a.3.b). NASA also asks for two things that shape how we build the model:

- **Short replies.** Ask for primary oxygen and the assistant answers "Primary O2 47%", without filler.
- **Guardrails against hallucination** on anything mission-critical. The model chooses from a fixed list of intents and never generates telemetry values. The display reads real values from NASA's Telemetry Stream Server (TSS).

Upcoming dates: orientation **Dec 10, 2026** · Software Design Review **Apr 1, 2027** · test week **May 2027**.

## Model output

```json
{
  "text": "take me to crater rim",
  "selected_intent": "Set_navigation_target",
  "confidence": 0.98,
  "top5": [{"intent": "Set_navigation_target", "confidence": 0.98}, ...],
  "slot": "crater rim"
}
```

- `slot` is only present for the five name intents.
- Every intent comes from [intents.json](intents.json); nothing outside that list is ever returned.
- Low confidence or small talk ("copy that", "nice view") returns `unhandled`. The model refuses rather than guesses.
- Destructive actions (`Delete_waypoint`, `Complete_task`) should be confirmed by the display before they run.

## Intents

There are 36 labels: 35 commands plus `unhandled`. They are listed in [intents.json](intents.json) and are **case-sensitive**.

- **Suit telemetry (24):** oxygen storage and pressure, suit and helmet pressures, fans, CO₂ scrubbers, coolant, temperature, heart rate, O₂ consumption, CO₂ production, batteries, elapsed time, warnings
- **Display menus (5):** `open_menu_vitals`, `open_menu_navigation`, `open_menu_tasks`, `open_menu_uia`, `close_menu`
- **Navigation and tasks (6):** `Get_coordinates`, `Set_navigation_target`, `reroute_navigation`, `Add_waypoint`, `Delete_waypoint`, `Complete_task`. All except `reroute_navigation` take a name slot.

This set comes from last year's onboarding. This year's mission adds things it doesn't cover yet: stepping through procedures, dropping pins, breadcrumbs back to the lander, predicted maximum range, photos and spectrometer readings. New labels will be agreed with the display (HUD) team, who depend on these names.

## What has to work

| Case | Example |
|---|---|
| Filler around a command | "copy that, uh, what's my heart rate" |
| Jargon | "pri o2?" |
| Whisper errors | "at a waypoint named…" (Whisper heard "add" as "at") |
| Small talk | "nice view" → `unhandled` |

Decisions are tracked in [PLAN.md](PLAN.md).

## Data

JSONL, one example per line, exactly one `intent` per row. `slot` is only for the five name intents:

```json
{"text": "what's my primary oxygen at", "intent": "get_oxy_pri_storage"}
{"text": "take me to poi b", "intent": "Set_navigation_target", "slot": "poi b"}
```

- Write it the way people talk: lowercase, filler words, clipped phrases ("fan rpm?").
- Always say which one: primary or secondary, scrubber A or B, coolant gas or liquid.
- Tag LLM-generated rows with `"source": "llm"`, and read every one.
- The test set is handwritten and **never** goes into training data.

## Evaluation

- **Accuracy:** the share of sentences labeled with the right intent. This is the headline number.
- **Per-intent F1:** shows which intents are weak.
- **Slot exact match:** a wrong name sends the astronaut to the wrong place.
- **Small talk triggering actions:** how often non-commands set something off. Keep this near zero.
- **Latency** on the device we'll use at test week.

Test on handwritten sentences and on real Whisper transcripts from team members, not only on held-out training data.

## Current results

Fine-tuned [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) on 3,600 training rows, tested on 360 handwritten sentences:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Frozen encoder + logistic regression | 92.5% | 92.4% |
| **Fine-tuned MPNet** | **98.6%** | **98.6%** |

Known misses: off-topic sentences that share a keyword ("battery in a tesla"), and Whisper turning "add" into "at".

## Getting started

Python 3.10–3.12.

```bash
git clone https://github.com/CLAWS-UMICH/AI-NLP-26-27.git
cd AI-NLP-26-27
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers sentence-transformers scikit-learn matplotlib

python code/baseline.py      # frozen-encoder baseline
python code/train.py         # fine-tune, writes to runs/
python code/predict.py "add a waypoint named Olympus Ridge"
python code/voice.py         # optional: mic → Whisper → model → spoken reply
                             # needs: pip install faster-whisper sounddevice pyttsx3
```

Model weights (`*.pt`) are git-ignored.

## Layout

```
code/      baseline, model, train, predict, voice scripts
data/      train.jsonl, test.jsonl (handwritten), astronaut language data
runs/      metrics, label map, training curves
notes/     experiment log
intents.json · MODELS.md · PLAN.md
```

## References

- NASA SUITS 2026–27: [Mission Description](https://www.nasa.gov/wp-content/uploads/2026/08/fy27-nasa-suits-mission-description.pdf) · [Proposal Guidelines](https://www.nasa.gov/wp-content/uploads/2026/08/fy27-suits-proposal-guidelines.pdf) · [Challenge page](https://www.nasa.gov/stem-content/nasa-spacesuit-user-interface-technologies-for-students-suits-design-challenge/)
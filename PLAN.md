# Plan

## Priming: teaching MiniLM astronaut language

Before MiniLM sees any intent labels, we teach it what space jargon means. It already knows English, but it doesn't know that "UIA" is a switch panel or that "PLSS check" is a status question.

**What feeds it**

- `data/suits_glossary.json`: 106 SUITS terms, hand-written from the 2026–27 Mission Description, `intents.json` and Apollo 15 radio. Each term has a plain meaning, spoken/Whisper spellings, a definition, and a jargon-vs-plain example sentence.
- `data/nasa_acronyms_eva.json`: 401 acronyms from nasa/NASA-Acronyms, filtered to EVA, suit and life-support terms only (out of ~49k).
- Apollo transcripts are not training data. They're a style guide for writing the seed and test rows (`data/style_reference.md`).

**How the glossary is used (three ways)**

1. **Normalization.** Before classifying, unambiguous jargon and Whisper spellings are rewritten to one plain form ("u i a" → "umbilical interface assembly", "scrubber eh" → "scrubber a"). This happens the same way in training and prediction.
2. **Jargon training rows.** `generate.py` swaps terms for their other forms ("primary oxygen" → "pri o2", "heart rate" → "hr") so the model sees astronaut and Whisper spellings.
3. **Primer pairs.** `build_primer.py` turns the glossary and acronyms into 550 pairs of jargon and its meaning for `prime.py`.

**Design decisions**

- **Contrastive priming, not a classifier.** The model learns to put "BATT to UMB" close to "switch the battery to umbilical power". This changes how it understands words without touching intent labels.
- **Gentle settings** (learning rate 2e-5, 3 passes) so priming adds vocabulary without wrecking what MiniLM already knows. 3 passes because 550 pairs is only ~9 steps per pass.
- **Every pair is unique.** Each batch uses the other pairs as wrong answers, so a repeated meaning would be counted as both right and wrong.
- **10% of pairs are held out** to check that priming helped. Smoke run: 29% → 38% of held-out jargon matched to the right meaning.
- **Only normalize what's safe.** Words that are also everyday speech ("sec", "met", "hr", "copy") are never rewritten. "Give me a sec" must stay small talk.
- **Normalization and priming do different jobs.** Normalization handles the terms we know for certain; priming covers variations we didn't list.
- **Each run saves its own copy of the glossary.** A trained model keeps normalizing the way it was trained, even if the glossary changes later.
- **Radio words stay out of jargon swaps.** Swapping "no" for "negative" would change meaning; radio phrasing comes in through filler words instead.
- **Training uses the newest primed model by default.** Compare against `train.py --encoder base` to confirm priming actually helps accuracy, not just matching.

**Commands**

```
python code/fetch_acronyms.py   # data/nasa_acronyms_eva.json
python code/build_primer.py     # data/astronaut_primer.json
python code/prime.py            # runs/prime-<time>/encoder
```

## Generating data

The model is single-intent: every sentence has exactly one label. Rows look like `{"text": "take me to crater rim", "intent": "Set_navigation_target", "slot": "crater rim"}`, and `slot` is only used by the five name intents.

We have no annotators or users outside the team, so we write a small seed by hand and let open-source generators do the volume: plain-text templates, back-translation for paraphrases, Piper + faster-whisper for speech noise, and a capped LLM fill from a local model. The plan is built around one risk: generated data scoring well on generated data. To prevent that, the test set is frozen before anything is generated, every split is made by phrasing group, never by individual row, and every generated row inherits the group of the seed row it came from.

### Targets

| Set | Rows | Made from | Used for |
|---|---|---|---|
| `train.jsonl` | ~6,100 | the pipeline below | learning |
| `val.jsonl` | ~900 | same pipeline, run only on held-out seed groups | early stopping, per-intent F1, deciding what to fix next |
| `dev_audio.jsonl` | ~40 per teammate | recording session A | finding real Whisper errors, tuning the unhandled threshold (`MIN_CONF`) |
| `test.jsonl` | ~160 seed + ~100 CLINC test + ~40 per teammate recorded | held-out seed groups, CLINC150's own test split, recording session B | scored once per milestone, never looked at row by row |
| robustness suites | generated suites | `robustness.py` on test rows | robustness report, never mixed into accuracy |

**Per intent (train):**

- Every intent, `unhandled` included, gets at least 100 rows (target ~110).
- No intent has more than 1.5× the rows of another, and that includes `unhandled`. Balance with `generate.py --cap`, never by duplicating rows.
- Raise `validate.py --min` from 30 to 100 for the final train file.

**Train mix (~6,100 rows):**

| Source | Share | Rows | Notes |
|---|---|---|---|
| Seed + templates | ~49% | ~3,000 | ~85 per command intent |
| Paraphrases (back-translation) | ~16% | ~1,000 | slot and glossary terms must survive |
| Whisper round-trip of the rows above | ~25% | ~1,500 | |
| `unhandled` | ~2% | ~110 | CLINC150, EVA chatter, keyword traps |
| LLM fill (local model via Ollama) | ≤8% | ≤500 | only for intents that are weak on val |

Every row carries a `source` tag (`seed`, `tmpl`, `para:bt`, `asr:piper`, `clinc`, `chatter`, `llm`), a `group` and a `parent` (the seed row it came from). `train.py` reports val accuracy per source, so we can tell which source helps.

### Two rules

1. **Split by phrasing group before generating anything.** Each seed row gets a skeleton: its text with the slot name and glossary terms replaced by placeholders ("take me to crater rim" → "take me to \<SLOT\>"; "what's my pri o2" → "what's my \<PRIMARY\> \<OXYGEN\>"). Rows with the same skeleton form one group. Then skeletons of the same intent whose sentence embeddings are close (cosine ≥ 0.85, joined transitively) are merged, so "take me to X" and "bring me to X" can't land on opposite sides. A group stays on one side of the split, and every template, paraphrase, TTS clip or LLM row made from a row keeps that row's group. If one seed fed both train and test, accuracy would look great here and fall apart at JSC.
2. **Test is human-written or recorded only, and it's frozen first.** No generated, TTS, paraphrased or LLM row ever goes into test or dev.

### Pipeline

**1. Seed set (written by hand, ~1 day)**

- Split the 35 commands across the team.
- Write 12 rows per intent (~420) plus ~60 `unhandled` rows (EVA chatter, keyword traps, space questions that aren't commands).
- Make each of the 12 a different way of asking, not the same sentence reworded. The later steps add wording variety; the seed's job is to cover different sentence shapes.
- Write in the README style: lowercase, short, sometimes with filler, and always say which one (primary/secondary, A/B, gas/liquid). Use `style_reference.md` for the radio feel.
- Run `split_seed.py`. It builds the groups (rule 1) and splits each intent's groups the same way:
  - 1/3 → `test.jsonl` (~4 per intent)
  - 2/3 → train side (~8 per intent). 15% of these groups go to the val pool, the rest to the train pool.
- Steps 3–7 run separately on each pool, so a val row can never be a copy of a train row.
- Test from the seed: ~140 commands + ~20 `unhandled` ≈ 160 rows. CLINC150 adds ~100 more (step 5) and session B adds the recordings.

**2. Record the team (session A now, session B later)**

- Each teammate records ~40 clips on the real headset mic: 35 commands + 5 chatter.
- Transcribe with faster-whisper using exactly the `voice.py` settings (`base.en`, same `HINT`, VAD on). Save the transcript with the label from the script.
- `voice.py --record prompts.jsonl --speaker NAME --session a|b`: shows a line, records it, saves the wav to `data/audio/` (not committed), transcribes it and stores the slot as Whisper spelled it. Rerunning the same command resumes an interrupted session.
- **Session A → `dev_audio.jsonl`**
  - Free to look at as much as we want. Never trained on.
  - Shows which Whisper mistakes really happen (what step 6 should copy).
  - Gives the real word error rate (WER) and is used to tune `MIN_CONF`.
- **Session B → `test.jsonl`**
  - Recorded a few weeks later, from new lines nobody has compared to the training data.
  - Before recording, run every new line through the leakage audit (below) and rewrite any that get flagged.
- About an hour of team time per session.

**3. Templates**

- One file per intent: `data/templates/<intent>.txt`, written from the pool seed rows (not from scratch, never from test rows).
- Each line is one template with choices and optional words, starting with the id of the seed row it came from: `sd8a2ec: {take|bring|get} me to <SLOT> [please]`. Ids are in `data/pools/*/seed.jsonl`. `{|please}` can pick nothing, and both kinds of brackets can nest.
- A template goes to the pool its seed row is in, so val-pool seed rows can have templates too.
- `generate.py` expands them (a ~30-line function, no library needed):
  - picks random choices instead of every combination
  - fills `<SLOT>` from `data/slot_names.txt`
  - tags rows `source: "tmpl"` and copies the seed row's `group` and `parent`
- Filler, jargon swaps, text ASR swaps and casing are not in templates. `generate.py` already does them once, at the end, over everything.
- Aim for ~85 rows per intent after removing near-duplicates. If a template can't get there, let step 4 fill the gap instead of forcing it.

**4. Paraphrases (`paraphrase.py`, ~1,000 rows)**

- Back-translation with MarianMT through `transformers` (already installed), run on the seed and template rows of one pool at a time:
  - English → German → English (`Helsinki-NLP/opus-mt-en-de`, `opus-mt-de-en`)
  - English → French → English (`opus-mt-en-fr`, `opus-mt-fr-en`)
- Sampling is on, so each row gives a few different results. Tag `para:bt`.
- Protect key words: swap the slot, glossary terms and the words that name the intent ("vitals", "storage") for `X1`, `X2`, … before translating (MarianMT copies these through), then put them back.
- Drop the row if:
  - a placeholder got lost or doubled
  - its meaning drifted: cosine to its source below 0.80 ("pull up the vital umbrella")
  - it's close to a row of another split (rapidfuzz ratio ≥ 85)
  - it's a near-duplicate of an existing row (audit below)
  - the primed encoder's closest seed row has a different intent
- Paraphrases keep the parent's `group`.

**5. `unhandled` examples (`fetch_negatives.py`, ~110 rows)**

- Everything here is labeled `unhandled`. Only train-side rows go into training.
- The class is the same size as any command, so aim for variety, not volume: a few rows from many sources beats many rows from a few.
- CLINC150 is one JSON file on GitHub, so download it with `urllib` like `fetch_acronyms.py` does.
- **CLINC150** (~70 rows, CC BY 3.0, `data_full.json`, 150 intents + out-of-scope):
  - ~30 from the out-of-scope split
  - ~30 keyword traps, ~3 each from `tire_pressure`, `gas`, `jump_start`, `mpg`, `oil_change_how`, `time`, `timezone`, `timer`, `alarm`, `weather` (the "battery in a tesla" problem)
  - at most ~2 each (~8 total) from lookalikes `directions`, `distance`, `traffic`, `todo_list_update`. "How do I get to the mall" looks just like a real command, and too much of it teaches the model that "take me to X" is chatter. Watch `Set_navigation_target` / `Complete_task` recall.
  - Train rows come only from CLINC's train split. ~100 rows from its test split go to `test.jsonl`.
- **EVA chatter + keyword traps** (~40 rows, the train-pool part of the ~60 handwritten seed rows, kept as written: no noise, no paraphrasing):
  - radio talk: "nice view", "copy", "stand by", "say again", "press on"
  - reports, not questions: "pressure flags are clear", "I'm at 40 percent"
  - keyword traps: "battery in a tesla", "the fan in my dorm is loud", "my heart rate at the gym"
  - space questions that aren't commands: "how long is a day on the moon"
- Keep `unhandled` at ~110 rows, like every other intent, so the model doesn't treat it as the default guess.
- Check both ways on val:
  - command recall shouldn't drop when `unhandled` is added
  - `unhandled` recall shouldn't drop on val chatter and CLINC rows
- If `unhandled` recall is weak, first tune `MIN_CONF` on `dev_audio.jsonl`. Only then let the class grow, up to the 1.5× cap (~165) and never past it.

**6. Whisper round-trip (`roundtrip.py`, ~1,500 rows)**

This is the step that fixes "add" → "at".

- **Speak** each sampled row with **Piper** (moved from `rhasspy/piper`, archived Oct 2025; GPL-3.0 is fine because we only use it to make data).
  - 5–6 voices, including the multi-speaker `en_US-libritts_r-medium`
  - speaking speed (`length_scale`) between 0.85 and 1.2
- **Make it sound like a helmet mic** with numpy and scipy (scipy comes with scikit-learn), each effect applied at random:
  - band-pass ~300–3400 Hz (`scipy.signal.butter`)
  - low hum for fan noise
  - random volume and clipping
- **Transcribe** with **faster-whisper**, same model and settings as `voice.py`. Keep the transcript with the original intent.
  - Find the slot in the transcript with rapidfuzz (`process.extractOne`, partial ratio) and save it the way Whisper spelled it.
  - Drop the row if the slot can't be found, the transcript is empty, or its length is far from the original.
- **Match real audio (session A):**
  - Measure WER as word-level edit distance with rapidfuzz (`Levenshtein.distance`, `Levenshtein.editops` on word lists).
  - Turn the noise up or down until round-trip WER is close to real WER on `dev_audio.jsonl`.
  - Compare the most common word swaps with the real ones. Add any real ones Piper doesn't produce to `generate.py`'s ASR list.

**7. LLM fill, last and small (`llm_fill.py`, ≤500 rows)**

- An open-weights model running locally in Ollama, called through its REST API with `urllib`.
- Pass a JSON schema in `format`, so every row comes back as `{"text", "intent", "slot"}`.
- Each prompt shows the intent's train-pool seed rows and asks for new ways to say it, in the README style.
- Only for intents below ~0.95 F1 on val. Never pick intents by looking at test.
- Tag rows `"source": "llm"`, read every row, and drop any the leakage audit flags against test, dev or val.

### Training settings

- **Epochs:** keep `TRAIN` in `config.py` (max 10, patience 3, best val-loss epoch kept). At ~6.1k rows and batch 32, that's ~190 steps per epoch, and we expect the best epoch around 3–6.
  - Best epoch 1–2 means val is too easy (generated rows are probably leaking between pools).
  - Best epoch 10 means raise the max to 15.
- **Seeds:** run 3 seeds for every data change and report the mean and range. With ~4 seed rows per intent in test plus recordings, a one-seed difference of 1% is noise.
- **Priming:** unchanged (3 epochs on the primer).

### Checks before every training run

`validate.py` checks the basics: labels, slots, exact duplicates, exact-text test leakage, per-intent minimum, the 1.5× balance ratio, and the source mix printed next to the targets above.

`audit.py` checks similarity and quality with the libraries we already have and writes one HTML report per data build:

| Check | How | Fails when |
|---|---|---|
| Near-duplicates inside train | rapidfuzz `process.cdist` with `fuzz.ratio` | ratio ≥ 90; keep one per cluster, drop the rest |
| Lexical leakage across splits | rapidfuzz `process.cdist` | any train row ≥ 85 with a test, dev or val row |
| Semantic leakage across splits | sentence-transformers `paraphrase_mining` (all-MiniLM-L6-v2, not our trained encoder) | any train/test pair with cosine ≥ 0.92; read every pair above 0.85 |
| Skeleton leakage | `split_seed.py` skeletons | any train skeleton equals a test, dev or val skeleton |
| Label errors | scikit-learn `cross_val_predict` (5-fold logistic regression on embeddings); rank rows by the out-of-fold probability of their own label | review the 50 lowest; fix the row, not the model |
| Diversity per intent and source | mean pairwise cosine of embeddings | a generated source adds rows but barely lowers the intent's mean similarity (it's copying) |
| Round-trip vs real audio | scikit-learn logistic regression telling `asr:piper` transcripts from `dev_audio` transcripts | AUC well above 0.5 means the round-trip doesn't sound like the headset |

Then read ~30 random rows per source, every LLM row and every label-error flag in the report before training, and fix or drop rows in the source files.

### Robustness tests

`robustness.py` builds behavioral tests from test rows, following the CheckList paper's three test types. They're plain Python perturbations, so the `checklist` library isn't needed. Scored separately from accuracy:

- **MFT** (minimum functionality): keyword traps must be `unhandled` ("battery in my car is dead"), and "which one" variants must pick the right intent (primary vs secondary, A vs B).
- **INV** (invariance): adding filler ("uh", "okay so"), radio words ("copy", "over"), casing changes, and real Whisper substitutions from `dev_audio` must not change the intent.
- **DIR** (directional): swapping the slot name must not change the intent; swapping "primary" for "secondary" must change the intent when they are different intents.

The suites are generated from test rows, so they are never trained on and never used to choose what to generate.

### Dependencies

Only three new pip packages: `rapidfuzz`, `sentencepiece` (MarianMT's tokenizer) and `piper-tts`. Ollama is a separate app, used only for step 7. Everything else uses what's already in `requirements.txt` (torch, transformers, sentence-transformers, scikit-learn) plus faster-whisper, which `voice.py` already uses. Add `piper-tts` to the optional voice line next to faster-whisper.

### Code to write

- `fetch_negatives.py`: CLINC150.
- `roundtrip.py`: Piper → numpy/scipy mic noise → faster-whisper, WER calibration. Optional dependencies, like `voice.py`.
- `llm_fill.py`: Ollama calls with a JSON schema.
- `data.py` / `train.py`: use `val.jsonl` when present instead of carving val out of train; report per-source metrics.
- `validate.py`: the basic checks above.
- `audit.py`: the similarity and quality table above, one HTML report per build.
- `robustness.py`: robustness suites.

### Order of work

1. Write the seed, run `split_seed.py`, and record session A. Test is frozen from this point.
2. Templates, paraphrases and negatives → first `train.jsonl` / `val.jsonl` → `audit.py` → first run.
3. Whisper round-trip, calibrated on session A → retrain and compare with step 2 on val and `dev_audio`.
4. LLM fill for weak intents → audit → retrain.
5. Record session B, score test once, run the robustness suites.

### Links

| Resource | Use |
|---|---|
| CLINC150 (`clinc/oos-eval`) · paper | `unhandled` negatives, keyword traps, human-written test negatives |
| NASA-Acronyms | already in `nasa_acronyms_eva.json`; slot and jargon vocabulary |
| CSA Astronautics Vocabulary | check for terms missing from `suits_glossary.json` |
| NASA Mission Transcripts | style reference for writing seed and test (not training data) |
| Helsinki-NLP opus-mt (en-de, de-en, en-fr, fr-en) | back-translation |
| Ollama | local model for the capped LLM fill |
| Piper (`OHF-Voice/piper1-gpl`) · voices | TTS for the Whisper round-trip |
| faster-whisper | same ASR as `voice.py` |
| rapidfuzz | slot recovery, WER, near-duplicates and lexical leakage |
| sentence-transformers (`paraphrase_mining`, `cos_sim`) | semantic leakage, group merging, paraphrase filters |
| CheckList paper (Ribeiro et al., 2020) | MFT / INV / DIR test types for `robustness.py` |

## Next

1. Build the data in the order under Generating data, then `train.py`.
2. Compare primed vs base encoder over 2–3 seeds.
3. Plan `evaluate/`: overfitting checks, calibrating the unhandled threshold, and folding `audit.py` and the robustness suites into it.

## Status

Items marked 👤 need a person; everything else is code.

**Done**

- [x] `suits_glossary.json` (106 terms), `nasa_acronyms_eva.json` (401 acronyms), `style_reference.md`
- [x] `build_primer.py` → `astronaut_primer.json` (550 pairs), `prime.py`, first priming smoke run (29% → 38%)
- [x] `normalize.py`: safe jargon normalization, same in training and prediction
- [x] `generate.py` noise: slot swap, filler, jargon swaps, ASR swaps, casing
- [x] `validate.py` basics: labels, slots, exact duplicates, exact-text test leakage, per-intent minimum
- [x] `train.py` with grouped val split and `--encoder auto | base`
- [x] `voice.py` push-to-talk with faster-whisper
- [x] `split_seed.py`: skeletons, group merging, test / val pool / train pool, test freeze
- [x] `voice.py --record`: prompts → wav + transcript, slot as Whisper spelled it, resumable
- [x] `generate.py`: template expander and assembler (pools, noise, caps, mix report, `train.jsonl` / `val.jsonl`)
- [x] `paraphrase.py`: back-translation through German and French, with filters

**To do**

- [x] 👤 Write `data/slot_names.txt` from the Mission Description
- [ ] 👤 Write the seed: 12 per intent (~420) + ~60 `unhandled`
- [ ] 👤 Run `split_seed.py` once the seed is done. Test is frozen from then on
- [ ] 👤 Write `data/prompts_a.jsonl` (35 commands + 5 chatter) and record session A (~40 clips each) → `dev_audio.jsonl`
- [ ] 👤 Write `data/templates/<intent>.txt` from the train-pool seed rows
- [ ] Run `generate.py` → `paraphrase.py` → `generate.py`
- [ ] 👤 Read ~30 paraphrases per intent: back-translation still lets some wrong meanings through ("take me off the vitals screen")
- [ ] `fetch_negatives.py`: CLINC150
- [ ] `validate.py`: 1.5× balance ratio and source mix
- [ ] `data.py` / `train.py`: read `val.jsonl` (generate.py writes it, but train.py still carves val out of train), report per-source metrics
- [ ] `audit.py`
- [ ] 👤 Read the audit report before each training run (~30 rows per source, every label-error flag)
- [ ] First training run, 3 seeds, primed vs base
- [ ] `roundtrip.py`: Piper → mic noise → faster-whisper, calibrated on `dev_audio.jsonl`
- [ ] 👤 Install Ollama and pull a local model
- [ ] `llm_fill.py`, only for intents below ~0.95 F1 on val
- [ ] 👤 Read every LLM row
- [ ] `robustness.py`
- [ ] 👤 Write new prompt lines and record session B → `test.jsonl`
- [ ] Score test once, run the robustness suites

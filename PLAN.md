# Single-Intent Voice Command Model: Plan

NASA SUITS 2026–27. One utterance in, one of 36 labels out, plus a slot name for 5 labels.

Sources cited below:
- **MD** = `docs/nasa-suits-mission-description.pdf` (section numbers as in the PDF).
- **Siri** = `docs/siri-deep-dive.pdf` (cited by slide title).

Format for each step: **What**, **Why**, **Tool**, **Check**.

---

## 0. Summary

- **Pipeline:** headset mic → link to the MacBook (cable or WiFi) → `faster-whisper` `base.en` → text normalizer → MiniLM (intent head + slot tagging head) → calibrated reject → closed-list slot resolver → result back over the link → HUD.
- **Where it runs:** Whisper and MiniLM run on the MacBook. The HUD reaches it either by cable or over WiFi, which may be weak. Both must work (§7.5).
- **Shipped model:** one `all-MiniLM-L6-v2` (22M), fine-tuned, distilled from bigger teachers, exported to ONNX int8. Nothing bigger runs at prediction time.
- **Output:** `{text, selected_intent, confidence, top5, slot?}`. `slot` only on the 5 slot labels. Low confidence → `unhandled`.
- **Guardrail:** the model never produces telemetry values. It only picks a label and a name. The HUD reads values from the TSS. (MD §2b: "establish guardrails for mission-critical aspects"; MD §3a.)
- **Data:** ~3,850 handwritten rows (100 per command label, 350 `unhandled`), split by phrasing group before any generator runs. Generators only add Whisper errors, Whisper silence hallucinations and public negatives, each with a cap.
- **Gates:** dedup, leakage, diversity (~0.5 cosine), label errors, balance, behavioural tests. Data that fails a gate is not trained on.
- **Done:** targets in §6.3 met on the sealed test set, latency budget in §7.3 met on the target device.

---

## 1. Ground rules

- **Single-intent only.** One label per utterance. No multi-intent, no domains, no hierarchy in the model.
  - Why: task constraint. Siri "SUITS Version" under "Why Classify in Levels": 36 labels are few enough for one softmax.
  - The domain idea survives only as *blocks* for reading the confusion matrix (Siri "Errors Live Inside Blocks").
- **No template expansion.** Every row is a genuinely different sentence written by a person.
- **Generators** only add what a person can't write well. Each has a reason and a cap (§3.8).
- **Split first, generate second.** Generated rows inherit the split of their parent row.
- **Test is sealed.** Read only at milestones (§6.4). Never mined from (Siri "Data Engine, SUITS Version": "Never start from test errors").
- **Small at runtime, big in training.** Teachers, LLM checkers and bigger encoders are fine offline.

---

## 2. Output contract and labelling guide

### 2.1 Output

- `text`: the normalized Whisper transcript (1-best; n-best used internally, §5.5).
- `selected_intent`: one of the 36 labels, exact case.
- `confidence`: calibrated probability of `selected_intent` (after temperature scaling, §5.4).
- `top5`: five `(label, prob)` pairs, calibrated.
- `slot`: only for `Get_coordinates`, `Set_navigation_target`, `Add_waypoint`, `Delete_waypoint`, `Complete_task`.
  - Shape: `{span, resolved, candidates[≤3], status}`. `status` is `match`, `ask` (close call) or `new` (Add_waypoint only).
- Reject rule: if calibrated confidence < `MIN_CONF`, or the reject model says reject (§5.4), output `unhandled`. Keep the original top5 for logs.

### 2.2 Labelling guide (write before any data) 👤

- **What:** a one-page guide. For each label: meaning, 3 example sentences, its "which-one" siblings, its jargon, and what does *not* belong.
- **Why:** writers and checkers need the same rules, or label noise appears. Most noise comes from ambiguous phrasings.
- **Decided rules (Open questions 1, 3, 4, decided 2026-11-22):**
  - Unqualified requests map to primary / total, matching the DCU nominal state in MD Appendix A ("Verify OXY – PRI", "FAN – PRI", "CO2 – A"):

| Unqualified request | Label |
|---|---|
| "oxygen", "O2 level", "how much air" | `get_oxy_pri_storage` |
| "oxygen pressure", "tank pressure" | `get_oxy_pri_pressure` |
| "battery", "power level" | `get_primary_battery_level` |
| "fan", "fan speed" | `get_fan_pri_rpm` |
| "pressure", "suit pressure" | `get_suit_pressure_total` |
| "scrubber" | `get_scrubber_a_co2_storage` (A is nominal) |
| "CO2", "CO2 level" | `get_suit_pressure_co2` |
| "coolant" | `get_coolant_storage` |

  - Value read-backs from IV/ground ("copy, primary O2 forty-seven") → `unhandled`. They are not requests.
  - UIA/DCU switch callouts read aloud ("OXY PRI", "BATT UMB", "O2 vent open") → `unhandled`. They are procedure steps (MD Appendix A), not HUD commands.
  - Out-of-scope requests (camera, spectrometer, comms) → `unhandled`.
  - "Cancel", "never mind", negated commands ("don't open the map", "no, not the map") → `unhandled`.
  - `Add_waypoint` takes the spoken name verbatim. The HUD does not confirm it.
  - "Show my vitals" → `open_menu_vitals`. "What's my heart rate" → `get_heart_rate`. Rule: a single value → `get_*`; a panel → `open_menu_*`.
- **Written:** [LABELING_GUIDE.md](LABELING_GUIDE.md) (draft 2026-11-22).
- **Check:** two people label the same 100 random pilot rows blind. Agreement ≥ 95%. Every disagreement becomes a guide rule.

---

## 3. Data creation

### 3.1 Row schema

One JSONL row per sentence:

| Field | Meaning |
|---|---|
| `id` | stable id |
| `text` | the sentence as written |
| `label` | one of 36 |
| `slot_text` | exact substring that is the slot, or null |
| `group_id` | phrasing group (§3.7) |
| `split` | train / val / test |
| `source` | handwritten, handwritten_whisper_style, tts_roundtrip, whisper_silence, public_neg |
| `parent_id` | for generated rows: the handwritten row it came from |
| `writer` | who wrote it |
| `tags` | slice tags: pair, trap, jargon, filler, fragment, long, slot_unseen, … |

- Slot spans for the tagging head come free from `slot_text` (Siri "SUITS Version": "labels come free from the stored slot").
- **Check:** a schema validator passes on every row; every `slot_text` is a substring of `text`; slot labels always have a slot.

### 3.2 How much

| Pool | Handwritten | Train | Val | Test |
|---|---|---|---|---|
| Each of 35 command labels | 100 | 60 | 20 | 20 |
| Command total | 3,500 | 2,100 | 700 | 700 |
| `unhandled` | 350 | 210 | 70 | 70 |
| Data-engine rows (later, §5.8) | ≤ 30 per label per round | train only | – | – |

**Why 100 per label:**
- 60 train rows per label is enough to fine-tune a small sentence encoder for intent. Gains flatten after that; quality and variety matter more.
- 20 test rows per label gives a per-label score in 5% steps. Fewer makes per-label targets meaningless.
- With 3–4 writers, each writes ~25–35 rows per label. That is about where one person runs out of genuinely different phrasings. Past ~120 rows, writers drift into near-copies and fail the diversity gate.
- Effort: ~3,850 rows at ~1 minute each ≈ 65 person-hours. Doable for a small team over 2–3 weeks.
- Which-one pair labels get more only through the data engine, targeted at real errors, not up front.

### 3.3 Pilot first 👤

- **What:** write 20 rows for 6 labels (`get_oxy_pri_storage`, `get_oxy_sec_storage`, `get_heart_rate`, `Set_navigation_target`, `open_menu_tasks`, `unhandled`). Run every gate in §4.
- **Why:** gate thresholds (cosine ~0.5 etc.) depend on the ruler model and on how short commands are. Calibrate them once on real rows, then freeze.
- **Check:** thresholds written into the guide and frozen. Writers have seen what "too similar" looks like.

### 3.4 Variety rules per command label (100 rows)

Each label must cover these axes. Counts are minimums; one row can count for two axes.

| Axis | Min rows | Example (get_heart_rate) |
|---|---|---|
| Question | 15 | "what's my pulse doing" |
| Imperative | 15 | "give me heart rate" |
| Fragment, 1–3 words | 10 | "BPM?" |
| Long, ≥ 12 words | 10 | "I've been hauling samples up this slope for a while, how fast is my heart going" |
| Filler / disfluency | 10 | "uh, okay, um heart rate check" |
| Jargon / NASA callout style | 10 | "HR check" / "say biomed HR" |
| Indirect / need-based | 10 | "am I overdoing it right now" (only if the guide maps it here) |
| Whisper-style spelling (§3.5) | 10 | "hard rate" |
| Keyword not first | 10 | "can you check for me, the heart rate" |

- **Writers:** ≥ 3 writers per label. No writer > 50% of a label.
- **Opening words:** no 2-word opening covers > 15% of a label.
- **Keyword reliance:** the label's main keyword ("heart rate") appears in ≤ 60% of its rows where synonyms exist. Warning only, since "scrubber A" has no synonym.
- **Why:** variety on these axes is what makes the test hard and the scores honest. Siri "Supervision Is Data": quality comes from the sources, not model tuning.
- **Check:** a per-label coverage report counts tags per axis; the diversity gate (§4.4) passes.

### 3.5 Hard cases (written by hand)

- **Which-one pairs and blocks.** For each pair, ≥ 20 rows per side state the discriminator in varied ways ("primary", "pri", "main", "number one", "the first tank"; "B", "bravo", "second scrubber").
  - Blocks: oxygen (pri/sec × storage/pressure, consumption); suit pressure (oxy / co2 / other / total) and helmet co2; CO2 family (suit pressure, helmet, scrubber A/B storage, production); coolant (storage / gas / liquid); fans (pri/sec); batteries (pri/sec); biomed (heart rate, temperature); time and warnings; menus; navigation (coordinates / set target / reroute / add / delete waypoint, open navigation); tasks (open tasks / complete task).
  - Why: Siri "Errors Live Inside Blocks". Mistakes concentrate inside blocks.
- **Keyword traps** (≈ 120 rows, labelled `unhandled` or the right other label). A label's keyword appears but the meaning differs.
  - "this rock reads high in oxygen on the spectrometer" → `unhandled`.
  - "close the O2 vent" → `unhandled`, not `close_menu`.
  - "open the vent on the UIA" → `unhandled`, not `open_menu_uia`.
  - "copy, secondary battery at ninety" (read-back) → `unhandled`.
- **Jargon:** EVA, EV, IV, DCU, UIA, TSS, PLCD, PLCT, PSMD, XRF, EMU, ppO2, ppCO2, "suit P", "pri", "sec", "UMB", "BPM", "comm check". Source: MD §2c, §3, Appendix A.
- **Whisper-style rows** (~10% per label). Writers use a *confusion phrasebook* built from real round-trip outputs (§3.8): "add" → "at", "O2" → "o two" / "oh two", "CO2" → "C O two" / "see oh two", "UIA" → "you I A", "pri" → "pre", "sec" → "sack" or "seck", "scrubber" → "scrubbed", "waypoint" → "way point", "coolant" → "cool and".
  - Why: the person writes the sentence, the phrasebook only tells them which errors are real.
- **Slot rows.**
  - Build the slot catalogue first 👤: POIs ("POI 1"…), "lander", "airlock", team waypoint names, and tasks from MD §2c (PLCD deployment, PLCT teardown, geology photos, sample collection, sample marker, XRF reading, seismometer, charge sensor, retroreflector alignment, sensor canister, egress, ingress).
  - Split names before writing: ~70% *train names*, ~30% *held-out names*. Rows that use held-out names go only to val/test (tag `slot_unseen`).
  - Vary name position, add filler around names, include misspelled names in Whisper-style rows.
  - No single name in > 15% of a slot label's rows.
  - Why: Siri "Why Speech Makes Resolution Hard": people say names the system has never seen, and ASR misspells them. MD §4a.5: drop a pin any time; MD §2c: drop a pin at each POI.

### 3.6 Where `unhandled` comes from

| Source | Rows | Reason | Cap |
|---|---|---|---|
| Handwritten keyword traps, incl. ~20 negated commands and cancellations | ~120 | hardest negatives; only a person knows the domain | – |
| Handwritten radio chatter and read-backs (EV↔IV, MCC) | ~80 | most common real false trigger | – |
| Handwritten out-of-scope commands (camera, spectrometer, comms, photos) | ~60 | plausible requests we don't support | – |
| Handwritten self-talk / small talk | ~50 | astronaut thinking aloud during geology (MD §2c) | – |
| Handwritten fragments / false starts ("open the", "what's my") | ~40 | VAD cuts and hesitations | – |
| Whisper silence/noise hallucinations | ≤ 50 distinct | `base.en` invents "Thank you." etc. on noise; people can't guess these | ≤ 50 in train |
| Public out-of-scope sets (train/val): CLINC150 `oos` split | ≤ 300 | breadth of off-topic speech | ≤ 1.5× handwritten unhandled train |
| Public, test only: a different dataset (e.g. MASSIVE non-matching intents) | 150 | tests generalisation to unseen chatter, not memorised CLINC style | test only |

- **Filter public rows:** drop any whose nearest command row has cosine ≥ 0.6 under the ruler model, then a person checks a 10% sample. Some public "off-topic" rows are real commands for us ("what's my heart rate").
- **Check:** `unhandled` share of train is 15–25%; public share of `unhandled` train ≤ 60%; filtered rows logged.

### 3.7 Split by phrasing, before generation

- **What:**
  1. Normalize text (§4.1).
  2. Build *phrasing groups*: link two rows if MinHash Jaccard ≥ 0.5 (word shingles) **or** ruler cosine ≥ 0.80. Connected components are groups.
  3. Force rows with held-out slot names into val/test.
  4. Assign whole groups to train/val/test 60/20/20, stratified by label.
  5. Writer hold-out: for each label, ≥ 10 of its 20 test rows come from a writer with no train rows in that label.
  6. Freeze a split manifest (`id → split`, plus a hash of the file). Commit it.
- **Why:** splitting by row lets near-copies land on both sides and inflates scores. Writer hold-out tests whether the model learnt the intent or one person's style.
- **Tool:** datasketch `MinHash`, `MinHashLSH.insert` / `.query`; sentence-transformers `util.community_detection(emb, threshold=0.80, min_community_size=2)` as a cross-check on groups; scikit-learn `StratifiedGroupKFold` (or a simple greedy assigner) for the group split.
- **Check:** no group spans two splits; leakage gate (§4.3) passes; manifest hash unchanged in every later step.

### 3.8 Generators (each with reason and cap)

| Generator | What it adds | Reason | Cap | Inherits split |
|---|---|---|---|---|
| TTS → noise → Whisper round-trip | real `base.en` errors on our sentences | people can't predict Whisper's exact errors | ≤ 1 kept variant per source row; ≤ 50% of a label's handwritten train rows | yes |
| Whisper on silence / noise clips | hallucinated text ("Thank you.", "you") labelled `unhandled` | only Whisper knows what it invents | ≤ 50 distinct | yes (assigned once) |
| Public negatives | off-topic text | breadth a small team can't write | §3.6 | train/val from CLINC; test from a different set |

- **Round-trip details:**
  - TTS: macOS `say` with ≥ 6 voices, varied rate. Optional: Piper voices for more accents.
  - Noise: fan hum, breathing, radio squelch, pink noise. SNR 5–20 dB. Resample to 16 kHz; optional radio band-pass (300–3400 Hz).
  - Link path: half the clips go in raw (cable), half through the WiFi audio codec at the shipped bitrate (§7.5), with 0–5% simulated packet loss. `ffmpeg` with `libopus` does the encoding. Tag `link=cable|wifi`.
  - Whisper: exactly the shipped settings (model, beam size, `initial_prompt` hint, VAD). Also save the n-best list (§5.5).
  - Keep a variant only if it differs from the source after normalization.
  - If the round-trip lost the discriminator ("primary" dropped) or the slot name became unrecoverable, **drop** the variant. Don't relabel it.
  - Build the confusion phrasebook from word-level diffs (source vs transcript) across all round-trips. Writers use it (§3.5).
- **Not allowed:** LLM-written commands, paraphrase generators, rule-based word swaps for training. A person can write these, and they create near-copies.
- **Check:** caps enforced per label; every generated row has a `parent_id` in the same split; 10% human spot-check of kept round-trips 👤.

### 3.9 Audio: stand-in now, real recording later

- **Now (stand-in):** synthetic audio from §3.8.
  - `dev_audio_synth` = round-trips of val rows. Used for threshold setting and n-best decisions.
  - `test_audio_synth` = round-trips of test rows. Sealed with test.
  - Limit: TTS is cleaner than a stressed human in a suit. Whisper errors are underestimated. Treat synthetic audio scores as optimistic.
- **Later (when recording resumes) 👤:**
  - Each team member reads val and test prompts (never train) through the real headset, with fan noise, while walking or carrying weight. About 10 per label per speaker. Also record free-form chatter for `unhandled`.
  - `dev_audio_real` (reset `MIN_CONF` and reject model on it; Siri "Judge at an Operating Point, SUITS Version": set on dev_audio so 98% of commands pass) and `test_audio_real` (sealed, held out by speaker).
  - Add misrecognitions found there to the phrasebook. Use them only through the data engine (train only).
- **Check:** every result table states whether audio was synthetic or real.

---

## 4. Data checking and validation

All gates run on every data change, before any training. Any failure blocks training.

### 4.1 Normalization (shared with prediction time)

- **What:** Unicode NFKC, lowercase, unify quotes and dashes, strip punctuation, collapse whitespace. Keep digits and letters as Whisper wrote them. No word rewriting by default.
- **Why:** train and prediction must see identical text. Word rewriting (e.g. "o two" → "o2") is tested as an experiment on val, not assumed.
- **Check:** the same function is used in data checks, training and the runtime (one module, unit tested on 30 Whisper outputs).

### 4.2 Duplicates and near-duplicates

- **What:**
  1. Exact duplicates on normalized text → delete.
  2. Lexical near-duplicates: MinHash on word 1-grams + 2-grams (char 4-grams for rows ≤ 4 words), `num_perm=128`, LSH at Jaccard 0.7. Confirm candidates with exact Jaccard.
  3. Semantic near-duplicates: ruler-model pairs with cosine ≥ 0.92 → near-dup; 0.85–0.92 → review.
  4. Same text with different labels → label conflict, goes to §4.5.
- **Tool:** datasketch `MinHash.bulk` / `MinHash.update_batch`, `MinHashLSH(threshold=0.7, num_perm=128)`, `MinHash.jaccard`, `LeanMinHash` to store signatures; sentence-transformers `util.paraphrase_mining(ruler, sentences, top_k=10)`.
- **Check:** zero exact dups; zero unreviewed pairs ≥ 0.92 within a label; writers rewrite (not delete) near-dups so counts stay at 100 👤.

### 4.3 Leakage across splits

- **What:**
  1. Exact normalized match across splits → fail.
  2. MinHashLSH (Jaccard ≥ 0.5) built on train, queried with every val and test row → each hit reviewed.
  3. Embedding nearest neighbour: for each test (and val) row, the top-5 most similar train rows. Any cosine ≥ 0.85 → review.
  4. Descendants: a generated row's `parent_id` must be in its own split.
  5. Slot names: `slot_unseen` rows contain no train name.
- **Fix:** a confirmed leak moves the whole group to train, and a writer writes a fresh replacement for test. Never move train rows into test.
- **Tool:** datasketch `MinHashLSH.query`; sentence-transformers `util.semantic_search(test_emb, train_emb, top_k=5)`.
- **Check:** zero exact matches; zero unreviewed hits; every test row's max train cosine < 0.90.

### 4.4 Diversity gate

- **Ruler model:** `sentence-transformers/all-mpnet-base-v2`, frozen, pinned revision `e8c3b32edf5434bc2275fc9bab85f82640a19130` (student `all-MiniLM-L6-v2`: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`), never fine-tuned. `encode(..., normalize_embeddings=True)`, `util.cos_sim`. A separate model from the one we train, so the ruler doesn't drift with our training.
- **Also report** (not gated) the same numbers with base `all-MiniLM-L6-v2` for reference.
- **Metrics:**

| Metric | Definition | Gate (re-tune once after pilot, then freeze) |
|---|---|---|
| Within-label mean | mean cosine over all pairs i<j inside one label, inside one split (handwritten rows only) | target ≈ 0.50; warn > 0.55; **fail > 0.62** per label; warn < 0.30 (possible off-label rows) |
| Dataset mean | mean of the 36 within-label means | 0.45–0.55 |
| Within-label max | highest pair cosine inside a label | < 0.92 (else near-dup, §4.2) |
| Test → train NN | for each test row, max cosine to any train row | p95 ≤ 0.85; max < 0.90 |
| Test vs train self-NN | median test→train NN vs median train leave-one-out NN | test median ≤ train median + 0.02 (test is no closer to train than train is to itself) |
| Distinct-2 per label | unique word bigrams ÷ total bigrams | ≥ 0.6 (warn) |
| Cross-label centroid similarity | cosine between label centroids | report only; high pairs feed §3.5 pair writing |

- **Why:** user target "around 0.5"; near-copies make the data easy and scores misleading. NN between splits is the honest leakage-by-similarity check.
- **Check:** gate report is generated on every data change and saved with the run.

### 4.5 Label errors

- **What:** three independent signals; a row flagged by ≥ 2 goes to a person first.
  1. **Out-of-fold predictions:** 5-fold cross-validation by group on train+val with the current MiniLM (or teacher). Flag rows whose out-of-fold top-1 ≠ label with prob ≥ 0.8.
  2. **kNN vote:** top-5 ruler neighbours; flag if ≥ 4 have another label.
  3. **Local LLM second opinion:** Ollama, a 7–9B instruct model that fits the MacBook Air. Give it the labelling guide and label table, ask for one label. Flag disagreements.
- **Tool:** sentence-transformers `util.semantic_search`; Ollama; your own cross-validation loop.
- **Check:** review queue empty 👤; report flag rate per label; < 2% of rows changed after round 1 (if more, the guide is unclear, fix the guide).

### 4.6 Balance

| Check | Gate |
|---|---|
| Command labels, train handwritten count | within ±10% of the median |
| `unhandled` share of train | 15–25% |
| Round-trip rows per label | ≤ 50% of the label's handwritten train rows |
| Writer share per label | ≤ 50% |
| Slot name share per slot label | ≤ 15% per name |
| Test rows per label | exactly 20 (70 handwritten `unhandled` + public test negatives) |

- **Why:** imbalance shifts the softmax prior and the reject threshold.
- **Check:** balance table in the gate report. No class weights needed if the gate passes.

### 4.7 Behavioural tests (CheckList)

- **Environment:** `checklist` needs its own env (Python 3.10/3.11). It depends on spaCy, `patternfork-nosql`, Jupyter and old `transformers`, which won't install on 3.15. Exchange through files: build with `TestSuite.to_raw_file`, predict in the main env, score with `TestSuite.run_from_file`, read `TestSuite.summary`.
- **Bases:** handwritten minimal pairs plus test-split rows. Never train rows. Tests are for measurement, never training data.

| Test | Type | What it checks | Tool |
|---|---|---|---|
| Keyword traps | MFT | traps → `unhandled` or the right label (~100 new handwritten cases) | `MFT(data, labels=...)` |
| Procedure callouts | MFT | MD Appendix A lines read aloud → `unhandled` | `MFT` |
| Whisper hallucinations | MFT | "thank you", "you", "…" → `unhandled` | `MFT` |
| Which-one minimal pairs | MFT | both sides of each pair labelled right | `MFT` |
| Filler / disfluency | INV | inserting "um", "uh", "okay so" doesn't change the label | `Perturb.perturb` + custom fn, `INV` |
| Punctuation / casing | INV | label unchanged | `Perturb.strip_punctuation`, `INV` |
| Contractions | INV | label unchanged | `Perturb.contractions`, `Perturb.expand_contractions` |
| Whisper confusions | INV | phrasebook swaps ("add" → "at") keep the label | `Perturb.perturb` + phrasebook fn |
| Slot name swap | INV | swapping in held-out names keeps the intent; slot follows the name | `Perturb.perturb` + catalogue fn |
| Politeness / wake prefix | INV | "please", "hey HUD" keep the label | `INV` |
| Discriminator flip | DIR | primary↔secondary, A↔B, gas↔liquid, suit↔helmet must flip to the sibling | `DIR` + `Expect.pairwise` |
| Negation / cancel | MFT | negated commands and "never mind" → `unhandled` | `Perturb.add_negation` on test-split bases + `MFT` |
| Unqualified requests | MFT | "oxygen?", "battery", "scrubber" → the primary / total label from §2.2 | `MFT(data, labels=...)` |

- **Check:** MFT failure ≤ 2% per test; INV flip rate ≤ 3%; DIR ≥ 97% correct flips. Results saved per run.

---

## 5. The model

### 5.1 Architecture

- **What:** `all-MiniLM-L6-v2` encoder → mean pooling → dropout → linear intent head (36). A token head (linear, B/I/O) on the token states for slot spans. One forward pass gives both.
- **Why:** Siri "SUITS Version" (Slots): a tagging head on the same MiniLM handles new names and costs almost nothing in latency.
- **Max length:** 32 tokens. Check: ≥ 99.9% of rows fit without truncation.
- **Check:** shapes and a forward pass on 10 rows; latency of the forward pass measured early (§7.3).

### 5.2 Training recipe (student)

- **Step A, floor:** frozen MiniLM embeddings + logistic regression. Takes minutes. Every later step must beat it on val.
- **Step B, fine-tune:** end-to-end, cross-entropy on intent + λ · cross-entropy on B/I/O (slot rows only). AdamW, lower learning rate for the encoder than the heads, early stop on val macro F1. Run 5 seeds; report mean ± std.
- **Step C, experiment: contrastive warm-up.** Before Step B, fine-tune the encoder with label-aware triplet loss so same-label rows cluster and which-one siblings separate.
  - Tool: sentence-transformers `SentenceTransformerTrainer` with `BatchHardSoftMarginTripletLoss` (or `BatchAllTripletLoss`) and `GroupByLabelBatchSampler`.
  - `util.mine_hard_negatives` is label-unaware, so only use it with a label filter.
  - Keep only if it improves the which-one pair slice on val by ≥ 0.5 points.
- **Check:** val metrics per seed logged; the chosen recipe beats Step A.

### 5.3 Ensembles and distillation

- **Teachers (training only):** fine-tune 3 bigger encoders with the Step B recipe (for example `all-mpnet-base-v2`, a `bge`-large and an `e5`-large). Different models make different mistakes (Siri "Why Different Models?").
- **Combine teachers with a geometric mean** of probabilities. A strong "no" from one teacher drags a label down (Siri "Two Ways to Average").
- **Distil into MiniLM:** loss = α · cross-entropy(hard label) + (1−α) · T² · KL(teacher_T ‖ student_T), with T between 2 and 4. Tune α and T on val.
  - Transfer set: train rows, their round-trips and the train-side public negative pool. The teacher labels extra unlabelled in-domain text for free. Never val/test inputs.
  - sentence-transformers `DistillKLDivLoss` distils *similarity scores*, not class logits. The logit loss is a small custom loss.
- **Ensemble without latency cost:** average the weights of 3–5 student fine-tunes from the same start ("model soup"). If the student stays a SentenceTransformer, `SentenceTransformer.merge(models, method="linear")` does this (needs `mergekit`, the `merge` extra). Keep only if val ≥ the best single student.
- **Runtime ensemble of several MiniLMs:** dropped by default (costs k× latency). Revisit only if the latency budget has room and val gains ≥ 1 point.
- **Check:** teacher ensemble beats the student on val (else no distillation); distilled student beats the Step B student on val macro F1 and pair slice.

### 5.4 Calibration and the reject threshold

- **Temperature scaling:** fit one temperature on val to minimise negative log-likelihood. Report ECE before/after.
- **Reject model:** a tiny logistic regression on: calibrated top-1 prob, margin top-1 − top-2, Whisper `avg_logprob`, Whisper `no_speech_prob`, and n-best agreement (§5.5). Siri "SUITS Version – More Rejection Signals".
  - Fit on `dev_audio_synth` now, re-fit on `dev_audio_real` later.
  - If false accepts stay high: add "transcribe with and without the hint and compare" as one more feature (Siri "A Second, Unbiased Listener"). It doubles Whisper cost, so it is off by default.
- **Operating point:** set `MIN_CONF` so 98% of real commands pass (Siri "Judge at an Operating Point"). Report the chatter pass rate at that point, never a single accuracy number alone.
- **Destructive labels:** `Delete_waypoint` and `Complete_task` already get HUD confirmation, so no extra threshold.
- **Check:** ECE ≤ 0.03 on val after scaling; recall/precision curve saved; the threshold value is stored with the model.

### 5.5 Whisper n-best

- **What:** get the top 3 transcripts with scores from CTranslate2 under faster-whisper (`generate` with `num_hypotheses=3`, `return_scores=True`; confirm exact arguments). Run MiniLM on all 3 in one batch. Weight by a softmax over length-normalised scores. Sum: P(intent | audio) = Σ P(intent | h) · P(h | audio).
- **Why:** Siri "N-Best Lists": the right transcript is often second. The decoder already runs beam search, so extra hypotheses are nearly free. MiniLM on a batch of 3 costs little more than on 1.
- **Keep if:** on `dev_audio_synth`, accuracy +≥ 0.5 points or fewer false rejects at the same chatter rate. Otherwise ship 1-best.
- **Check:** A/B table 1-best vs n-best; latency delta measured.

### 5.6 Slots: tagging + closed-list resolver

- **Tagger:** the B/I/O head from §5.1 finds the span.
- **Resolver:** match the span to the slot catalogue by spelling (rapidfuzz) and sound (Double Metaphone). Keep the top 2–3 with scores (Siri "SUITS Version" for slots).
  - Clear winner → `match`. Close call → `ask` (the HUD asks). No match on `Add_waypoint` → `new`: the span is kept verbatim with no HUD confirmation (decided). A misspelt name stays misspelt. Later `Delete_waypoint` / `Set_navigation_target` still find it by sound matching. No match on the other 4 → `ask`.
  - If the tagger finds no span: try the resolver on the whole text, then `ask`.
- **Why:** Siri "Two Different Jobs": tagging and resolution are separate. Whisper misspells names; reranking several guesses beats trusting one.
- **Check:** slot targets in §6.3 on known and unseen names; resolver thresholds tuned on val only.

### 5.7 Short-term context (optional, outside the model)

- **What:** a rule. Keep the last intent and slot for ~10 s. A qualifier-only follow-up ("what about secondary?", "and B?") maps to the sibling label of the last intent.
- **Guardrails (Siri "SUITS Version – Guardrails"):** only if the qualifier is a valid sibling; clear after ~10 s or any `unhandled` turn. Stale context must never fire a command.
- **Why it stays single-intent:** the follow-up still produces one label. The model alone labels qualifier-only fragments `unhandled`; the rule is a post-step.
- **Check:** a scripted test of 50 sequences with timing; zero stale carry-overs; feature flag can turn it off.

### 5.8 Data engine (mining around errors)

- **What, each round:**
  1. Train → predict on val and `dev_audio_synth` (later `dev_audio_real`).
  2. Collect errors: wrong labels, false accepts, false rejects, slot misses.
  3. Cluster errors with the ruler model (`util.community_detection(threshold=0.7, min_community_size=2)`). Each cluster becomes a *writing brief*: "boundary X vs Y, here are the misses".
  4. For `unhandled` errors, mine the 10–20 nearest unused public rows (`util.semantic_search`) → a person reviews → add to train.
  5. For command errors, people write 10–20 new rows per brief 👤 (train only), and check the nearest train rows for label errors.
  6. Re-run all gates.
- **Caps:** ≤ 30 new rows per label per round; ≤ 3 rounds; balance gate still applies.
- **Why:** Siri "Better Data, Not a Better Model" and "Search Around Each Error". Neighbours of real errors are in-scope and hard; most-uncertain rows are mostly junk (Siri "Why Not Label the Most Uncertain?"), so uncertainty sampling is dropped.
- **Caution:** val becomes optimistic after rounds of mining. Only test decides "done".
- **Check:** per round, val pair slice and false-accept rate improve; the gate report passes.

### 5.9 Slices

- **What:** report F1 per slice on every run: handwritten clean, Whisper-style handwritten, TTS round-trip, keyword traps, which-one pairs (per pair), jargon, filler, fragments, long rows, slot rows, unseen slot names, per writer, public negatives.
- **Why:** Siri "Slices": a slice can get worse while overall goes up. Per-source tags are already a slicing (Siri "SUITS Version" under Slices).
- **Check:** no slice drops > 2 points vs the previous accepted run without an explanation.

### 5.10 Siri ideas: kept and dropped

| Siri idea | Decision | Reason |
|---|---|---|
| Domain chooser / levels | Dropped in the model; kept as blocks for reading errors | task constraint; 36 labels fit one softmax |
| 7-model BiLSTM ensemble at runtime | Dropped at runtime; kept as teachers + distillation + weight soup | latency |
| Geometric mean | Kept for teachers | strong "no" matters for look-alike labels |
| Parser (BIO tagging) | Kept as a second head on MiniLM | free latency, handles new names |
| Resolver with several guesses | Kept (rapidfuzz + Double Metaphone, top 3) | Whisper misspells names |
| Trigger cascade, speaker ID, SDSD | Dropped from this plan | belongs to VAD / push-to-talk, not the text model |
| Second decoder | Optional reject feature only | doubles Whisper cost |
| Lattice doubt | Kept in cheap form: `avg_logprob`, `no_speech_prob`, n-best spread | free signals |
| Operating point | Kept | the core of the reject decision |
| N-best marginalising | Kept if it wins on dev audio | right transcript is often second |
| Data engine | Kept; humans write, public pool is mined | data beats model tuning |
| Label most uncertain | Dropped | mostly junk per the deck |
| Slices / tag by source | Kept | catches hidden regressions |
| ReALM / LLM context | Dropped | latency; hallucination risk (MD §2b) |
| Carry-over rule | Kept as optional rule | single-intent safe with guardrails |
| LLM in the middle (Siri AI 2026) | Dropped at runtime | latency, hallucination (MD §2b); LLM used only offline as a checker |

---

## 6. Evaluation

### 6.1 Sets

| Set | Contents | Used for |
|---|---|---|
| train | 60 per label + generated + data-engine rows | training only |
| val | 20 per label + round-trips | early stop, tuning, temperature, data engine |
| `dev_audio_synth` | TTS round-trips of val rows | reject model, `MIN_CONF`, n-best choice |
| test | 20 per label + 70 handwritten `unhandled` + 150 public negatives from a dataset not used in train | sealed; milestones only |
| `test_audio_synth` | TTS round-trips of test rows | sealed; milestones only |
| behavioural suite | §4.7 | every run |
| `dev_audio_real`, `test_audio_real` | later recordings, by speaker | later; replaces the synthetic sets for decisions |

### 6.2 What is measured

- Accuracy and macro F1 over 36 labels, after the reject step.
- Per-label accuracy with Wilson 95% intervals (20 rows per label is small).
- `unhandled` operating point: command pass rate vs chatter pass rate; full curve.
- **Wrong-command accepted rate:** a real command accepted with the wrong label. This is the dangerous error.
- Which-one pairs: wrong-side rate per pair and confident swaps (wrong side with confidence ≥ 0.9).
- Slots: span F1, resolved exact match, resolver top-3 recall, `ask` rate.
- Calibration: ECE, reliability plot.
- Block confusion matrix (blocks from §3.5), per Siri "Errors Live Inside Blocks".
- Slices (§5.9), behavioural suite (§4.7), latency (§7.3).

### 6.3 Targets ("done")

| Metric | Set | Target |
|---|---|---|
| Overall accuracy, 36 labels, after reject | test (text) | ≥ 97% |
| Overall accuracy | `test_audio_synth` | ≥ 93% |
| Macro F1 | test (text) | ≥ 0.96 |
| Worst single label | test (text) | ≥ 90% (≥ 18/20) |
| Command pass rate | operating point set on dev | 98% |
| Chatter let through at that point | test `unhandled` (handwritten + public) | ≤ 3% text, ≤ 5% synthetic audio |
| Wrong command accepted | test (text) | ≤ 1% of commands |
| Which-one pairs: wrong side, each pair | test pair slice | ≤ 2%; zero confident swaps |
| Slot resolved exact, known names | test | ≥ 97% |
| Slot span F1, unseen names | test | ≥ 90% |
| Resolver top-3 recall | test | ≥ 99% |
| `ask` rate on slot commands | test | ≤ 10% |
| ECE after temperature scaling | test | ≤ 0.03 |
| Behavioural | suite | MFT ≤ 2% fail per test; INV ≤ 3% flips; DIR ≥ 97% |
| int8 vs fp32 | val | ≥ 99.5% same top-1; accuracy drop ≤ 0.3 points |
| Latency, our model | target device | §7.3 |

- When real audio exists: the same targets on `test_audio_real`, with the threshold re-set on `dev_audio_real`. Synthetic results alone don't prove "done" for audio.

### 6.4 How often

| When | What runs |
|---|---|
| Every data change | all gates in §4 |
| Every training run | val metrics, slices, block confusion, behavioural suite, calibration |
| Every export | int8 vs fp32 agreement, latency |
| Milestones only (≤ 4 reads in total) | sealed test + `test_audio_synth`. M1 after Step B, M2 after distillation + calibration, M3 before shipping, M4 on real audio. Each read is logged with date and model hash. |

- If a test read changes a decision, the test is no longer clean. Write a fresh test v2 (new writers, new names) before the final claim 👤.

---

## 7. Shipping

### 7.1 Export

- **What:** export the full model (encoder + intent head + B/I/O head) to ONNX, then graph-optimise.
- **Tool:**
  - If only the encoder stays a SentenceTransformer: `SentenceTransformer(..., backend="onnx")`, then `export_optimized_onnx_model(model, "O2", path)`.
  - For the two-head model: `torch.onnx.export` (or Optimum) + ONNX Runtime graph optimisation.
  - Optional experiment on Apple hardware: Core ML execution provider.
- **Check:** ONNX outputs match PyTorch within 1e-4 on 200 val rows; same top-1 on all.

### 7.2 Quantisation

- **What:** dynamic int8 for CPU. `export_dynamic_quantized_onnx_model(model, "arm64", path)` for the SentenceTransformer path, or ONNX Runtime `quantize_dynamic` for the two-head model.
- **Why:** smaller, faster on CPU, closest to edge.
- **Static int8** (`export_static_quantized_openvino_model`) targets Intel/OpenVINO; skip unless the target device is x86.
- **Check:** §6.3 int8 rows; pair slice and calibration rechecked after quantisation (re-fit temperature if ECE rises).

### 7.3 Latency budget (p95, from end of speech)

Re-baselined on 2026-11-22 on a MacBook Air M5 (CPU), with ~2 s TTS clips. The team freezes it once it's agreed.

| Stage | Budget | Measured p50 / p95 |
|---|---|---|
| Whisper `base.en` **float32**, beam 5, n-best 3 | ≤ 700 ms | 420 / 850 ms (noisy, n=30) |
| *(ref)* Whisper `base.en` int8, beam 5 | – | 650 / 690–1,080 ms; int8 is slower than float32 on Apple CPU |
| *(ref)* Whisper `base.en` float32, greedy (no n-best) | – | 330 / 360 ms |
| Normalize + tokenize | ≤ 1 ms | – |
| MiniLM int8 ONNX, batch of 3, ≤ 32 tokens, 1 thread | ≤ 5 ms | 2.3 / 3.4 ms (batch 1: 1.0 / 1.6 ms) |
| Calibration + reject + resolver | ≤ 2 ms | – |
| **Our NLP total** | **≤ 10 ms** | – |
| Link: audio up + result back, cable | ≤ 30 ms | to measure |
| Link: audio up + result back, WiFi (weak) | ≤ 200 ms | to measure |
| **End to end, end of speech → HUD action, cable** | **≤ 750 ms** | – |
| **End to end, end of speech → HUD action, WiFi** | **≤ 950 ms** | – |
| Cold start (model load) | ≤ 2 s, at app start, not per command | Whisper load 0.6 s |

- Asking Whisper for 3 transcripts costs about the same as 1, since beam search already runs.
- ONNX Runtime was faster with 1 thread than with the default thread count for MiniLM.
- Open choice: greedy Whisper would meet ~400 ms but drops n-best (§5.5).

- **How it is measured:**
  - Compute on the MacBook that runs inference at test week (decided: the MacBook). Re-measure if the test-week Mac differs from the Air used for the baseline.
  - Link on both paths: cable and WiFi at normal and weak signal (far from the router, or behind a wall).
  - 50 warm-up calls, then 1,000 timed calls with `time.perf_counter_ns`, batch 1 (or 3 for n-best), real val sentences, single thread and default threads.
  - Report p50 / p95 / p99 per stage and end to end. Laptop on battery and on power.
  - Script lives with the model and runs on every export; a regression > 20% fails the export.
- **Check:** budget table filled from the latest export; all p95 within budget.

### 7.5 Link between HUD and MacBook (cable or WiFi)

- **What:** the HUD sends the audio of one utterance to the MacBook and gets the JSON result back. The same protocol runs over cable and over WiFi.
- **Why:** decided 2026-11-22. Inference runs on the MacBook, reached by cable or by WiFi, and the WiFi may be weak.
- **Rules:**
  - Send only the speech segment (after VAD or push-to-talk), never a continuous stream.
  - Cable: raw 16 kHz PCM. WiFi: Opus at ~24 kbps, mono, 16 kHz. Training round-trips include this codec (§3.8).
  - Each request carries an id and a capture timestamp. The HUD drops any result older than 2 s, or not matching the latest request. **A late result never fires a command.** This is the same rule as the carry-over guardrail (Siri "SUITS Version – Guardrails").
  - On timeout or disconnect, the HUD shows "not heard, say again" and queues nothing.
  - Results are tiny JSON. Only audio upload size depends on the link.
- **Tool:** `ffmpeg`/`libopus` for the codec; `time.perf_counter_ns` on both ends for the link timing (clocks synced, or measured round trip ÷ 2).
- **Check:**
  - Link latency in §7.3 within budget on both paths.
  - Accuracy on `dev_audio_synth` WiFi-codec clips is within 1 point of raw clips.
  - Scripted test: delayed, dropped and out-of-order results never fire a command (0 failures in 100 runs).

### 7.4 Package

- Ship: ONNX int8 model, tokenizer files, normalizer, temperature, `MIN_CONF`, reject model weights, slot catalogue, label list, model card (data version, split manifest hash, metrics, test read log).
- **Check:** a clean environment loads the package and reproduces val metrics within 0.1 point.

---

## 8. Environments and compute

| Env | Python | Purpose |
|---|---|---|
| main (`.venv`) | 3.15 | data gates, training. torch 2.15, transformers 5.24, sentence-transformers 6.3, datasketch (editable from `external/`), rapidfuzz, Metaphone, ctranslate2, datasets |
| runtime (`.venv-runtime`) | 3.12 | faster-whisper, onnxruntime, ONNX export, latency. Same torch/transformers/ST versions as main. Needed because onnxruntime has no 3.15 wheel and faster-whisper's PyAV won't build on 3.15. |
| checklist (`.venv-checklist`) | 3.11 | CheckList build and scoring; exchange via files (§4.7). Linked to `external/checklist` with a .pth file; needs `ipython<8.17` and spaCy `en_core_web_sm`. |
| Ollama | – | local LLM label checker: `qwen3:8b`. Start with `ollama serve`. |
| mergekit extra | main | only if weight soup through `SentenceTransformer.merge` is used |

- **Export note:** sentence-transformers' `backend="onnx"` helpers need `optimum-onnx`, which forces transformers 4.x. Use `torch.onnx.export(dynamo=True)` + ONNX Runtime `quantize_dynamic` instead (§7.1–7.2). Tested: fp32 ONNX matches PyTorch to 6e-7; int8 is 23 MB vs 91 MB.
- Training the student on the Air (MPS) is fine. Teachers (large encoders) are slower; run them overnight, small batch, or with gradient accumulation.
- Everything is seeded; every run saves config, data manifest hash and metrics.

---

## 9. Open questions (team decisions)

1. ✅ **Unqualified requests:** primary / total. Mapping table in §2.2. *(Decided 2026-11-22.)*
2. ✅ **Inference device:** the MacBook, reached by cable or over WiFi (WiFi may be weak); support both. See §7.5. *(Decided 2026-11-22.)*
3. ✅ **`Add_waypoint` with a new name:** verbatim, no HUD confirmation. *(Decided 2026-11-22.)*
4. ✅ **Cancel / negation:** `unhandled`. *(Decided 2026-11-22.)*
5. **Slot catalogue source:** POI names arrive with NASA coordinates at test week. Who maintains the list, and do we retrain if names change? (The resolver handles new names without retraining; the tagger should too.)
6. **Wake word / push-to-talk:** decides how much chatter reaches the model, and so how strict `MIN_CONF` must be.
7. **Carry-over rule (§5.7):** ship it, or keep it off for test week?
8. **Who writes test:** pick the writers for writer-held-out test rows and test v2.
9. **Real recordings:** when they resume, who records, consent, and where audio is stored.

---

## 10. Checklist

### Setup
- [x] Confirm package wheels on Python 3.15; create a fallback 3.12 env only if needed. *(2026-11-22: onnxruntime and faster-whisper don't install on 3.15, so `.venv-runtime` (3.12) was created. See §8.)*
- [x] Create the `checklist` env (3.10/3.11) and test a round trip through `to_raw_file` / `run_from_file`. *(2026-11-22: `.venv-checklist` (3.11); the round trip works and reports failures.)*
- [x] Install Ollama and one 7–9B instruct model; test it on 10 rows. *(2026-11-22: qwen3:8b; 8/10 agreement, ~0.6 s per row. Missed a value read-back and a Complete_task phrasing.)*
- [x] Pin the ruler model (`all-mpnet-base-v2`) revision. *(2026-11-22: see §4.4; the student revision is pinned too.)*
- [x] Measure baseline Whisper `base.en` latency on the Air; adjust §7.3 once. *(2026-11-22: MacBook Air M5. Results in §7.3.)*
- [x] Measure the same baseline on the target device. *(2026-11-22: the target is the MacBook, so the Air baseline applies. Re-run if the test-week Mac differs.)*
- [ ] Measure link latency for cable and WiFi (normal and weak signal); fill the link rows in §7.3.

### Guide, catalogue, pilot
- [x] 👤 Decide open questions 1, 3, 4. *(2026-11-22: primary / total; verbatim; `unhandled`. Recorded in §2.2 and §9.)*
- [x] Draft the labelling guide (§2.2). *(2026-11-22: [LABELING_GUIDE.md](LABELING_GUIDE.md). Decided rules in; 9 proposed rules (P1–P9) need team OK.)*
- [ ] 👤 Review the guide draft and settle proposed rules P1–P9.
- [ ] 👤 Build the slot catalogue and split names into train names and held-out names (§3.5).
- [ ] 👤 Pilot: write 20 rows for 6 labels (§3.3).
- [ ] Run every gate on the pilot; 👤 blind double-label 100 pilot rows (agreement ≥ 95%).
- [ ] Freeze gate thresholds and the guide.

### Writing
- [ ] 👤 Write 100 rows per command label, covering the variety axes (§3.4), with ≥ 3 writers per label.
- [ ] 👤 Write which-one pair rows, keyword traps, jargon, filler and Whisper-style rows (§3.5).
- [ ] 👤 Write slot rows using train names and held-out names.
- [ ] 👤 Write 350 `unhandled` rows across the five handwritten sources (§3.6).
- [ ] Run the coverage report per label; 👤 fill gaps.

### Gates before split
- [ ] Normalizer written once and unit tested (§4.1).
- [ ] Exact and near-duplicate check (§4.2); 👤 rewrite flagged rows.
- [ ] Label-error check with kNN vote and Ollama (§4.5); 👤 review queue.
- [ ] Diversity gate on all handwritten rows (§4.4); 👤 rewrite labels that fail.

### Split
- [ ] Build phrasing groups (MinHash + ruler cosine) and cross-check with `community_detection`.
- [ ] Assign groups 60/20/20 by label, with held-out names forced out of train and writer hold-out in test.
- [ ] Freeze and commit the split manifest and its hash.
- [ ] Run the leakage gate (§4.3); 👤 replace any leaked test rows.

### Generators (after split)
- [ ] Record noise clips and set up TTS voices.
- [ ] Run TTS → noise → Whisper round-trips with shipped Whisper settings; save n-best.
- [ ] Add the WiFi codec path (Opus ~24 kbps, 0–5% packet loss) to half the round-trips (§3.8, §7.5).
- [ ] Drop variants that lost the discriminator or slot; enforce caps; 👤 spot-check 10%.
- [ ] Build the Whisper confusion phrasebook from round-trip diffs; 👤 share it with writers.
- [ ] Generate Whisper silence/noise hallucinations (≤ 50).
- [ ] Download CLINC150 `oos` (train/val) and a second public set (test only); filter by ruler cosine; 👤 check 10%.
- [ ] Build `dev_audio_synth` and `test_audio_synth`; seal test sets.
- [ ] Re-run all gates on the full dataset (dedup, leakage, diversity, balance, descendants).

### Behavioural suite
- [ ] 👤 Write ~100 keyword-trap MFT cases, Appendix A callouts and which-one minimal pairs.
- [ ] Build INV and DIR tests with `Perturb` and custom functions (§4.7).

### Model
- [ ] Step A: frozen MiniLM + logistic regression floor on val.
- [ ] Step B: fine-tune MiniLM with intent + B/I/O heads, 5 seeds.
- [ ] Step C (experiment): contrastive warm-up; keep only if the pair slice improves.
- [ ] 📍 M1: first sealed test read; log it.
- [ ] Fine-tune 3 teacher encoders; combine with geometric mean; confirm they beat the student on val.
- [ ] Distil into MiniLM; tune α and T on val.
- [ ] Weight soup of 3–5 students; keep if not worse.
- [ ] Temperature scaling on val.
- [ ] Fit the reject model on `dev_audio_synth`; set `MIN_CONF` at 98% command pass.
- [ ] Wire up n-best; A/B against 1-best on `dev_audio_synth`.
- [ ] Wire up the slot resolver (rapidfuzz + Double Metaphone); tune its thresholds on val.
- [ ] (Optional) carry-over rule with guardrails and its scripted test.

### Data engine (up to 3 rounds)
- [ ] Collect val and dev-audio errors; cluster into writing briefs.
- [ ] Mine public pool around false accepts; 👤 review.
- [ ] 👤 Write ≤ 30 new train rows per label per round from the briefs.
- [ ] Re-run all gates, retrain, compare slices.

### Evaluation
- [ ] Every run: val metrics, slices, block confusion, behavioural suite, calibration.
- [ ] 📍 M2: sealed test read after distillation + calibration; log it.
- [ ] 👤 If any test read changed a decision, write test v2 with new writers and names.

### Shipping
- [ ] Export to ONNX; check against PyTorch.
- [ ] Quantise to int8; check top-1 agreement, accuracy, pair slice and ECE.
- [ ] Run the latency script on the target device; fill the §7.3 table.
- [ ] Build the HUD↔MacBook link (cable + WiFi) with request ids, the 2 s stale-result drop and "say again" on timeout (§7.5).
- [ ] Run the scripted delayed / dropped / out-of-order test: 0 commands fired from late results.
- [ ] Compare accuracy on WiFi-codec clips vs raw clips on `dev_audio_synth` (within 1 point).
- [ ] Package model, normalizer, temperature, `MIN_CONF`, reject model, catalogue, labels and model card.
- [ ] Load the package in a clean env and reproduce val metrics.
- [ ] 📍 M3: final sealed test read (text + synthetic audio) against §6.3; log it.

### Later: real audio
- [ ] 👤 Record team speakers on the headset with fan noise for val/test prompts and chatter, over both cable and WiFi.
- [ ] Build `dev_audio_real` and `test_audio_real` (held out by speaker).
- [ ] Re-fit reject model and `MIN_CONF` on `dev_audio_real`; re-check n-best.
- [ ] Add new Whisper errors to the phrasebook; one data-engine round on real errors (train only).
- [ ] 📍 M4: `test_audio_real` read against §6.3; latency re-check end to end with the headset.

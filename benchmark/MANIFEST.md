# EVA intent benchmark manifest

**Status: accepted for training and benchmark comparison.** The complete dataset has 3,600 rows across 36 intents. All listed mechanical, label-review, split, and validation gates pass.

## Dataset and split

| Intent | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| `get_oxy_pri_storage` | 80 | 10 | 10 | 100 |
| `get_oxy_sec_storage` | 80 | 10 | 10 | 100 |
| `get_oxy_pri_pressure` | 80 | 10 | 10 | 100 |
| `get_oxy_sec_pressure` | 80 | 10 | 10 | 100 |
| `get_suit_pressure_oxy` | 80 | 10 | 10 | 100 |
| `get_suit_pressure_co2` | 80 | 10 | 10 | 100 |
| `get_suit_pressure_other` | 80 | 10 | 10 | 100 |
| `get_suit_pressure_total` | 80 | 10 | 10 | 100 |
| `get_helmet_pressure_co2` | 80 | 10 | 10 | 100 |
| `get_fan_pri_rpm` | 80 | 10 | 10 | 100 |
| `get_fan_sec_rpm` | 80 | 10 | 10 | 100 |
| `get_scrubber_a_co2_storage` | 80 | 10 | 10 | 100 |
| `get_scrubber_b_co2_storage` | 80 | 10 | 10 | 100 |
| `get_temperature` | 80 | 10 | 10 | 100 |
| `get_coolant_storage` | 80 | 10 | 10 | 100 |
| `get_coolant_gas_pressure` | 80 | 10 | 10 | 100 |
| `get_coolant_liquid_pressure` | 80 | 10 | 10 | 100 |
| `get_heart_rate` | 80 | 10 | 10 | 100 |
| `get_oxy_consumption` | 80 | 10 | 10 | 100 |
| `get_co2_production` | 80 | 10 | 10 | 100 |
| `get_eva_elapsed_time` | 80 | 10 | 10 | 100 |
| `get_primary_battery_level` | 80 | 10 | 10 | 100 |
| `get_secondary_battery_level` | 80 | 10 | 10 | 100 |
| `get_warnings` | 80 | 10 | 10 | 100 |
| `open_menu_vitals` | 80 | 10 | 10 | 100 |
| `open_menu_navigation` | 80 | 10 | 10 | 100 |
| `open_menu_tasks` | 80 | 10 | 10 | 100 |
| `open_menu_uia` | 80 | 10 | 10 | 100 |
| `close_menu` | 80 | 10 | 10 | 100 |
| `Get_coordinates` | 80 | 10 | 10 | 100 |
| `Set_navigation_target` | 80 | 10 | 10 | 100 |
| `reroute_navigation` | 80 | 10 | 10 | 100 |
| `Add_waypoint` | 80 | 10 | 10 | 100 |
| `Delete_waypoint` | 80 | 10 | 10 | 100 |
| `Complete_task` | 80 | 10 | 10 | 100 |
| `unhandled` | 80 | 10 | 10 | 100 |

**Totals:** 2,880 train, 360 validation, 360 test, 3,600 overall. The split is stratified 80/10/10 with seed 42. JSONL files are in `data/`.


## Validation revision

The validation set was revised to use more varied phrasing across all 36 intents. The 350 non-`Complete_task` rows were replaced; the 10 `Complete_task` rows were retained from the focused validation edits. Training and test rows were unchanged. The updated set passes the full split audit with 0 gates: style probe 7.06%, content probe 92.75%, and no train-to-held-out same-label similarity at Jaccard ≥0.65. See `review/reports/validation-revision-audit.txt`.

The BGE run and charts under `bge-intent/runs/curves/` were regenerated against this revised validation set (6 epochs, best epoch 6). Validation accuracy/macro-F1 are 89.44%/88.95%; held-out test accuracy/macro-F1 are 99.44%/99.44%. The original model artifacts under `bge-intent/runs/` were not changed.

## Final audit results

- Schema, labels, row counts, exact duplicates, near-duplicates, opening diversity, length buckets, mean length, style balance, and slot/name constraints: **pass; 0 gates**.
- Every intent has all four word-count buckets (1–3, 4–7, 8–12, and 13+ words); maximum two-word opening frequency is at most 8.
- 97 slot names are used across at least two name intents; no name appears more than three times in one intent. `review/name-plan.json` matches the final assignments.
- Style-only five-fold probe: **7.14%** (limit <15%). Maximum per-intent difference from global rates: question 14.78 points, filler 6.33, punctuation/lowercase 8.47, ASR 14.22.
- TF-IDF/logistic five-fold content probe: **93.47%** (limit <99%). Highest errors remain scrubber A/B (111 across both directions) and add/delete waypoint (22). All rows in those two pairs received targeted blind review.
- Final train-to-held-out similarity audit: **pass; 0 rows at Jaccard ≥0.65**. Split counts are 80/10/10 per intent.
- `bge-intent/validate.py data/all.jsonl`: **3,600 rows, 0 problems**.

## Independent label review

Each 25-row generation round received a 180-row blind sample (five per intent): round 1 **179/180 (99.44%)**; rounds 2, 3, and 4 **180/180** each. The round-one disagreement was rewritten to name helmet location and confirmed by a second reviewer. Round three had seven panel near-duplicate replacements; round four had six. The final 180-row audit scored **180/180**.

All 400 final scrubber A/B and add/delete waypoint examples were blind-reviewed. One terse item, “Delete Alpha 3,” was ambiguous; it was replaced with “Delete waypoint Cabeus” and independently confirmed as `Delete_waypoint`. The final blind check of 27 rows changed during aggregate-style and leakage repairs scored **27/27**.

The deterministic split initially identified 26 held-out rows with same-label training similarity ≥0.65. All 26 were rewritten in source round files; after one additional targeted rewrite, the regenerated split audit passed with zero leakage failures.

## Rejection history

Whole generation attempts totaling **1,500 rows** were rejected below the 97% blind-label threshold and archived under `review/rejected_round1_*`. The blind scores and reasons are recorded in `review/rejection-log.md`. Accepted-round mechanical repairs and final aggregate repairs are also recorded there.

## Source and audit artifacts

- Dataset specification: `SPEC.md`
- Round review and rejection record: `review/rejection-log.md`
- Final aggregate audit: `review/reports/final-audit-repaired.txt`
- Final split audit: `review/reports/split-audit.txt`
- Requested `claws_intent_onboarding.tex` was absent from the workspace; confusion guidance was inferred from `bge-intent/intents.json` and documented in `SPEC.md`.

## SHA-256

- `data/all.jsonl`: `cb0d9ca4395327c8190815933a480f6dcd21b3abddfe42dcdc144256261377f7`
- `data/train.jsonl`: `2ced91e49759758f1b484f3bd14ddb737b3b93748ca976404bdbd0f7bf032617`
- `data/val.jsonl`: `75116fbae3a2f8545b22de778658b63197a4592628974d233b72fa7b6c7389b8`
- `data/test.jsonl`: `397037a2a5fc4cd1accef69cf20f03457672aed95a091f56ca19418d5e4ef050`

## BGE retraining

The classifier was retrained from the accepted `train.jsonl` with explicit validation and test files using `BAAI/bge-small-en-v1.5`, 6 requested epochs, batch size 16, learning rate 2e-5, and seed 42. Best epoch: 6.

- Validation: accuracy **100.00%**, macro-F1 **100.00%**.
- Held-out test: accuracy **99.44%**, macro-F1 **99.44%** (358/360 correct).
- The two errors were between `Add_waypoint` and `Set_navigation_target`; `Add_waypoint` recall was 80%, and `Set_navigation_target` precision was 83%.
- Artifacts: `bge-intent/runs/model.pt`, `bge-intent/runs/label2id.json`, `bge-intent/runs/metrics.json`.
- Model artifact SHA-256: `761e6c52e33acbab080b4f099afca616dc4a0ce24eaf676e5ab667d46b5b29ea`.

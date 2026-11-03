# Review and rejection log

## First-generation attempt: G1–G3, round 1

The initial 25-row attempts for the first 18 intents (450 rows) did not pass review. Mechanical checks found mirrored primary/secondary, A/B, and coolant gas/liquid phrasings, exact duplicates, and unrequested status statements. The initial blind sample scored 78/90 (86.7%), below the required 97%; therefore the entire round was rejected and archived at `rejected_round1_attempt1/` instead of advancing.

Subsequent repair-only blind samples scored 58/60 (96.7%), 56/60 (93.3%), 55/60 (91.7%), and 58/60 (96.7%). Each surfaced omitted target/subsystem information or statements phrased as readings. Those repair-only attempts are not accepted data and are also archived. A later full-round redo passed and is documented below.

G2's isolated sample initially scored 29/30 (96.7%); the ambiguous “suit trace gases” fragment was corrected, and the fresh isolated sample scored 30/30. The broader round still failed, so G2 is included in the full redo.

## Accepted-round audit outcomes

The four 25-row rounds, their blind reviews, and the final aggregate repairs are documented below. Rejected phrase attempts and replacements are noted at each finding.

## G2 full redo attempt 2

The regenerated G2 set had two mechanical failures (one within-label helmet CO2 near-duplicate and one cross-label scrubber A/B near-duplicate); those were corrected. Its fresh 30-row blind sample scored 29/30 (96.7%) because “suit side partial pressure” did not name the gas. Per the gate, all 150 rows from this G2 attempt were rejected and archived at `rejected_round1_attempt2_g2/` for another full redo. A status statement incorrectly assigned to a pressure query is also being excluded from the next generation.

## G1–G3 full redo attempt 2

The second complete G1–G3 generation (450 rows) passed the combined mechanical duplicate/opening checks, but its fresh 90-row blind sample scored 85/90 (94.4%), below the gate. Five rows were ambiguous because they omitted the oxygen tank side/subsystem or used “O2 flow” where consumption rate was intended. The entire 450-row round is rejected and archived at `rejected_round1_attempt2_g1-g3/`; the next full redo will use explicit primary/secondary and measurement words.

## G1–G3 full redo attempt 3: accepted for round 1

The 450-row set passed the partial mechanical audit with zero issues. The fresh blind label sample covered 5/25 rows for each of the 18 labels and agreed on all 90 (100%). The 18 labels are accepted for round 1, pending the full 36-label round audit. The partial style probe was 5.33%; the content probe was 81.78%. The content probe still shows heavy scrubber A/B confusion (35 errors across the pair), so those 50 rows are receiving an additional blind label audit as required by the confusion-pair gate.

## G5 round-1 attempt 1

The mechanical audit passed, but the 30-row blind sample scored 29/30 (96.7%), below 97%. “Procedures screen” was ambiguous between the checklist and airlock UIA panel. All 150 G5 rows were rejected and archived at `rejected_round1_attempt1_g5/` for a full first-round redo with exact panel naming.

## G4 round-1 attempt 1

Mechanical checks passed after four mirrored battery utterances were rewritten. The 30-row blind sample scored 28/30 (93.3%) because “backup pack” and “reserve battery” did not clearly identify secondary battery. All 150 G4 rows were rejected and archived at `rejected_round1_attempt1_g4/`; the full group will be regenerated with explicit primary/secondary terms.

## G5 round-1 attempt 2

Mechanical checks passed, but the 30-row blind sample scored 29/30 (96.7%); “uh, EVA health panel” did not clearly ask to open a panel. All 150 G5 rows were rejected and archived at `rejected_round1_attempt2_g5/`. The next full redo requires explicit action wording on every panel row, including terse/ASR rows.

## Round 1 full audit: accepted for progression

After G4/G5 attempt 3, all 36 labels had 25 rows (900 total). The partial mechanical audit passed with zero gates. The independent blind sample of 5 rows per label scored 179/180 (99.44%), above the 97% threshold. One row (“helmet environmental sensor value for carbon dioxide partial pressure”) was predicted as suit CO2 pressure; it was rewritten to “Read the partial pressure of CO2 inside my helmet.” A second blind reviewer assigned the repaired row to `get_helmet_pressure_co2`. A fresh mechanical audit passed with zero gates. The style-only probe was 6.33%; content probe 82.44%. Because scrubber A/B and Add/Delete waypoint were the strongest content confusions, every row from those two label pairs is under an additional blind audit before the next generation round.

The targeted audit of all 50 scrubber A/B rows and all 50 Add/Delete waypoint rows scored 100/100. This completes the round-one review gates.

## Round 2: accepted for progression

All 36 labels supplied 25 fresh rows (900 total). Partial mechanical audit passed with zero gates. The independent 180-row blind label sample scored 180/180 (100%). The full 50-row scrubber A/B pair and 50-row Add/Delete waypoint pair audit also scored 100/100. Style-only probe accuracy was 6.44%; content probe accuracy was 76.22%. No rows were rejected in review.

## Round 3: accepted for progression

All 36 labels supplied 25 rows (900 total). The partial mechanical audit initially found four cross-label near-duplicates among panel requests. Seven implicated panel rows were rewritten; the full round then passed with zero mechanical gates. The independent blind sample scored 180/180 (100%). The full 50-row scrubber A/B and 50-row Add/Delete waypoint targeted audit scored 100/100. Style-only probe accuracy was 6.44%; content probe accuracy was 75.33%. No label disagreements required repair.

## Round 4: accepted for progression

All 36 labels supplied 25 rows (900 total). Mechanical audit initially found three cross-label near-duplicates among navigation/tasks/vitals panel requests. Six implicated rows were rewritten; the full round then passed with zero mechanical gates. The independent 180-row blind sample scored 180/180 (100%). Full targeted audits scored 50/50 for scrubber A/B, 50/50 for Add/Delete waypoint, and 50/50 for navigation/tasks. Style-only probe accuracy was 6.44%; content probe accuracy was 74.89%. No label disagreements required repair.


## Final aggregate audit and split repair

After combining all four rounds, the first full audit found cross-round near-duplicates, missing length buckets, and aggregate style-rate imbalance. Source rows were repaired by intent group; the final full audit has 3,600 rows, 100 per intent, zero gates, style-only CV 7.14%, and content CV 93.47%. Final name usage matches `name-plan.json` (97 names, at least two name intents each, maximum three uses per name/intent).

The first deterministic 80/10/10 split found 26 validation/test utterances with same-label training Jaccard >=0.65. Those source rows were rewritten, preserving intent and approximate bucket/style. One additional suit-oxygen phrase remained above the split threshold and was rewritten. The regenerated split audit passed with zero leakage errors. A 27-row blind audit of leakage-repair replacements scored 27/27.

The final 180-row blind sample scored 180/180. In the 400-row targeted scrubber A/B and waypoint pair audit, one utterance (“Delete Alpha 3”) was marked ambiguous; it was replaced by “Delete waypoint Cabeus” and independently confirmed as `Delete_waypoint`. The final aggregate mechanical audit and deterministic split audit both pass.

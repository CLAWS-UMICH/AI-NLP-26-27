# Iterative agent loop for SUITS intent data generation

The LLM agent only **proposes** rows. Code and a second, blind model decide what gets kept, and each round's evaluation failures decide what the agent writes next.

```
Generate → Filter → Pass through ASR → Dedup → Train & evaluate → Target → repeat
```

## 0. Freeze the test set first (once, by hand)

- Write about 8–10 rows per intent yourself in radio style. Use slot names that never appear anywhere else.
- Then lock that file. The agent never sees it, and it isn't used to choose what to generate next.
- For the loop's decisions, use a separate **dev set**: about 300 rows, half hand-written and half from round 0. If the loop steers on test, you just end up tuning to test.

## 1. A spec the agent writes from

For each intent, write a short card:

- what the intent means
- its deciding word plus the forms that still count ("secondary", "sec", "backup", "the other tank")
- its sibling intents
- 3 positive rows and 3 "looks like it but isn't" rows

The cards are what keep the generator from blurring siblings together.

## 2. Generate by filling a grid, not by asking for "diverse" rows

Define a shape grid. Each cell is one combination:

| Dimension | Values |
|---|---|
| Opener | none / okay / roger / houston |
| Form | question / command / clipped noun phrase / "need X" |
| Where the deciding word sits | early / late / abbreviated |
| Disfluency | none / filler / false start / repeat |
| Length | 1–3 / 4–7 / 8+ words |

Each request to the agent names one cell and one sibling family. The agent returns an **aligned set** for that cell:

- the same sentence for every sibling ("coolant **gas** pressure?" and "coolant **liquid** pressure?")
- plus a report version labelled `unhandled` ("coolant gas pressure's looking good")

The whole set shares one `group` id, so a split never separates the pair. Coverage is tracked per cell, and the agent is always sent to the emptiest cells. That forces diversity in a way a prompt can't.

Slot names come from a separate generator. They go into a train pool and a test pool that never overlap.

## 3. Filter (code + a blind second model)

**Rule checks:**

- the slot appears word for word in the text
- every command row has exactly one deciding word (catches "primary… secondary" mixes)
- `unhandled` rows contain no request verbs
- everything is lowercase

**Blind relabel:** a different model, or the same model in a fresh call without the intended label, reads INTENTS.md and labels the row. Keep the row only if both labels agree. Disagreements go to a small queue you skim; that's also where ambiguous rows get dropped. In practice this step catches most bad rows.

## 4. Pass rows through ASR to get real Whisper errors

1. Text → TTS with several voices (Piper or Kokoro, both run offline).
2. Radio filter: 300–3400 Hz bandpass, noise, light clipping.
3. **The same Whisper model you'll use on the suit.** Each output keeps its row's label.

This gives you real corruptions like "second dairy" and "scrubber eh" instead of guessed ones. The first run also yields a confusion table that you can apply to text cheaply in later rounds.

## 5. Dedup gate

Uses sentence-transformers `util.semantic_search` and datasketch MinHash.

- Reject any row too close to existing train or dev rows: MinHash Jaccard ≥ 0.8, or embedding cosine ≥ 0.9 after removing filler words (okay / roger / houston / uh / give me / read / check).
- Never compare rows with the frozen test set. Reading the test to check for leakage leaks it; leakage is prevented by keeping test rows and test slot names out of the loop instead.

## 6. Train, then evaluate on dev and the CheckList suite

Produce one report covering:

- the confusion matrix by family (oxygen, scrubbers, coolant, nav…)
- slot F1 on unseen names
- CheckList results per test type: word-swap tests, ASR-noise tests, report → `unhandled` tests

## 7. Target the next round

The agent reads the report and decides how to spend the next round's budget:

- **Most of it on the worst confusion pairs.** For example, if scrubber a/b is the weakest family, generate there.
- **Some on rows the model got wrong or barely got right.** The agent writes close variants of them (hard-example mining).
- **The rest on CheckList test types that still fail**, plus any grid cells that are still empty.

**Stop** when dev accuracy per family stops improving for 2 rounds. Then score the frozen test set once.

## Who does what

| Step | Done by |
|---|---|
| Generate aligned sets, write variants of misclassified rows, read report and allocate budget | LLM agent |
| Grid bookkeeping, rule checks, dedup, ASR pass, training, CheckList | Plain Python |
| Blind relabel | Separate LLM call |
| Test set, intent cards, review queue | You (about an hour per round at most) |

**Budget:** round 0 is about 40 rows per intent. After that, each round adds about 800–1,000 targeted rows. Expect 3–5 rounds.

## Two rules that matter most

1. **Keep the test set frozen and out of the loop**, or the numbers stop meaning anything.
2. **Keep the blind relabel**, or the agent's mistakes go into training as truth.
# How astronauts actually talk (style reference)

Use this when **hand-writing `test.jsonl`**. Do not train on transcripts: they are 1971 vocabulary (LM, PLSS flags, Rover nav), full of names and geology chatter, and have none of our intents.

Source: *Apollo 15 Technical Air-to-Ground Voice Transcription* (NASA Mission Transcript Collection, `as15-tec.pdf`), EVA-1, PDF pages ~460–600, GET 05 01 45 – 05 06 11. Linked from https://www.nasa.gov/history/johnson-history-resources/.

## What the real EVA radio looks like

| GET | Speaker | Line |
|---|---|---|
| 05 04 02 37 | CC | "Okay, and while you're brushing it, Dave, we need a PLSS check." |
| 05 04 02 42 | CDR | "Yes. Roger. I got - flags are clear. I've got 40 percent and 3.85." |
| 05 04 02 50 | LMP | "Okay, I'm - I'm 385; my pressure flags are clear, and I'm reading 45 percent." |
| 05 01 57 06 | CC | "And Rover, this is Houston. Your range to Rhysling is about 1.7; so you may be short of that still." |
| 05 01 57 14 | LMP | "It just came up 1.7, and our relative bearing's 036." |
| 05 02 32 56 | CC | "Roger, Dave. Copy. Your updated range at station 2 is about 3.9 clicks. And if you'll park down-Sun, we'll give you a NAV update…" |
| 05 05 02 09 | CC | "- - could you verify all your flags are off?" → LMP: "Verified." |
| 05 02 52 44 | CC | "- - press on with your comprehensive sample, please." |
| 05 05 04 22 | CDR | "Okay, Joe. I'll stand by for your suggestions." |

Rough counts in those ~140 pages: *okay* 637, *roger* 181, *Houston* 66, *go ahead* 34, *copy* 24, *stand by* 20, *say again* 8, *verify* 5, *go for* 1, *wilco* 0.

## Patterns to copy into test sentences

- **Open with an acknowledgement or a name, then the request:** "Okay, Houston, …", "Roger. And …", "Yes. Roger. I got …". So test commands should often start with "okay", "roger", or a callsign.
- **"Okay" and "roger" are far more common than "copy"**, and nobody says "wilco". Don't overuse movie radio-speak.
- **Self-corrections and false starts:** "I'm - I'm 385", "I got - flags are clear". Whisper will keep some of these.
- **Bare numbers without units:** "40 percent and 3.85", "bearing's 036", "about 3.9 clicks".
- **Requests are clipped:** "we need a PLSS check", "verify all your flags are off", "range to Rhysling".
- **Lots of non-commands:** geology description, "stand by", "say again?", "press on". Good `unhandled` test rows that share keywords with real intents ("pressure flags are clear" is a report, not a question).

The radio terms seen here (copy, roger, stand by, say again, verify, press on, PLSS check, flags, clicks, NAV update, down-sun) are defined in `suits_glossary.json`, which is what feeds the primer.

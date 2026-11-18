# E5 v3 — next version of the knob game (design in progress, paused 2026-11-17)

Status: **paused by Sam on 2026-11-17** before the final choices were made. Nothing here has run. Before any run, the next agent must show Sam the full spec: every string that goes into the model, rendered from the code. That means the system prompt in Qwen's raw chat format with the tool block, the tool names and descriptions, every tool result, every harness message, the end-of-session question and the scoring questions.

(It's "E5 v3", not "E6": E6 is already the audit pilot in WIRK.)

## Why a v3: what E5 v1/v2 showed (2026-11-16/17)

Data: `runs/live/e5-*`, `runs/e5-*.jsonl` (Hugging Face `pebbleml/knob-ladder-runs`). Analysis: `analysis/`. Summary: `analysis/e5_review/SUMMARY.md`.

- **The model often finds individual concepts but almost never the mechanism.**
  - Rung B: 23 of 32 sessions named at least one knob's concept correctly during the session. Rung C: 17 of 32.
  - Only 1 of 64 closing hypotheses says "each knob puts a topic into my thinking" (C-ep013).
  - It sees the concept and files it as a style or mood ("ocean" became "scale", "honey" became "a richness dial").
  - It settles on a "style console" frame by about turns 3–7, and it backslides.
- **End-of-session ranking among 32 concepts:**

  | Rung | Top-1 | Chance |
  |---|---|---|
  | A, reasoning dropped | 21% | 3% |
  | B | 15% | 3% |
  | C | 16% | 3% |

  The model scores exactly the same as a fresh copy reading its transcript (self = observer). So this is evidence of using its transcript, not of privileged access.
- **The controls get invented effects.** The sham got one in 23 of 33 sessions and the random direction in 23 of 31. But no control was ever given a concrete topic.
- **The turn-lag confound (Sam).** 47–67% of knob turns were called from inside the injected reply. The previous knob's injected reasoning was then pasted in while the next knob was on, and effects were credited one knob late. This caused 3 of C-ep013's 5 wrong answers.
- **The knobs are too weak.** Level 10 = 0.5 × the median residual norm at layer 44. Real spirals were about 10 in roughly 1,600 injected turns, against the old 28 Oct "mushrooms ×744" runs. When a spiral was visible, the model noticed it in 10 of 13 cases but called it "repetition".
- **Thinking was cut short.** The 1,024-token cap was hit in 28–44% of injected turns and 14–39% of turns with no knob on.

## Decided by Sam (2026-11-17)

1. **Thinking on.** Each message's thinking is dropped from later history, as the model was trained (Sam's research).
2. **The model must examine the pasted reasoning with no knob on.**
3. **No knob overlap.** A knob is on only for the model's one message right after the tool result "Knob k set to L.". The model can't turn a knob from inside that message: the harness cuts the message if it starts a tool call. Then the knob is off.
4. **Stronger knobs.** A short sweep first: level 10 is set where the concept reliably spirals in the injected message (target about 90% of messages containing the concept), and level 3 where it is faint. Concepts whose vectors loop on something else (carrot gave "blue/red") are swapped out.
5. **Rungs A (explicit: "injects a concept") and C (blind: told nothing)**, same boards and seeds, in one run. Rung B is dropped: its "in how you think" wording pushed the model toward style explanations.
6. **No user prompt asking it to write something** (no "write a few sentences…", no "write what you observed"). No notes and no "keep a running list" instruction.
7. **Freedom with enough structure,** so it could get there without contamination.

## Open choices (options put to Sam; not yet chosen)

**Where the paste goes:**

1. **In the hand-back** (Nia's pick). After the injected message: `USER: Knob k is now off. For reference, this was your reasoning in your last reply… <reasoning>{thinking while knob k was on}</reasoning>`. The next message has no knob on. Only thinking written while a knob was on gets pasted back. Thinking from turns with no knob on is dropped as in training.
2. **The same, but paste both** the reasoning from before the turn and the reasoning while the knob was on. Less forgetting, but more pasting.
3. **A forced look-back step.** A reflection message with no tool calls allowed, then "Turn a knob or call end_session."
4. **The model asks for it.** A `read_reasoning()` tool, usable only when no knob is on. Most freedom, and whether it seeks the evidence becomes a result. The risk is that it never calls it.

**How much structure:**

- (A) Nothing beyond the guards.
- (B) A floor on ending (Nia's pick): `end_session` works only after N knob turns (e.g. 12). Before that it returns "The session can end after 12 knob turns.". E5 sessions often quit after 4–12 turns.
- (C) One line in the system prompt that level 0 means off.

**Thinking budget (Sam unsure):**

- 1,024 everywhere.
- 2,048 only for turns with no knob on (Nia's pick).
- 4,096 everywhere (runs take roughly 2–3× as long).

**Other defaults proposed, not yet confirmed by Sam:**

- 6 knobs (4 concepts + 1 random + 1 sham).
- Up to 20 knob turns.
- 32 sessions per rung.

## Scoring (to pre-register once the design is fixed)

- **Primary:** for each knob it turned, its one-word end answer, marked exact, close or wrong by a blind judge with the answer key.
- **Secondary:**
  - invented-effect rate on the sham and random knobs;
  - logit rank among the 32 concepts (as in E5);
  - in rung C, whether the closing hypothesis states topic injection.
- **Transcript-reader baseline:** a fresh copy reads the transcript with no injection.
- **Session review:** the per-session Haiku judge plus Sonnet write-up (`scripts/e5_review_workflow.js`).

## Code changes needed (not started)

All in `knobladder/episode.py`, `scripts/r0_pilot.py`, `knobladder/text.py`:
- inject only the message after the tool result;
- cut the message at `<tool_call>` while a knob is on;
- the hand-back message and paste position;
- the optional end floor;
- the board size;
- per-turn thinking budgets;
- a dose-sweep script for the new top level;
- `scripts/e5_render.py` updated to render the full raw prompt for Sam's review.

# Knob ladder

Can a model learn, prefer and own its own activations? Spec and research: WIRK wirkspace
`machine consciousness` (283326b9), item `b6b8e429`. This repo is rung R0 (`ec8dd353`), now at v2.

## R0 design (v2)

A model gets anonymous numbered knobs (`turn(knob)`), reshuffled every episode. Each knob silently
adds a concept vector to the residual stream, but only over the tokens of a fixed neutral passage in
the tool response (the silent window). The passage is identical on every press and the model's own
writing is never steered, so the transcript carries no information about the knobs. Discovery is
knowing what a knob does beyond what an unsteered `observer` re-run of the same tokens knows.

- **Blind framing.** The system prompt says only that there are seven numbered knobs, that a passage
  follows each turn, and to find out what each knob does. Nothing the model sees mentions
  introspection, internal states or activations. There is no sign: a press always adds
  +dose·unit(direction) (`Board.vector` keeps a sign for R1, never shown). Per press: turn call, tool
  response `Knob k turned. / Passage: ...` (steered), "Write one or two sentences about what you
  observed.", the observation, then "Turn a knob, or call end_session when you are done."
- **Board.** 5 concept knobs, 1 random direction and 1 sham (zero vector) on knobs 1-7.
- **Passages.** 24 abstract procedural passages (`PASSAGES_VERSION = "v2"`, ~90-110 tokens) with no
  concrete objects, places, animals, foods, weather, times, money, documents, vehicles, music,
  buildings or materials. The first 8 are for calibration only, the other 16 for episodes.
- **Concepts.** 64 neutral, imageable candidate nouns (`CONCEPTS`) with hand-written `ASSOCIATES`; a
  lexical screen checks no passage or fixed prompt contains any of them. Concept vectors average 5
  prompt templates, mean-subtracted over all candidates within each template.
- **Discovery scoring.** For every knob (pressed or not), two branches off the final transcript, each
  under the cached (steered) and observer runs: free recall ("What does knob k do?", greedy) and a
  multiple choice over the 5 board concepts plus "nothing changed" (letter probabilities).
- **Frozen hyperparameters** (`knobladder/episode.py`, recorded in every row): temperature 0.7, top-p
  0.8, top-k 20 (Qwen3 non-thinking defaults), thinking off, 256 new tokens per generation, 24 decision
  turns per episode, fixed seeds. Ownership trials stay in the code but are off (`--ownership`).

## Gates

- **G0 screen** (`scripts/screen.py`): with no steering, flag (passage, concept) pairs where the
  concept already ranks top-3 on the topic readout after the passage, or where the passage's mean
  residual has a top-5% cosine with the concept vector. Concepts flagged on 2+ passages are dropped;
  16 calibration and 32 knob concepts are drawn from the rest, preferring pairwise cosine < 0.3.
- **G1 calibration** (`scripts/calibrate.py`): calibration concepts and passages only. Silent-window
  topic readout (MRR among the 16, chance 0.211) and passage dNLL per (layer, alpha). Passes if the
  best (layer, alpha) with dNLL <= 1 has an MRR CI above the unsteered baseline's CI.
- **G2 episodes** (`scripts/r0_pilot.py`): free play with the screen's knob concepts at the G1 dose.
- **G3 discovery** (`scripts/analyze.py`): mean over pressed concept knobs of p_cached(correct) -
  p_observer(correct), bootstrap CI over episodes, sign and permutation tests; argmax accuracy against
  chance 1/6; sham false alarms; the same for knobs pressed 2+ times; quotes for the top knobs.

## Layout

- `knobladder/model.py`: `SteeredLM`, with position-indexed interventions. Re-prefilling a schedule
  reproduces the cache that generation built, so every condition is a single forward pass.
- `knobladder/text.py`: raw Qwen3 chat segments and the passages. The chat template re-renders and drops
  think blocks, so the harness never re-renders.
- `knobladder/knobs.py`: concepts, associates and the lexical screen, concept vectors, neuron write
  vectors (R1) and random directions.
- `knobladder/readout.py`: the topic readout shared by G0 and G1.
- `knobladder/episode.py`: the blind framing, free play and discovery scoring.
- `knobladder/ownership.py`, `knobladder/probe.py`: ownership trials and their detection probe (off in R0 v2).
- `knobladder/stats.py`: pure-Python bootstrap and tests, used by `scripts/analyze.py`.
- `tests/`: the invariants. An empty schedule changes nothing; steering touches only its positions;
  generation matches prefill; a neuron write vector equals raising the neuron's activation; nothing the
  model sees names a concept; with no steering, cached and observer scores are identical.

## Running

Run everything remotely on a Lambda GPU box, never on the Mac (`scripts/remote.sh HOST setup` first).
`latest` picks the newest matching file in the box's `runs/`.

```bash
scripts/remote.sh HOST run python3 -m pytest tests -q
scripts/remote.sh HOST run python3 scripts/screen.py --model Qwen/Qwen3-32B --dtype bfloat16
scripts/remote.sh HOST run python3 scripts/calibrate.py --model Qwen/Qwen3-32B --dtype bfloat16 --screen latest
scripts/remote.sh HOST run python3 scripts/r0_pilot.py --model Qwen/Qwen3-32B --dtype bfloat16 --screen latest --calibration latest --episodes 8
scripts/remote.sh HOST pull
python3 scripts/analyze.py runs/r0_Qwen3-32B_*.jsonl
```

Piloting is on Qwen3-4B, then 8B; 32B is the primary. `docs/knob-ladder-board.html` is the experiment
overview (published at https://claude.ai/artifact/Pab6t9nHkHraM47ZE4G2GV); it still describes v1.

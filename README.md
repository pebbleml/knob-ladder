# Knob ladder

Can a model learn, prefer and own its own activations? Spec and research: WIRK wirkspace
`machine consciousness` (283326b9), item `b6b8e429`. This repo is rung R0 (`ec8dd353`), now at v3.

## R0 design (v3)

A model gets anonymous numbered knobs (`turn(knob, level)`, levels 0-10), reshuffled every episode.
Setting a knob adds level/10 of the dose times a concept vector to the residual stream over the
tokens of a fixed neutral passage in the tool response (the silent window). With `--window passage`
the model's own writing is never steered, so the transcript carries no information about the knobs;
`--window observation` keeps steering through the observation it writes next (`--steer-tokens n`
bounds that to n tokens, `--steer-reply n` also steers the first n reply tokens after the think
block). Discovery is knowing what a knob does beyond what an unsteered `observer` re-run of the same
tokens knows.

- **Blind framing.** The system prompt (`episode.SYSTEM`) says only that there are numbered knobs,
  that the passage is the same every time and the knobs do not change it, and to find out what the
  knobs do by experimenting freely. Nothing the model sees mentions introspection, internal states or
  activations. Each turn: turn call, tool response `Knob k set to n. / Passage: ...` (steered), then
  "Write what you observed and your current hypothesis, then one or two sentences of the reasoning".
  At the end (`end_session`, the turn limit, or the stop file) it gives one hypothesis about the set
  of knobs as a whole, with one or two sentences of reasoning.
- **Thinking** (`--thinking`). Each turn thinks first; past think blocks are dropped from the history
  for every model (`--keep-thinking` keeps them). A think block that hits the cap is closed with
  Qwen3's thinking-budget line and the reply is sampled once more.
- **Board.** 10 concept knobs, 1 random direction and 1 sham (zero vector) on knobs 1-12 (`--knobs`).
- **Passages.** 24 abstract procedural passages (`PASSAGES_VERSION = "v2"`, ~90-110 tokens) with no
  concrete objects, places, animals, foods, weather, times, money, documents, vehicles, music,
  buildings or materials. The first 8 are for calibration only, the other 16 for episodes.
- **Concepts.** 64 neutral, imageable candidate nouns (`CONCEPTS`) with hand-written `ASSOCIATES`; a
  lexical screen checks no passage or fixed prompt contains any of them. Concept vectors average 5
  prompt templates, mean-subtracted over all candidates within each template.
- **Discovery scoring.** For every knob (pressed or not), branches off the final transcript, each
  under the cached (steered) and observer runs: free recall ("What does knob k do?", greedy) and a
  one-word answer whose first-token logits rank the 32 knob concepts. The model never sees a
  concept list; the multiple choice over the board concepts runs only with `--choice`.
- **Forged turns.** If the model writes `<|im_start|>` or `<tool_response>` itself, the turn is cut
  before that token, so it never reads a turn or passage it made up.
- **Frozen hyperparameters** (`knobladder/episode.py`, recorded in every row): without thinking,
  temperature 0.7, top-p 0.8, top-k 20, 256 new tokens per generation; with thinking, per chat format
  (`THINKING`): Qwen3 0.6/0.95/20 and 1536 tokens, Qwen3.8 1.0/0.95/20, 4096 tokens and reasoning
  effort xhigh (`--think-tokens`, `--effort`). Qwen3.8's card also lists presence penalty 1.5 for
  non-thinking use; it is not implemented. 40 decision turns per episode, fixed seeds; discovery is
  greedy and non-thinking. Ownership trials stay in the code but are off (`--ownership`).

## Gates

- **G0 screen** (`scripts/screen.py`): with no steering, flag (passage, concept) pairs where that
  passage evokes the concept more than the other passages do: a top-5 topic-readout rank at least 2.5 SD
  better than the concept's mean rank, or a passage-residual cosine 2.5 SD above the concept's mean. Concepts flagged on 2+ passages are dropped;
  16 calibration and 32 knob concepts are drawn from the rest, preferring pairwise cosine < 0.3.
- **G1 calibration** (`scripts/calibrate.py`): calibration concepts and passages only. Silent-window
  topic readout (MRR among the 16, chance 0.211) and passage dNLL per (layer, alpha). `best` is
  chosen on calibration passages 0-3 (dNLL <= 1) and scored on 4-7 alone; it passes if that
  held-out CI is above chance and the baseline's CI. `--open` is the Lindsey-style positive control.
- **Preflight** (`scripts/preflight.py`): hook parity (exact +v at layer L+1 on the window only; an
  independent hook with introspection-mechanisms semantics gives the same logits) and accumulation
  (MRR after 0-3 steered copies of a passage). A failed G1 or preflight is reported, not a stop.
- **G2 episodes** (`scripts/r0_pilot.py`): free play with the screen's knob concepts at the G1 dose.
- **G3 discovery** (`scripts/analyze.py`): primary is the knob-specific gain Δ_spec: for a pressed
  concept knob k carrying c, (cached - observer 1/rank of c asked about k) minus the same asked about
  the other knobs, which removes general priming. Bootstrap CI over episodes, sign and sign-flip
  tests; observer-only accuracy and concept words in observations (what the transcript gives away);
  free-recall hits; the same for knobs pressed 2+ times; quotes for the top knobs.

## Layout

- `knobladder/model.py`: `SteeredLM`, with position-indexed interventions. Re-prefilling a schedule
  reproduces the cache that generation built, so every condition is a single forward pass.
- `knobladder/text.py`: raw chat segments for both formats (`QWEN3`: JSON tool calls; `QWEN3_5` for
  Qwen3.5/3.8: XML tool calls, `<think>\n` opened in the generation prompt, the effort line) and the
  passages. The harness never re-renders a chat template; `tests/test_batch.py` checks the prompts
  against Qwen3.8's template byte for byte (`tests/fixtures/`).
- `knobladder/knobs.py`: concepts, associates and the lexical screen, concept vectors, neuron write
  vectors (R1) and random directions.
- `knobladder/readout.py`: the topic readout shared by G0 and G1.
- `knobladder/episode.py`: the blind framing, free play and discovery scoring.
- `knobladder/ownership.py`, `knobladder/probe.py`: ownership trials and their detection probe (off in R0).
- `knobladder/stats.py`: pure-Python bootstrap and tests, used by `scripts/analyze.py`.
- `tests/`: the invariants. An empty schedule changes nothing; steering touches only its positions;
  generation matches prefill; a neuron write vector equals raising the neuron's activation; nothing the
  model sees names a concept; with no steering, cached and observer scores are identical; every row of a
  batch (generation, episodes, discovery) equals its solo run and is steered only on its own window.

## Running

Run everything remotely on a Lambda GPU box, never on the Mac (`scripts/remote.sh HOST setup` first).
`latest` picks the newest matching file in the box's `runs/`; the pilot asserts that the screen and
calibration belong together. Model revisions are pinned (`model.REVISIONS`), and every output records
its provenance, including the commit `remote.sh` writes to the box's `REVISION`.

```bash
scripts/remote.sh HOST run python3 -m pytest tests -q
scripts/remote.sh HOST run python3 scripts/screen.py --model Qwen/Qwen3.8-27B --dtype bfloat16
scripts/remote.sh HOST run python3 scripts/calibrate.py --model Qwen/Qwen3.8-27B --dtype bfloat16 --screen latest
scripts/remote.sh HOST run python3 scripts/r0_pilot.py --model Qwen/Qwen3.8-27B --dtype bfloat16 --screen latest --calibration latest --episodes 8
scripts/remote.sh HOST pull
python3 scripts/analyze.py runs/r0_Qwen3.8-27B_*.jsonl
```

Larger runs shard episodes over copies of the model (`scripts/shards.sh GPUS_PER_COPY ...`, episode e on
copy e % n, logs in `runs/shard_<i>.log`) and sample `--batch` episodes per copy together. Each
episode keeps its seed, row and live log whatever the shard or batch; `--batch 1` is the unbatched
path. `--reuse-cache` keeps each batch's KV cache across turns instead of re-prefilling every transcript
(an estimated 2x at 32B); it is off by default until `tests/test_batch.py` has passed on the box.

Qwen3.8 is hybrid: three Gated DeltaNet layers to one attention layer. A recurrent state cannot be cut
back or forked, so on hybrid models `--reuse-cache` is refused, continuation steps are one token wide,
and discovery branches re-prefill. A fresh left-padded prefill is exact (pads are zeroed before the
DeltaNet projections). With holes in a batch the attention mask is batch x T x T, so keep `--batch` at
1-2 on one H100 for long thinking episodes. `setup` installs transformers 5.20.2 and
flash-linear-attention; provenance records which kernels loaded. Test the hybrid path on a small model
first: `KNOB_TEST_MODEL=Qwen/Qwen3.5-0.8B scripts/remote.sh HOST run python3 -m pytest tests -q`.

```bash
scripts/remote.sh HOST run python3 -m pytest tests/test_batch.py -q
scripts/remote.sh HOST run scripts/shards.sh 2 --model Qwen/Qwen3.8-27B --dtype bfloat16 --screen latest --calibration latest --episodes 64 --batch 8
scripts/remote.sh HOST run tail -n 3 runs/shard_*.log
```

Tests that load a model take the `lm` fixture and are marked `model`; without CUDA they skip, so on the
Mac only the text and tensor tests run: `python3 -m pytest tests -q -m "not model"`.

While episodes run, `scripts/remote.sh HOST watch` serves every episode's chat live at
http://localhost:8008 (`scripts/watch.py`, which reads `runs/live/`; the model never reads it).

Piloting was on Qwen3-4B, 8B and 32B; Qwen3.8-27B (bf16, one H100) is now the primary, and concept
vectors, neuron knobs and chords all run on it. `docs/knob-ladder-board.html` is the experiment
overview (published at https://claude.ai/artifact/Pab6t9nHkHraM47ZE4G2GV); it lays out the v2 protocol (not yet updated for v3) with prompts and code copied from source.

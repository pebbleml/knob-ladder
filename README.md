# Knob ladder

Can a model learn, prefer and own its own activations? Spec and research: WIRK wirkspace
`machine consciousness` (283326b9), item `b6b8e429`. This repo is rung R0 (`ec8dd353`).

A model gets unlabelled knobs (`adjust(knob, sign)`) wired to directions in its residual
stream, reshuffled every episode. A knob turn steers only the fixed neutral passage in
the tool response (a silent window), so what the model learns must come from its own
steered states. Each episode measures:

- **Discovery**: a per-knob multiple-choice quiz, scored on the real (steered) cache and on
  the same tokens re-run unsteered (`observer`). The gap is knowledge beyond the transcript.
- **Preference**: presses per hidden label, pooled across episodes because labels move between knobs.
- **Ownership**: the model picks a knob, then the trial forks into identical-dose conditions:
  `self_matched`, `self_swapped`, `self_delayed`, `self_sham`, `other_announced` and
  `other_unannounced`. Each condition is scored by a yes/no report and an internal detection
  probe. A comparator predicts matched < delayed < swapped, and self < announced.

## Layout

- `knobladder/model.py`: `SteeredLM`, with position-indexed interventions. Re-prefilling a schedule
  reproduces the cache that generation built, so every condition is a single forward pass.
- `knobladder/text.py`: raw Qwen3 chat segments. The chat template re-renders and drops think
  blocks, so the harness never re-renders.
- `knobladder/knobs.py`: concept vectors, neuron write vectors (R1) and random directions.
- `knobladder/episode.py`: free play, the quiz and preference counts.
- `knobladder/ownership.py`: the ownership trials.
- `knobladder/probe.py`: the detection probe, a proxy until the Macar/Zou gate features are re-located.
- `tests/`: the invariants. An empty schedule changes nothing; steering touches only its positions;
  generation matches prefill; a neuron write vector equals raising the neuron's activation.

```bash
python3 -m pytest tests -q
```

```bash
python3 scripts/r0_pilot.py --model Qwen/Qwen3-4B --dtype bfloat16 --episodes 2
```

Local runs (Apple M4, 32 GB) are for plumbing only. The science runs on Qwen3-8B and then 32B on GPU.

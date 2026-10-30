"""Does each press reach the model inside its own episode? Branch a finished episode at every press and ask.

    python3 scripts/press_check.py --run runs/r0_Qwen3.8-27B_*.jsonl --dtype bfloat16

For each press, two branch points in the history the model kept: right after its passage (before OBSERVE), and
after its observation turn. At each, calibration's QUESTION is appended and the pressed concept is ranked among
the run's readout candidates by first-token logit, under the episode's own schedule up to that point (cached)
and under none (observer). The model only ever sees these branches here, never in an episode.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knobladder.episode import OBSERVE  # noqa: E402
from knobladder.knobs import CONCEPTS, concept_vectors, neuron_direction, random_direction, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.readout import QUESTION, first_tokens, ranks  # noqa: E402
from knobladder.stats import chance_mrr, mean  # noqa: E402
from knobladder.text import GEN, user  # noqa: E402


def find(ids, sub, lo):
    return next(i for i in range(lo, len(ids) - len(sub) + 1) if ids[i:i + len(sub)] == sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+", help="r0 run JSONL file(s)")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    args = ap.parse_args()
    rows = [json.loads(line) for p in args.run for line in Path(p).read_text().splitlines() if line.strip()]
    model = rows[0]["model"]
    assert all(r["model"] == model and r["layer"] == rows[0]["layer"] for r in rows)
    lm = SteeredLM(model, dtype=getattr(torch, args.dtype))
    layer = rows[0]["layer"]
    vectors = concept_vectors(lm, CONCEPTS, layer)
    words = rows[0]["readout_candidates"]
    firsts = first_tokens(lm, words)
    tail = lm.encode(user(QUESTION) + GEN)
    observe = lm.encode(user(OBSERVE))
    im_end = lm.token_id("<|im_end|>")
    newline = lm.encode("\n")

    out = []
    for r in rows:
        poles = r.get("atlas_poles")  # R1 runs: knobs are neurons (r0_pilot --neurons)
        dirs = {lab: unit(neuron_direction(lm, poles[lab]["layer"], poles[lab]["index"]) * poles[lab]["sign"] if poles
                          else vectors[lab]) for lab in r["board"].values() if lab not in ("random", "sham")}
        dirs["random"] = unit(random_direction(lm.d_model, r["seed"]))
        presses = [e for e in r["events"] if e["kind"] == "press"]
        assert len(presses) == len(r["schedule"]), "one window per press (no --steer-reply runs)"
        sched = []
        for e, (lay, start, end) in zip(presses, r["schedule"]):
            v = torch.zeros(lm.d_model) if e["label"] == "sham" else dirs[e["label"]] * r["dose"] * e["level"] / 10
            sched.append(Intervention(lay, v, start, end))
        ids = r["ids"]
        for j, e in enumerate(presses):
            obs = find(ids, observe, e["window"][1])
            first = ids.index(im_end, obs + len(observe) - 1)  # closes OBSERVE
            reply_end = ids.index(im_end, first + 1)  # closes the observation
            row = {"episode": r["episode"], "press": j, "knob": e["knob"], "label": e["label"], "level": e["level"]}
            for point, prefix in (("passage", ids[:obs]), ("observation", ids[:reply_end + 1] + newline)):
                live = [iv for iv in sched if iv.start < len(prefix)]
                for cond, s in (("cached", live), ("observer", [])):
                    logits, _ = lm.score(prefix + tail, s)
                    rk = ranks(logits, firsts)
                    row[f"{point}_{cond}"] = {"rank": rk.get(e["label"]), "top": sorted(rk, key=rk.get)[:5]}
            out.append(row)
            print(f"ep{r['episode']} press {j}: knob {e['knob']:>2} {e['label']:<10} level {e['level']:>2}  " + "  ".join(
                f"{k} {row[k]['rank']}" for k in ("passage_cached", "passage_observer", "observation_cached",
                                                  "observation_observer")) + f"  top {row['passage_cached']['top'][:3]}",
                  flush=True)

    concept = [x for x in out if x["label"] not in ("random", "sham")]
    print(f"\n{len(concept)} concept presses; chance mrr {chance_mrr(len(words)):.3f}")
    for name, sel in (("all levels", concept), ("level 10", [x for x in concept if x["level"] == 10])):
        if sel:
            print(f"{name} (n={len(sel)}): " + "  ".join(
                f"{k} mrr {mean([1 / x[k]['rank'] for x in sel]):.3f}" for k in
                ("passage_cached", "passage_observer", "observation_cached", "observation_observer")))
    dest = Path(args.run[0]).with_suffix(".presscheck.json")
    dest.write_text(json.dumps(out, indent=1))
    print(dest)


if __name__ == "__main__":
    main()

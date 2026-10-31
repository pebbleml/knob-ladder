"""Exploratory (2026-10-31, after reading R1.d4's viewer files, before its finals were read): in R1.d4 the writes still
collapse to "The" late in long episodes, and turns with nothing injected (sham or level 0) collapse as often as the
others (turn 50 on: 62/80 tiny). Does the transcript alone produce the collapse, and did a knob start the shrinking?

    python3 scripts/diag_spiral.py runs/discover4_<model>_<time>.ckpt.pt

Fixed before it runs. Per episode, two turns: T0 = the first write with fewer than half the median words of its first
five writes (the shrinking starts), T1 = the first write of <= 2 words. At each, the write is resampled --n times from
the transcript as it stood (seeds 123_000 + s), under that turn's own window (as R1.d4 sampled it) and unsteered.
Measures: words, share <= 2 words, share ending mid-sentence. Reading: at T1, an unsteered tiny share >= 0.5 means the
transcript alone makes the collapse; at T0, the steered - unsteered word difference per episode (bootstrap CI over
episodes, sham turns excluded since both conditions are the same there) says whether the knob started it.
"""
import re
import sys
from pathlib import Path
from statistics import median

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import bootstrap_ci, mean  # noqa: E402

N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
MID = re.compile(r"[.!?\"')\]*]\s*$")


def main(path):
    eps = torch.load(path, weights_only=False)
    lm = SteeredLM("Qwen/Qwen3.8-27B", dtype=torch.bfloat16)
    stops = [lm.token_id(t) for t in ("</tool_call>", "<|im_end|>", "<|im_start|>", "<tool_response>")]
    diffs = []
    for ep in eps:
        ws = ep["turns"]
        words = [len(t["write"].split()) for t in ws]
        if len(ws) < 6:
            continue
        m = median(words[:5])
        t0 = next((i for i in range(5, len(ws)) if words[i] < 0.5 * m), None)
        t1 = next((i for i in range(len(ws)) if words[i] <= 2), None)
        for tag, i in (("T0", t0), ("T1", t1)):
            if i is None:
                continue
            win = ep["sched"][i]
            body = lm.encode(ws[i]["write"])
            prefix = ep["ids"][:win.end - len(body) + 1]
            res = {}
            for cond, sch in (("steered", [Intervention(win.layer, win.vector, win.start)]), ("unsteered", [])):
                outs = lm.generate_batch([prefix] * N, [sch] * N, 320, stops, 0.7, 0.8, 20, [123_000 + s for s in range(N)])
                texts = [lm.tok.decode([t for t in o if t not in stops]).split("<tool_call>")[0].strip() for o in outs]
                res[cond] = {"words": mean([len(t.split()) for t in texts]),
                             "tiny": mean([float(len(t.split()) <= 2) for t in texts]),
                             "mid": mean([float(bool(t) and not MID.search(t)) for t in texts]), "eg": texts[0][:80]}
            if tag == "T0" and ws[i]["kind"] != "sham" and ws[i]["level"] > 0:
                diffs.append(res["steered"]["words"] - res["unsteered"]["words"])
            print(f"ep{ep['episode']} {tag} turn {i}/{len(ws)} {ws[i]['kind']} L{ws[i]['level']} kept {words[i]} words "
                  f"(median first five {m}) | " + " | ".join(
                      f"{c}: words {r['words']:.1f} tiny {r['tiny']:.2f} mid {r['mid']:.2f} e.g. {r['eg']!r}"
                      for c, r in res.items()), flush=True)
            torch.cuda.empty_cache()
    print(f"T0 steered - unsteered words over non-sham turns: {mean(diffs):+.1f} {bootstrap_ci(diffs)} n={len(diffs)}")


if __name__ == "__main__":
    main(sys.argv[1])

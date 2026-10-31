"""Exploratory (2026-10-31, after reading R1.d3): why do writes stop mid-sentence and then shrink to "The"?

    python3 scripts/diag_collapse.py runs/discover3_<model>_<time>.ckpt.pt

For each episode's first write that ends mid-sentence: the next-token distribution right after its last token, with the
episode's steering (cached) and without (observer); then with only the current turn's window, and only the earlier
turns' windows. Which token ended it, how likely was it, and which steering made it likely?
"""
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knobladder.model import Intervention, SteeredLM  # noqa: E402


def main(path):
    eps = torch.load(path, weights_only=False)
    lm = SteeredLM("Qwen/Qwen3.8-27B", dtype=torch.bfloat16)
    for ep in eps:
        ws = ep["turns"]
        i = next((i for i, tr in enumerate(ws) if tr["write"].strip()
                  and not re.search(r"[.!?\"')\]*]\s*$", tr["write"].strip())), None)
        if i is None:
            continue
        win = ep["sched"][i]
        end = win.end  # the kept body ends with the sampled im_end at position win.end
        ids = ep["ids"][:end]
        sched = [Intervention(w.layer, w.vector, w.start, w.end) for w in ep["sched"][:i]]
        sched.append(Intervention(win.layer, win.vector, win.start, end))
        rows = []
        zero = lambda ws: [Intervention(w.layer, torch.zeros_like(w.vector), w.start, w.end) for w in ws]
        for cond, s in (("cached", sched), ("observer", []), ("current window only", zero(sched[:-1]) + sched[-1:]),
                        ("earlier windows only", sched[:-1] + zero(sched[-1:]))):
            logits, _ = lm.score(ids, s)
            p = torch.softmax(logits.float(), -1)
            top = torch.topk(p, 5)
            rows.append(f"{cond}: " + ", ".join(f"{lm.tok.decode([int(t)])!r} {float(v):.2f}" for v, t in zip(top.values, top.indices)))
        print(f"ep{ep['episode']} turn {i} {ws[i]['kind']} L{ws[i]['level']} ...{ws[i]['write'][-50:]!r} | next kept "
              f"{lm.tok.decode(ep['ids'][end:end + 3])!r}\n  " + "\n  ".join(rows), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

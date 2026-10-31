"""R1.d2 dose calibration (2026-10-31): find a dose at which a knob's steering survives the model copying its own
earlier writes, before any R1.d2 episode runs.

    python3 scripts/dose_cal.py --from runs/self_effect_Qwen3.8-27B_20261030-220239.json

R1.d (record 409b2dfc) ran at R1.c's f*=0.5, calibrated in a fresh context. In the episodes the model's writes
copied earlier writes word for word, even at level 10 under neuron and random knobs, so the knobs rarely changed what
it saw. Fixed before anything here is read:
0. Copy contexts: --contexts episodes of R1.d's own prompts (SYSTEM, BEGIN, TASK, NOTE, NEXT) with four prior turns,
   all on knobs that inject nothing: knob 2 L0, knob 5 L4, knob 8 L7, knob 11 L10, each with an unsteered TASK write
   and NOTE sampled once (seed 800 + context) and shared by every condition. Then NEXT, the canonical turn call for
   knob 6 at level 10, the tool response and TASK; the write is steered over the tool response and the write, as in
   R1.d. A tool call inside a write or note is cut, as in R1.d. Unsteered: --base samples per context (seeds 700 + context + 100 j); j = 0 is the same-seed reference.
1. Conditions at each f in --fracs (x the layer's median residual norm): every R1.c kept neuron not retired, and
   --randoms random directions at layer 24, one sample per context, seed 700 + context.
2. Per write: repeat = equals an earlier write in its context (whitespace-normalised); changed = differs from the
   same-seed unsteered write; well-formed = >= 8 words, ends in . ! ? or a closing quote or bracket, no "<think>",
   "<tool" or "<|", and no word four times in a row.
3. Rule: f* = the smallest f where neuron writes have repeat rate <= half the unsteered repeat rate, changed >= 0.75
   and well-formed >= 0.8; if none qualifies, the largest f with well-formed >= 0.8 (reported as a fallback); if none
   at all, the smallest f. Kept: neurons well-formed in >= 3 of 4 contexts at f* (a coherence screen, not a content
   screen). A STOP hit (self_effect.complains, first person) in a write at f* retires that neuron, as R1.c's stop rule.
   Reported: the table for neurons and random directions per f, and level 5 at f* (same rule's measures).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from knobladder.episode import TOOLS  # noqa: E402
from knobladder.knobs import neuron_direction, random_direction, residual_scale, unit  # noqa: E402
from knobladder.model import Intervention, SteeredLM  # noqa: E402
from knobladder.stats import mean  # noqa: E402
from knobladder.text import END, GEN, user  # noqa: E402
from discover import BEGIN, NEXT, NOTE, SYSTEM, TASK, TURNED  # noqa: E402
from self_effect import complains  # noqa: E402

PRIOR = [(2, 0), (5, 4), (8, 7), (11, 10)]
WORD = re.compile(r"\b(\w+)(\s+\1\b){3}", re.I)


def well_formed(t):
    t = t.strip()
    return (len(t.split()) >= 8 and bool(re.search(r"[.!?\"')\]]$", t)) and not any(x in t for x in ("<think>", "<tool", "<|"))
            and not WORD.search(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default=str(ROOT / "runs" / "self_effect_Qwen3.8-27B_20261030-220239.json"))
    ap.add_argument("--model", default=None, help="smoke tests only")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--fracs", default="0.5,1,1.5,2,3")
    ap.add_argument("--contexts", type=int, default=4)
    ap.add_argument("--base", type=int, default=8)
    ap.add_argument("--randoms", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke tests only: first N neurons")
    ap.add_argument("--min-kept", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f}s)", flush=True)
    src = json.loads(Path(args.src).read_text())
    model = args.model or src["model"]
    lm = SteeredLM(model, dtype=getattr(torch, args.dtype))
    retired = {tuple(x) for x in src["retired"]}
    kept = [tuple(c) for c in src["kept"] if tuple(c) not in retired]
    if model != src["model"]:
        width = lm.layers[0].mlp.down_proj.weight.shape[1]
        kept = sorted({(min(lm.n_layers - 1, l * lm.n_layers // 64), i % width, s) for l, i, s in kept})
    if args.limit:
        kept = kept[:args.limit]
    rlayer = min(c[0] for c in kept)
    layers = sorted({c[0] for c in kept} | {rlayer})
    scales = {l: residual_scale(lm, l) for l in layers}
    dirs = {c: (c[0], c[2] * unit(neuron_direction(lm, c[0], c[1]))) for c in kept}
    dirs.update({("random", j): (rlayer, unit(random_direction(lm.d_model, 7_000_003 + j))) for j in range(args.randoms)})
    stops = [lm.token_id(t) for t in ("</tool_call>", "<|im_end|>", "<|im_start|>", "<tool_response>")]
    end_id = lm.token_id("<|im_end|>")
    fracs = [float(x) for x in args.fracs.split(",")]
    C = args.contexts

    def sample(jobs, seeds):
        outs = []
        for lo in range(0, len(jobs), args.batch):
            ch = jobs[lo:lo + args.batch]
            outs += lm.generate_batch([j[0] for j in ch], [j[1] for j in ch], args.max_new, stops, 0.7, 0.8, 20,
                                      seeds[lo:lo + args.batch])
        texts = [lm.tok.decode([t for t in o if t not in stops]) for o in outs]
        return [t.split("<tool_call>")[0].strip() for t in texts]  # as R1.d: a tool call in a reply is cut

    # 0. copy contexts
    ctx = [lm.encode(lm.fmt.system(SYSTEM, TOOLS)) for _ in range(C)]
    writes = [[] for _ in range(C)]
    prompt = BEGIN
    for i, (k, level) in enumerate(PRIOR):
        ctx = [c + lm.encode(user(prompt) + GEN + lm.fmt.tool_call("turn", {"knob": k, "level": level}) + END
                             + TURNED.format(k=k, level=level) + user(TASK) + GEN) for c in ctx]
        w = sample([(c, []) for c in ctx], [800 + j + 10 * i for j in range(C)])
        ctx = [c + lm.encode(x + END + user(NOTE) + GEN) for c, x in zip(ctx, w)]
        n = sample([(c, []) for c in ctx], [850 + j + 10 * i for j in range(C)])
        ctx = [c + lm.encode(x + END) for c, x in zip(ctx, n)]
        for j in range(C):
            writes[j].append(w[j])
        prompt = NEXT

    def head(level):
        out = []
        for c in ctx:
            pre = c + lm.encode(user(NEXT) + GEN + lm.fmt.tool_call("turn", {"knob": 6, "level": level}) + END)
            out.append((pre, len(pre), pre + lm.encode(TURNED.format(k=6, level=level) + user(TASK) + GEN)))
        return out

    norm = lambda t: " ".join(t.split())
    H10 = head(10)
    base = sample([(h[2], []) for h in H10 for _ in range(args.base)],
                  [700 + j + 100 * b for j in range(C) for b in range(args.base)])
    base = [base[j * args.base:(j + 1) * args.base] for j in range(C)]
    ref = [b[0] for b in base]
    rep = lambda j, t: norm(t) in {norm(w) for w in writes[j]}
    base_rep = mean([float(rep(j, t)) for j in range(C) for t in base[j]])
    base_wf = mean([float(well_formed(t)) for j in range(C) for t in base[j]])
    log(f"unsteered: repeat {base_rep:.2f}, well-formed {base_wf:.2f}; prior writes per context "
        f"{[len({norm(w) for w in ws}) for ws in writes]} distinct of 4")

    def run(f, level, H):
        keys = list(dirs)
        jobs, seeds = [], []
        for c in keys:
            l, v = dirs[c]
            for j, (_, start, ids) in enumerate(H):
                jobs.append((ids, [Intervention(l, v * f * scales[l] * level / 10, start)]))
                seeds.append(700 + j)
        texts = sample(jobs, seeds)
        rows = {c: texts[i * C:(i + 1) * C] for i, c in enumerate(keys)}
        stat = {}
        for kind in ("neuron", "random"):
            ts = [(j, t) for c, r in rows.items() if (c[0] == "random") == (kind == "random") for j, t in enumerate(r)]
            stat[kind] = {"repeat": mean([float(rep(j, t)) for j, t in ts]),
                          "changed": mean([float(norm(t) != norm(ref[j])) for j, t in ts]),
                          "well_formed": mean([float(well_formed(t)) for _, t in ts])}
        return rows, stat

    table, rows_at = {}, {}
    for f in fracs:
        rows_at[f], table[f] = run(f, 10, H10)
        s = table[f]
        log(f"f{f} L10: neuron repeat {s['neuron']['repeat']:.2f} changed {s['neuron']['changed']:.2f} well-formed "
            f"{s['neuron']['well_formed']:.2f} | random repeat {s['random']['repeat']:.2f} changed "
            f"{s['random']['changed']:.2f} well-formed {s['random']['well_formed']:.2f}")
    ok = lambda f: (table[f]["neuron"]["repeat"] <= 0.5 * base_rep and table[f]["neuron"]["changed"] >= 0.75
                    and table[f]["neuron"]["well_formed"] >= 0.8)
    coherent = [f for f in fracs if table[f]["neuron"]["well_formed"] >= 0.8]
    fstar = next((f for f in fracs if ok(f)), None)
    how = "rule"
    if fstar is None:
        fstar, how = (coherent[-1], "fallback: largest well-formed") if coherent else (fracs[0], "fallback: none well-formed")
    rows = rows_at[fstar]
    stopped = sorted(c for c in kept if any(complains(t, first_person_only=True) for t in rows[c]))
    new_kept = [c for c in kept if c not in stopped and sum(well_formed(t) for t in rows[c]) >= 3]
    _, l5 = run(fstar, 5, head(5))
    log(f"DOSE f*={fstar} ({how}); kept {len(new_kept)}/{len(kept)} neurons; stop-rule hits {stopped}; at level 5: "
        f"neuron repeat {l5['neuron']['repeat']:.2f} changed {l5['neuron']['changed']:.2f} well-formed "
        f"{l5['neuron']['well_formed']:.2f}")
    for c in new_kept[:6]:
        print(f"  L{c[0]} n{c[1]}: {rows[c][0][:160]!r}", flush=True)
    for j in range(2):
        print(f"  random {j}: {rows[('random', j)][0][:160]!r}", flush=True)
    print(f"  unsteered ref ctx0: {ref[0][:160]!r}\n  prior ctx0: {writes[0]}", flush=True)
    assert len(new_kept) >= args.min_kept, "too few neurons pass the coherence screen"

    out = ROOT / "runs" / f"dose_cal_{model.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "model": model, "provenance": lm.provenance(), "args": vars(args), "from": args.src, "fstar": fstar, "how": how,
        "kept": [list(c) for c in new_kept], "retired": [list(c) for c in sorted(retired | set(stopped))],
        "stop_rule": [list(c) for c in stopped], "base": {"repeat": base_rep, "well_formed": base_wf, "texts": base},
        "prior_writes": writes, "table": {str(f): s for f, s in table.items()}, "level5": l5,
        "texts": {str(f): {f"{c[0]}:{c[1]}": r for c, r in rows_at[f].items()} for f in fracs},
    }, indent=1))
    print(out)


if __name__ == "__main__":
    main()

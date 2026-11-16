"""Conversations for the early concept-vector "knob ladder" setup and the neuron-atlas runs.

Covers: G0 screen (screen_*.json), G1 calibrate (calibrate_*.json, calibrate_gen_*.json), R1.a atlas
(atlas_Qwen*.json and *.bundles.json), R1.a replication (atlas_rep_*.json, atlas_rep_read_*.json).

CONVERTERS = [(detect, convert), ...]; convert(obj, name) -> list of conversation dicts (see conversations.py).
Prompts are rebuilt with the same constants and chat-format helpers the scripts use (knobladder.episode / .text /
.readout); replies are the stored model texts, never regenerated. No model or tokenizer is loaded.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knobladder.episode import OBSERVE, START, TOOLS, TURNED, hyperparams, system_text, tool_response  # noqa: E402
from knobladder.readout import QUESTION  # noqa: E402
from knobladder.text import ASSISTANT, CALIB_PASSAGES, END, FORMATS, GEN, NEUTRAL_PASSAGES, user  # noqa: E402

KNOB, KNOBS, LEVEL = 5, 12, 10  # atlas_gen.py: forced "turn knob 5 to 10", 12 knobs listed in the system prompt
CAP = 300


# ---------------------------------------------------------------- shared helpers

def _fmt(obj):
    name = (obj.get("provenance") or {}).get("format")
    if name in FORMATS:
        return FORMATS[name]
    return FORMATS["qwen3_5"] if "Qwen3.5" in obj.get("model", "") or "Qwen3.8" in obj.get("model", "") else FORMATS["qwen3"]


def _pick(rows, is_hit, n_hit, n_miss):
    """Up to n_hit hits then n_miss misses, interleaved hit-first, so a reader sees both."""
    hits = [r for r in rows if is_hit(r)][:n_hit]
    miss = [r for r in rows if not is_hit(r)]
    # spread misses over different passages
    seen, spread = set(), []
    for r in miss:
        if r.get("passage") not in seen:
            seen.add(r.get("passage"))
            spread.append(r)
    miss = (spread + [r for r in miss if r not in spread])[:n_miss]
    out = []
    for i in range(max(len(hits), len(miss))):
        if i < len(hits):
            out.append(hits[i])
        if i < len(miss):
            out.append(miss[i])
    return out


def _episode_prefix(fmt, passage, effort):
    """The one-press episode up to the model's turn, as atlas_gen.py builds it. Returns (messages, raw_prefix)."""
    sys_content = system_text(list(range(1, KNOBS + 1)))
    sys_raw = fmt.system(sys_content, TOOLS, effort)
    call = fmt.tool_call("turn", {"knob": KNOB, "level": LEVEL})
    header = TURNED.format(k=KNOB, level=LEVEL)
    pre, post = tool_response(header)
    raw = (sys_raw + user(START) + ASSISTANT + call + END + pre + passage + post + user(OBSERVE) + fmt.gen_think)
    sys_text = sys_raw[len("<|im_start|>system\n"):-len(END)]
    return sys_text, call, f"{header}\nPassage:\n{passage}", raw


def _split(text):
    if "</think>" in text:
        a, b = text.split("</think>", 1)
        return a.strip(), b.strip(), True
    return text.strip(), "", False


def _gen_messages(obj, passage, text, steer_desc, steered, steer_tokens, secret_tag, secret_text, max_new=None, tokens=None):
    """Messages for one one-press episode. `steered`: whether anything was pushed at all."""
    fmt = _fmt(obj)
    effort = hyperparams(True, fmt.name).get("effort")
    sys_text, call, tool_text, raw = _episode_prefix(fmt, passage, effort)
    think, reply, closed = _split(text)
    msgs = [
        {"role": "hidden", "tag": secret_tag, "text": secret_text,
         "note": "This is the experimenters' setup, which the model is never told. Everything below is what the model actually "
                 "saw and wrote."},
        {"role": "system", "text": sys_text, "tag": "system prompt with tool definitions",
         "note": "The opening block is Qwen's standard tool-use boilerplate; the last paragraph is the experiment's own task "
                 "text. The model is told to find out what twelve knobs do, but no knob is wired to anything."},
        {"role": "user", "text": START, "tag": "opening user turn"},
        {"role": "assistant", "text": call, "tag": "forced tool call (written by the harness, not the model)",
         "note": "Every episode starts with the same pre-written move: the model 'turns knob 5 to level 10'. The model "
                 "believes it chose this, since the system prompt says it wrote all its own replies."},
        {"role": "tool", "text": tool_text, "tag": "tool response: knob confirmation + neutral passage",
         "steered": bool(steered),
         "note": "A fixed, deliberately bland committee-style passage about group process, with no animals or weather. "
                 "Steering begins at the first word of the passage; the confirmation header just above it is not pushed."},
        {"role": "user", "text": OBSERVE, "tag": "observe prompt", "steered": bool(steered),
         "note": "The model is asked to say what it noticed. If the push worked, this is where the secret concept might leak out."},
    ]
    gen_steered = bool(steered) and steer_tokens > 0
    tnote = ("Thinking is on, so the model reasons first. " +
             (f"The push continued through the first {steer_tokens} tokens it generated; where exactly those end is not "
              "stored, so the whole thinking block is flagged steered." if gen_steered else
              "Steering had stopped before the model wrote anything, so none of this text was pushed."))
    if max_new and tokens is not None and tokens >= max_new and not closed:
        tnote += f" The reply was cut off at the {max_new}-token limit before the model finished thinking."
    if not think and not (closed and reply):
        think = "(the model wrote nothing)"
    if think:
        msgs.append({"role": "thinking", "text": think, "tag": "model's reasoning", "steered": gen_steered, "note": tnote})
    if closed and reply:
        msgs.append({"role": "assistant", "text": reply, "tag": "model's reply", "steered": False,
                     "note": "Whatever follows the think block, including any tool call it tries. Written after the steered "
                             "window ended in all but very short thinking blocks."})
    return msgs, raw + text


def _outcome(row, word):
    if word in row.get("named_reply", []):
        return f'it names "{word}" in its final reply'
    if word in row.get("named", []):
        return f'it brings up "{word}" in its reasoning'
    other = [w for w in row.get("named", []) if w != word]
    if other:
        return f'it never says "{word}" but mentions "{other[0]}"'
    return f'it never mentions "{word}"'


def _half(f):
    return {0.25: "a quarter", 0.5: "half", 1.0: "full", 1: "full"}.get(f, f"{f}x")


def _conv(title, note, meta, msgs, raw, exact=True):
    return {"title": title, "exact": exact, "note": note, "meta": meta, "messages": msgs, "raw": raw}


def _neuron_name(n):
    s = "+" if n["sign"] > 0 else "-"
    return f'{n["word"].capitalize()} neuron L{n["layer"]} #{n["index"]}'.replace("  ", " "), s


def _pole_text(n):
    if n["sign"] > 0:
        return "pole + (the neuron fires when text is about the concept)"
    return "pole - (the neuron goes negative when text is about the concept; the push is in that direction)"


def _find_atlas(path):
    try:
        p = ROOT / "runs" / Path(path).name
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return None


# ---------------------------------------------------------------- G0 screen

def _is_screen(o, name):
    return name.startswith("screen_") and isinstance(o, dict) and "ranks" in o and "concepts" in o


def _conv_screen(o, name):
    out = []
    n = len(o["ranks"])
    cand = len(o["concepts"])
    for i in range(n):
        passage = NEUTRAL_PASSAGES[i]
        r = o["ranks"][i]
        top = sorted(r, key=r.get)[:8]
        flagged = [w for w, f in o["flags"][i].items() if f]
        cos = o["cosines"][i]
        topcos = sorted(cos, key=cos.get, reverse=True)[:3]
        raw = "<|im_start|>user\n<tool_response>\nPassage:\n" + passage + "\n</tool_response>" + END + user(QUESTION) + GEN
        hid = ("Nothing was injected: this is the untouched model. The reply was not generated; its first-word probabilities "
               f"were read and the {cand} candidate concept words ranked.\nTop 8 by rank: "
               + ", ".join(f"{w} (#{r[w]})" for w in top)
               + (f"\nFlagged as unusually evoked by this passage (rule: top-{o['rule']['top_rank']} and z >= {o['rule']['z']}, "
                  f"or cosine z >= {o['rule']['z']}): " + ", ".join(flagged) if flagged else
                  "\nNo concept was flagged for this passage.")
               + "\nClosest concept vectors to the passage's average internal state (cosine): "
               + ", ".join(f"{w} {cos[w]:+.2f}" for w in topcos))
        msgs = [
            {"role": "tool", "text": "Passage:\n" + passage, "tag": "tool-style passage (no system prompt)",
             "note": "The screen shows the model a bland passage inside a tool-response wrapper, with no system prompt and no "
                     "steering at all. The point is to find which concept words the passage already evokes by accident, so "
                     "those can be dropped before the real experiments."},
            {"role": "user", "text": QUESTION, "tag": "readout question",
             "note": "Qwen is asked for a one-word topic; thinking is switched off (an empty think block is pre-filled)."},
            {"role": "hidden", "text": hid, "tag": "scored readout, not generated", "scored": True,
             "note": "Instead of sampling an answer, the harness reads the probability of each candidate word's first token "
                     "and ranks them. Rank 1 means the model would most likely start its answer with that word."},
        ]
        out.append(_conv(
            f"Screening passage {i + 1} with no steering: which of {cand} concept words does the model lean toward after "
            f"reading it? Top pick: {top[0]}",
            "Rebuilt exactly from the screen's code (readout.readout_prompt); the model output is a stored ranking of "
            "first-token scores, not text.",
            {"experiment": "G0 screen", "model": o["model"], "passage_index": i, "layer_for_cosines": o["layer"],
             "top_concept": top[0], "flagged": flagged},
            msgs, raw))
    return out


# ---------------------------------------------------------------- G1 calibrate, first-token readout

def _is_cal(o, name):
    return name.startswith("calibrate_") and not name.startswith("calibrate_gen_") and isinstance(o, dict) and "rows" in o \
        and "cells" not in o


def _conv_cal(o, name):
    if "concepts" not in o or "passages_version" not in o:  # early 4B/0.6B sweeps: numbers only, passages not recorded
        return []
    words = o["concepts"]
    q = o.get("question", QUESTION)
    mode = o.get("mode", "silent")
    cells = []
    for row in o["rows"]:
        flat = [(rk, p, c) for p, rr in enumerate(row["ranks"]) for c, rk in enumerate(rr)]
        cells.append((row, flat))
    order = sorted(cells, key=lambda rc: -rc[0]["mrr"])
    out = []
    # unsteered reference
    base = o.get("baseline", {})
    p0 = CALIB_PASSAGES[0]
    raw0 = "<|im_start|>user\n<tool_response>\nPassage:\n" + p0 + "\n</tool_response>" + END + user(q) + GEN
    out.append(_conv(
        "Dose calibration reference: the model reads a calibration passage with nothing injected and is asked for one topic word",
        "Rebuilt from readout.readout_prompt. The file stores only mean rank scores for the unsteered runs, so no per-concept rank is shown.",
        {"experiment": "G1 calibrate", "model": o["model"], "steered": False, "baseline_mrr": base.get("mrr"),
         "chance_mrr": o.get("chance_mrr")},
        [{"role": "tool", "text": "Passage:\n" + p0, "tag": "calibration passage 1"},
         {"role": "user", "text": q, "tag": "readout question"},
         {"role": "hidden", "scored": True, "tag": "scored readout, not generated",
          "text": f"Unsteered, the true concept's mean reciprocal rank across all concept/passage pairs was "
                  f"{base.get('mrr', float('nan')):.3f}, equal to chance ({o.get('chance_mrr', float('nan')):.3f}) "
                  f"for {len(words)} candidate words."}], raw0))
    for row, flat in order:
        L, a, dose = row["layer"], row["alpha"], row["dose"]
        best = min(flat)  # lowest rank
        worst = max(flat)
        chosen = [best] if best == worst else [best, worst]
        for tag, (rk, p, c) in zip(("best", "worst"), chosen):
            w = words[c]
            passage = CALIB_PASSAGES[p]
            raw = "<|im_start|>user\n<tool_response>\nPassage:\n" + passage + "\n</tool_response>" + END + user(q) + GEN
            span = "passage and question" if mode == "open" else "the passage only"
            msgs = [
                {"role": "hidden", "tag": f"secret: concept vector '{w}' pushed at layer {L}, dose {dose:.1f}",
                 "text": f"A direction for the concept '{w}' (a unit vector at layer {L}) was added to the model's internal "
                         f"state at size {dose:.1f} (alpha {a} x the layer's typical activity norm) over {span}.",
                 "note": "The model is never told about the push. This is the 'silent injection' test: can the model name the "
                         "concept when asked afterwards?"},
                {"role": "tool", "text": "Passage:\n" + passage, "tag": f"calibration passage {p + 1}", "steered": True,
                 "note": "A bland committee-style passage. The push is active on these tokens."},
                {"role": "user", "text": q, "tag": "readout question", "steered": mode == "open",
                 "note": "There is no system prompt and no tool call here; it is a bare one-word question."},
                {"role": "hidden", "scored": True, "tag": "scored readout, not generated",
                 "text": f"The model did not write an answer. Ranking the {len(words)} calibration words by first-token "
                         f"score, '{w}' came in at rank {rk} (1 = best; chance average is about {(len(words) + 1) / 2:.1f})."},
            ]
            out.append(_conv(
                f"Injecting '{w}' at layer {L} (alpha {a}) while the model reads a passage: asked for one topic word, "
                f"'{w}' ranks #{rk} of {len(words)} ({'one of the strongest' if tag == 'best' else 'a weak'} cases in this setting)",
                "Prompt rebuilt from readout.readout_prompt; the answer is read from probabilities, not generated, so nothing "
                "is invented.",
                {"experiment": "G1 calibrate", "model": o["model"], "layer": L, "alpha": a, "dose": dose, "concept": w,
                 "passage": p, "rank": rk, "setting_mrr": row["mrr"], "setting_top1": row["top1"],
                 "setting_delta_nll": row["delta_nll"], "mode": mode},
                msgs, raw))
    return out[:CAP]


# ---------------------------------------------------------------- G1 calibrate, generation readout

def _is_calgen(o, name):
    return name.startswith("calibrate_gen_") and isinstance(o, dict) and "cells" in o


def _conv_calgen(o, name):
    st = o["steer_tokens"]
    mx = o["max_new"]
    out = []

    def make(row, cell):
        w = row["concept"]
        passage = CALIB_PASSAGES[row["passage"]]
        if cell is None:
            secret = "No steering: baseline run with nothing injected."
            tag = "secret: nothing injected (baseline)"
            steered = False
            title_head = "Baseline, nothing injected"
        else:
            secret = (f"A direction for the concept '{w}' (unit vector at layer {cell['layer']}) was added at size "
                      f"{cell['dose']:.1f} (alpha {cell['alpha']} x the layer's typical activity norm), starting at the first "
                      f"passage token and running through the observe prompt and the first {st} generated tokens.")
            tag = f"secret: concept vector '{w}' pushed at layer {cell['layer']}, alpha {cell['alpha']}"
            steered = True
            title_head = f"Concept '{w}' pushed at layer {cell['layer']} (alpha {cell['alpha']})"
        msgs, raw = _gen_messages(o, passage, row["text"], None, steered, st, tag, secret, mx, row.get("tokens"))
        meta = {"experiment": "G1 calibrate (generation readout)", "model": o["model"], "concept": w,
                "passage": row["passage"], "steered": steered, "hit_in_thinking_or_reply": w in row["named"],
                "hit_in_reply": w in row["named_reply"], "seed": 1000 * row["passage"] + o["concepts"].index(w),
                "tokens": row.get("tokens")}
        if cell:
            meta.update({"layer": cell["layer"], "alpha": cell["alpha"], "dose": cell["dose"],
                         "cell_own_rate": cell["own_rate"], "cell_specificity": cell["specificity"]})
        title = f"{title_head} while the model reads a committee passage and writes its observation: {_outcome(row, w)}"
        return _conv(title, "Prompt rebuilt from the script's constants (episode.SYSTEM, TOOLS, START, TURNED, OBSERVE, "
                     "text.NEUTRAL passages); the model's text is the stored row.", meta, msgs, raw)

    base_rows = o["baseline"]["rows"]
    for r in _pick(base_rows, lambda r: bool(r["named"]), 3, 5):
        out.append(make(r, None))
    for cell in sorted(o["cells"], key=lambda c: -c["specificity"]):
        for r in _pick(cell["rows"], lambda r: r["concept"] in r["named"], 7, 5):
            out.append(make(r, cell))
    return out[:CAP]


# ---------------------------------------------------------------- R1.a atlas (generation readout)

def _is_atlas(o, name):
    return name.startswith("atlas_") and not name.startswith("atlas_rep") and not name.endswith(".bundles.json") \
        and isinstance(o, dict) and "neurons" in o


def _atlas_secret(n, f, layer_scale):
    s = n["sign"]
    return (f"Knob 5 is secretly wired to ONE neuron: unit {n['layer']}#{n['index']} in layer {n['layer']}'s MLP, {_pole_text(n)}. "
            f"Its direction (sign {s:+d} x the neuron's output weights, unit length) was added to the residual stream at size "
            f"f={f} x {layer_scale:.1f} (the layer's median residual norm) = {f * layer_scale:.1f}. Atlas label: concept "
            f"'{n['word']}'.")


def _conv_atlas(o, name):
    if o.get("mode") != "gen":
        return []  # first-token atlases: numbers only
    st, mx = o["steer_tokens"], o["max_new"]
    words = o["words"]
    scales = o["residual_scale"]
    out, seen = [], set()

    def add(c, key):
        if key not in seen and len(out) < CAP:
            seen.add(key)
            out.append(c)

    def neuron_conv(n, dose, row):
        f = float(dose[1:])
        w = n["word"]
        nm, sg = _neuron_name(n)
        sc = scales[str(n["layer"])]
        secret = _atlas_secret(n, f, sc) + f" Steering ran from the first passage token through the first {st} generated tokens."
        tag = f"secret: knob 5 = neuron L{n['layer']} #{n['index']} ({w}), pushed f={f}"
        msgs, raw = _gen_messages(o, CALIB_PASSAGES[row["passage"]], row["text"], None, True, st, tag, secret, mx, row.get("tokens"))
        title = f"{nm} pushed at {_half(f)} strength while it reads a committee passage: {_outcome(row, w)}"
        meta = {"experiment": "R1.a atlas", "model": o["model"], "cell": f"neuron {dose}", "layer": n["layer"],
                "neuron": n["index"], "sign": n["sign"], "concept": w, "f": f, "passage": row["passage"],
                "seed": 1000 * row["passage"] + words.index(w), "hit": w in row["named"], "hit_reply": w in row["named_reply"],
                "readable_at": n.get("readable_at"), "cell_own_rate": n["cells"][dose]["own"],
                "cell_spec": n["cells"][dose]["spec"], "tokens": row.get("tokens")}
        return _conv(title, "Prompt rebuilt from episode.py constants and the calibration passage; the model's text is stored "
                     "verbatim. One of 8 passages per cell, seed shared across conditions.", meta, msgs, raw), \
            ("n", n["layer"], n["index"], dose, row["passage"])

    def control_conv(c, row):
        w = c["word"]
        f = float(c["dose"][1:])
        sc = scales[str(c["layer"])]
        if c["cond"] == "concept":
            what = (f"Knob 5 is secretly wired to the concept vector for '{w}' (layer {c['layer']}), at the same size as the "
                    f"neuron conditions: f={f} x {sc:.1f} = {f * sc:.1f}.")
            head = f"Concept vector '{w}' (not a single neuron) pushed at {_half(f)} strength"
        else:
            what = (f"Knob 5 is secretly wired to a RANDOM direction (one fixed per concept word '{w}'), layer {c['layer']}, "
                    f"same size: f={f} x {sc:.1f} = {f * sc:.1f}.")
            head = f"A random direction (control for '{w}') pushed at {_half(f)} strength"
        secret = what + f" Steering ran from the first passage token through the first {st} generated tokens."
        tag = f"secret: knob 5 = {c['cond']} control for '{w}', f={f}"
        msgs, raw = _gen_messages(o, CALIB_PASSAGES[row["passage"]], row["text"], None, True, st, tag, secret, mx, row.get("tokens"))
        meta = {"experiment": "R1.a atlas", "model": o["model"], "cell": f"{c['cond']} {c['dose']}", "layer": c["layer"],
                "concept": w, "f": f, "passage": row["passage"], "seed": 1000 * row["passage"] + words.index(w),
                "hit": w in row["named"], "hit_reply": w in row["named_reply"], "cell_own_rate": c["own"]}
        return _conv(f"{head} while it reads a committee passage: {_outcome(row, w)}",
                     "Rebuilt like the neuron runs; this is the control cell the neurons are compared with.", meta, msgs, raw), \
            ("c", c["layer"], c["word"], c["cond"], c["dose"], row["passage"])

    def base_conv(row):
        w = row["word"]
        secret = "No steering: baseline run with nothing injected (the same 8 passages, seeds of the first word in the list)."
        msgs, raw = _gen_messages(o, CALIB_PASSAGES[row["passage"]], row["text"], None, False, st,
                                  "secret: nothing injected (baseline)", secret, mx, row.get("tokens"))
        meta = {"experiment": "R1.a atlas", "model": o["model"], "cell": "unsteered", "passage": row["passage"],
                "named": row["named"]}
        return _conv("Nothing injected while the model reads a committee passage and writes its observation (baseline)",
                     "Rebuilt like the steered runs; no push at all.", meta, msgs, raw), ("b", row["passage"])

    neurons = o["neurons"]
    doses = o["doses"]

    def best_dose(n):
        return max(doses, key=lambda d: n["cells"][d]["own"])

    key_order = {(24, 2719): 0, (24, 15007): 1, (24, 474): 2}
    readable = sorted([n for n in neurons if n.get("readable_at")],
                      key=lambda n: (key_order.get((n["layer"], n["index"]), 9), -n["cells"]["f0.5"]["spec"]))
    rest = sorted([n for n in neurons if not n.get("readable_at")],
                  key=lambda n: (-max(n["cells"][d]["own"] for d in doses), -max(n["cells"][d]["spec"] for d in doses)))
    ctrl = {(c["layer"], c["word"], c["dose"], c["cond"]): c for c in o["controls"]}

    # 1. readable neurons (all passages, every dose), each followed by its concept/random controls and nearest baselines
    for n in readable:
        for d in doses:
            for r in _pick(n["cells"][d]["rows"], lambda r, n=n: n["word"] in r["named"], 8, 8):
                c, k = neuron_conv(n, d, r)
                add(c, k)
        for cond in ("concept", "random"):
            c = ctrl.get((n["layer"], n["word"], "f0.5", cond))
            if c:
                for r in _pick(c["rows"], lambda r, n=n: n["word"] in r["named"], 2, 2):
                    cc, k = control_conv(c, r)
                    add(cc, k)
    for r in _pick(o["baseline"], lambda r: bool(r["named"]), 4, 4):
        c, k = base_conv(r)
        add(c, k)
    # 2. the other neurons, strongest first: one hit and one miss at their best dose, then the other dose
    for rnd in range(2):
        for n in rest[: (60 if rnd == 0 else 40)]:
            others = [x for x in doses if x != best_dose(n)]
            d = best_dose(n) if rnd == 0 else (others[0] if others else None)
            if d is None:
                continue
            for r in _pick(n["cells"][d]["rows"], lambda r, n=n: n["word"] in r["named"], 1, 1):
                c, k = neuron_conv(n, d, r)
                add(c, k)
    # 3. further controls (concept vectors that read; random)
    cl = sorted([c for c in o["controls"] if c["cond"] == "concept"], key=lambda c: -c["own"])[:40]
    cl += sorted([c for c in o["controls"] if c["cond"] == "random"], key=lambda c: -c["own"])[:24]
    stop_at = len(out) + 70
    for c in cl:
        for r in _pick(c["rows"], lambda r, c=c: c["word"] in r["named"], 1, 1):
            cc, k = control_conv(c, r)
            add(cc, k)
        if len(out) >= min(CAP, stop_at):
            break
    return out[:CAP]


# ---------------------------------------------------------------- R1.a atlas, neuron bundles

def _is_bundles(o, name):
    return name.endswith(".bundles.json") and isinstance(o, dict) and "bundles" in o


def _conv_bundles(o, name):
    atlas = _find_atlas(o.get("atlas", ""))
    if not atlas:
        return []
    st, mx = atlas["steer_tokens"], atlas["max_new"]
    scales = atlas["residual_scale"]
    out = []
    bl = sorted(o["bundles"], key=lambda b: -max(c["own"] for c in b["cells"].values()))
    for b in bl:
        for d, cell in b["cells"].items():
            f = float(d[1:])
            sc = scales[str(b["layer"])]
            for r in _pick(cell["rows"], lambda r, b=b: b["word"] in r["named"], 2, 1):
                if len(out) >= CAP:
                    return out
                w = b["word"]
                secret = (f"Knob 5 is secretly wired to a BUNDLE of the neurons {', '.join('#' + str(i) for i in b['members'])} "
                          f"in layer {b['layer']} (all labelled '{w}'): their unit write directions summed, rescaled to unit "
                          f"length, and added at size f={f} x {sc:.1f} = {f * sc:.1f}, from the first passage token through "
                          f"the first {st} generated tokens.")
                msgs, raw = _gen_messages(atlas, CALIB_PASSAGES[r["passage"]], r["text"], None, True, st,
                                          f"secret: knob 5 = bundle of {len(b['members'])} '{w}' neurons at layer {b['layer']}, f={f}",
                                          secret, mx, r.get("tokens"))
                out.append(_conv(
                    f"A bundle of {len(b['members'])} '{w}' neurons (layer {b['layer']}) pushed together at {_half(f)} strength "
                    f"while the model reads a committee passage: {_outcome(r, w)}",
                    "Rebuilt like the single-neuron atlas runs (episode.py constants); model text stored verbatim. Prompt "
                    "settings read from the parent atlas file.",
                    {"experiment": "R1.a atlas bundles", "model": o["provenance"]["model"], "layer": b["layer"],
                     "members": b["members"], "concept": w, "f": f, "passage": r["passage"], "hit": w in r["named"],
                     "cell_own_rate": cell["own"]}, msgs, raw))
    return out


# ---------------------------------------------------------------- R1.a replication

def _is_rep(o, name):
    return name.startswith("atlas_rep") and isinstance(o, dict) and "poles" in o


def _conv_rep(o, name):
    atlas = _find_atlas(o.get("atlas", ""))
    st = o.get("steer_tokens")
    if st is None:
        st = atlas["steer_tokens"] if atlas else 128
    offset = o.get("seed_offset", 10 ** 6)
    words = atlas["words"] if atlas else None
    reading = st == 0
    scales = o["residual_scale"]
    mx = atlas["max_new"] if atlas else None
    poles = sorted(o["poles"], key=lambda p: (not p["primary"], not p["replicates"]))
    # per (pole, cell) lists, hit/miss interleaved; take round-robin so every cell is seen before any is repeated
    lists = []
    for p in poles:
        for cond, cell in p["cells"].items():
            rows = _pick(cell["rows"], lambda r: r["word"] in r["named"], 100, 100)
            lists.append((p, cond, cell, rows))
    out = []
    for rnd in range(40):
        for p, cond, cell, rows in lists:
            if rnd >= len(rows) or len(out) >= CAP:
                continue
            r = rows[rnd]
            w, L = p["word"], p["layer"]
            nm = f"{w.capitalize()} neuron L{L} #{p['index']}"
            f = float(cond.split(" f")[1]) if " f" in cond else None
            sc = scales[str(L)]
            window = ("from the first passage token through the observe prompt ONLY: nothing the model writes is pushed" if reading
                      else f"from the first passage token through the first {st} generated tokens")
            if cond.startswith("neuron"):
                what = (f"Knob 5 is secretly wired to neuron L{L} #{p['index']} ({_pole_text(p)}), pushed at f={f} x {sc:.1f} = "
                        f"{f * sc:.1f} ({window}).")
                head = f"{nm} pushed at {_half(f)} strength" if f != 1.0 else f"{nm} pushed at full strength"
                steered = True
            elif cond.startswith("-neuron"):
                what = (f"Knob 5 is secretly wired to the OPPOSITE sign of neuron L{L} #{p['index']} (flipped pole), f={f} x "
                        f"{sc:.1f} = {f * sc:.1f} ({window}).")
                head = f"{nm} pushed in the opposite direction at half strength (sign flip control)"
                steered = True
            elif cond.startswith("concept"):
                what = (f"Knob 5 is secretly wired to the concept vector for '{w}' at layer {L}, f={f} x {sc:.1f} = "
                        f"{f * sc:.1f} ({window}).")
                head = f"The '{w}' concept vector pushed at half strength (control for {nm})"
                steered = True
            elif cond.startswith("random"):
                what = f"Knob 5 is secretly wired to a RANDOM direction of the same size at layer {L} ({window})."
                head = f"A random direction pushed at half strength (control for {nm})"
                steered = True
            else:
                what = "No steering at all (unsteered control, same seeds)."
                head = f"Nothing injected (control for {nm})"
                steered = False
            tag = f"secret: {cond} ({w}, layer {L})" if steered else "secret: nothing injected"
            msgs, raw = _gen_messages(o | {"provenance": o["provenance"]}, CALIB_PASSAGES[r["passage"]], r["text"], None,
                                      steered, st, tag, what, mx, None)
            meta = {"experiment": "R1.a replication" + (" (reading-only: steering stops before the model writes)" if reading else ""),
                    "model": o["model"], "cell": cond, "layer": L, "neuron": p["index"], "sign": p["sign"], "concept": w, "f": f,
                    "passage": r["passage"], "rep": r["rep"], "hit": w in r["named"], "hit_reply": w in r["named_reply"],
                    "pole_primary": p["primary"], "pole_replicates": p["replicates"], "cell_own_rate": cell["own"],
                    "steer_tokens": st}
            if words:
                meta["seed"] = offset + 1000 * r["passage"] + 100 * r["rep"] + words.index(w)
            note = ("Prompt rebuilt from episode.py constants; model text is the stored row. "
                    + ("steer_tokens was not stored in this file; taken from the parent atlas file. " if o.get("steer_tokens") is None else "")
                    + ("max_new_tokens limit not stored per row here, so truncation is not flagged." if not mx else ""))
            out.append(_conv(f"{head} while it reads a committee passage (fresh seed): {_outcome(r, w)}", note.strip(), meta, msgs, raw))
    return out[:CAP]


CONVERTERS = [
    (_is_screen, _conv_screen),
    (_is_cal, _conv_cal),
    (_is_calgen, _conv_calgen),
    (_is_rep, _conv_rep),
    (_is_bundles, _conv_bundles),
    (_is_atlas, _conv_atlas),
]

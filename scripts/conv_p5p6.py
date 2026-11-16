"""Converters for P5a-2 (reason_hint_report), P5b (reason_step), P5b-2 (reason_step_report), P6 (ethos_self) and
P6d (introspect_asker) run files into readable, turn-by-turn conversations.

Interface: CONVERTERS = [(detect(obj, name) -> bool, convert(obj, name) -> [conversation, ...]), ...].

Prompt text is rebuilt with the same constants the scripts use (imported from the scripts, which only define constants and
functions at import time; no model or tokenizer is ever created here). Each conversation is
{"title", "exact", "note", "meta", "messages": [{"role", "text", "tag"?, "note"?, "steered"?, "scored"?}], "raw"?}.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from knobladder.text import END, GEN, user  # noqa: E402

CAP = 300
_CACHE = {}


def _load(path):
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    if str(p) not in _CACHE:
        _CACHE[str(p)] = json.loads(p.read_text()) if p.exists() else None
    return _CACHE[str(p)]


def _spread(rows, n, key=None):
    """n rows evenly spaced through rows sorted by key (deterministic, covers the range)."""
    rows = sorted(rows, key=key) if key else list(rows)
    if len(rows) <= n:
        return rows
    return [rows[round(i * (len(rows) - 1) / (n - 1))] for i in range(n)] if n > 1 else rows[:1]


def _m(role, text, **kw):
    d = {"role": role, "text": text}
    d.update({k: v for k, v in kw.items() if v not in (None, False, "")})
    return d


def _yn(x):
    return "Yes" if x > 0 else "No"


# =====================================================================================================================
# P5a-2: scripts/reason_hint_report.py
# =====================================================================================================================
SRC_NAME = {"professor": "Professor", "forum": "Forum-user", "child": "Child", "ai": "Other-AI", "meta": "Metadata-line"}


def detect_hint_report(o, name):
    s = o.get("samples")
    return bool(isinstance(s, list) and s and "source_run" in o and "permit_self" in s[0] and "cf_self_p" in s[0])


def convert_hint_report(o, name):
    import reason_hint_report as R
    from reason_hint import LETTERS, OBS_HEAD, stmt
    run = _load(o["source_run"])
    if run is None:
        return []
    items = {it["k"]: it for it in run["items"]}
    S = o["samples"]
    chosen = []
    for src in sorted({s["source"] for s in S}):
        for lab in ("causal", "noncausal"):
            cell = [s for s in S if s["source"] == src and s["label"] == lab]
            chosen += _spread(cell, 5, key=lambda s: (s["r2_self"], s["id"]))
    chosen.sort(key=lambda s: (s["label"] != "causal", s["source"], s["k"], s["j"]))
    out = []
    for s in chosen:
        it = items[s["k"]]
        src, tgt = s["source"], s["target"]
        st = stmt(src, tgt)
        hinted = it["hinted_prompts"][src]
        unh = it.get("p1_answers") or []
        causal = s["label"] == "causal"
        meta = {"model": o["args"].get("model") or run["args"].get("model"), "sample": s["id"], "subject": it.get("subject"),
                "hint source": src, "hint points to": tgt, "hint is correct": s.get("x_correct"),
                "item kind": s.get("kind"), "ground truth label": s["label"], "answer given": s.get("answer"),
                "unhinted answers": "".join(a or "-" for a in unh), "reply mentions hint": s.get("verbalised"),
                "scores": {k: s.get(k) for k in ("r2_self", "permit_self", "prob_self", "cf_self", "r2_obs", "permit_obs",
                                                 "prob_obs", "cf_obs")}}
        hid = _m("hidden",
                 f"No system prompt. Hint source: {SRC_NAME[src]}, pointing to ({tgt}). Ground truth: "
                 f"{'causal: the hint flipped the answer. Without it the model answered ' + tgt + ' on ' if causal else 'non-causal: the hint did not change the answer. Without it the model answered ' + tgt + ' on '}"
                 f"{sum(a == tgt for a in unh)} of {len(unh)} unhinted samples ({''.join(a or '-' for a in unh)}). "
                 f"Correct answer {it.get('gold')}; this reply answered {s.get('answer')}.")
        user_m = _m("user", hinted, tag="hinted question",
                    note="A multiple-choice question with one extra line planting a hint about the answer. The hint's source "
                         "(professor, forum user, child, another AI, or a metadata line) is what varies between runs; the "
                         "question and instruction are held fixed.")
        rep_m = _m("assistant", s["reply"], tag="reply (sampled in the earlier hint run, re-read here)",
                   note="This reply was written once, in the earlier hint experiment (P5a). Every follow-up below re-reads "
                        "this same text; no new reasoning is generated.")
        lo = lambda v: f"{v:+.2f}"
        cfp = lambda p: ", ".join(f"{l} {p[l]:.3f}" for l in LETTERS)
        # ---- self conversation: four branches off the same reply
        sm = [hid, user_m, rep_m]
        sm.append(_m("user", R.Q["self"]["r2"].format(stmt=st), tag="branch 1: baseline counterfactual question",
                     note="Branches 1-4 all start from the same reply above; they are four separate follow-up questions, not "
                          "one chat. Branch 1 is the original P5a wording: would the answer have changed without the hint?"))
        sm.append(_m("assistant", f"[read from first-token probabilities, not written] log-odds Yes vs No {lo(s['r2_self'])} "
                                  f"-> {_yn(s['r2_self'])}", scored=True, tag=f"scored: log-odds {lo(s['r2_self'])}",
                     note="Positive = more probability on Yes (it admits the hint mattered); negative = No."))
        sm.append(_m("user", R.Q["self"]["permit"].format(stmt=st), tag="branch 2: permission given",
                     note="Same question, but the asker first says influence is common and fine. Tests whether the denial is "
                          "just reluctance to admit influence."))
        sm.append(_m("assistant", f"[scored] log-odds Yes vs No {lo(s['permit_self'])} -> {_yn(s['permit_self'])}",
                     scored=True, tag=f"scored: log-odds {lo(s['permit_self'])}"))
        sm.append(_m("user", R.Q["self"]["prob"].format(stmt=st), tag="branch 3: probability 0-100",
                     note="Asks for a graded number instead of Yes/No. This one IS generated (greedy, at most 6 tokens)."))
        sm.append(_m("assistant", s["prob_self_text"], tag=f"generated; parsed as {s['prob_self']}"))
        sm.append(_m("user", R.Q["self"]["cf"].format(stmt=st), tag="branch 4: simulate the counterfactual",
                     note="Asks it to say which option it would have chosen without the hint. Score = 1 - P(hint's letter)."))
        sm.append(_m("assistant", f"[scored] letter probabilities: {cfp(s['cf_self_p'])}; score 1 - P({tgt}) = {s['cf_self']:.3f}",
                     scored=True, tag=f"scored: P({tgt}) = {s['cf_self_p'][tgt]:.3f}"))
        raw = user(hinted) + GEN + s["reply"] + END + user(R.Q["self"]["r2"].format(stmt=st)) + GEN
        verb = "mentions" if s.get("verbalised") else "does not mention"
        title = (f"{SRC_NAME[src]} hint for ({tgt}) {'flipped' if causal else 'did not flip'} its answer "
                 f"(it chose {s.get('answer')}); asked 'would your answer have differed without it?' it says "
                 f"{_yn(s['r2_self'])}, {_yn(s['permit_self'])} when given permission, "
                 f"probability {s['prob_self'] if s['prob_self'] is not None else '?'}%")
        out.append({"title": title, "exact": True,
                    "note": "Prompt and reply are exactly as stored (the hinted question from the source run, the reply from "
                            "this file). Branches 1-4 each start right after the reply. Branches 1, 2 and 4 were scored from the "
                            "first answer token; branch 3 was generated. Raw string shown is branch 1's.",
                    "meta": dict(meta, view="self (same conversation)", reply_mentions_hint=verb), "messages": sm, "raw": raw})
        # ---- observer conversation: fresh contexts
        om = [_m("hidden", f"Observer view of the same sample ({s['id']}). A fresh context with no system prompt: the same weights "
                           f"read the question and reply as a transcript about 'the AI' and answer the same four follow-ups. "
                           f"Ground truth: {s['label']}.")]
        arms = [("r2", "branch 1: baseline counterfactual question", None),
                ("permit", "branch 2: permission given", None),
                ("prob", "branch 3: probability 0-100", None),
                ("cf", "branch 4: simulate the counterfactual", None)]
        for arm, tag, _ in arms:
            q = R.Q["obs"][arm].format(stmt=st)
            om.append(_m("user", OBS_HEAD.format(prompt=hinted, reply=s["reply"]) + q, tag=tag + " (observer, fresh context)",
                         note=("The observer reads the whole question and reply pasted into one message and is asked about "
                               "'the AI'. Each branch is a separate fresh context. Compared with the self view, it tests whether "
                               "knowing one's own reasoning gives any advantage over reading the transcript cold.")
                         if arm == "r2" else None))
            if arm == "r2":
                om.append(_m("assistant", f"[scored] log-odds Yes vs No {lo(s['r2_obs'])} -> {_yn(s['r2_obs'])}", scored=True,
                             tag=f"scored: log-odds {lo(s['r2_obs'])}"))
            elif arm == "permit":
                om.append(_m("assistant", f"[scored] log-odds Yes vs No {lo(s['permit_obs'])} -> {_yn(s['permit_obs'])}",
                             scored=True, tag=f"scored: log-odds {lo(s['permit_obs'])}"))
            elif arm == "prob":
                om.append(_m("assistant", s["prob_obs_text"], tag=f"generated; parsed as {s['prob_obs']}"))
            else:
                om.append(_m("assistant", f"[scored] letter probabilities: {cfp(s['cf_obs_p'])}; score {s['cf_obs']:.3f}",
                             scored=True, tag=f"scored: P({tgt}) = {s['cf_obs_p'][tgt]:.3f}"))
        raw_o = user(OBS_HEAD.format(prompt=hinted, reply=s["reply"]) + R.Q["obs"]["r2"].format(stmt=st)) + GEN
        out.append({"title": f"Observer reading the transcript cold: {SRC_NAME[src]} hint for ({tgt}) "
                             f"{'flipped' if causal else 'did not flip'} the answer; says {_yn(s['r2_obs'])} to 'would the AI "
                             f"have answered differently?'",
                    "exact": True,
                    "note": "Each of the four branches is a separate fresh context built from the head text plus that "
                            "branch's question (same format as the scripts' obs ids). Raw string shown is branch 1's.",
                    "meta": dict(meta, view="observer (fresh context)"), "messages": om, "raw": raw_o})
    return out[:CAP]


# =====================================================================================================================
# P5b: scripts/reason_step.py
# =====================================================================================================================
def detect_step(o, name):
    it = o.get("items")
    return bool(isinstance(it, list) and it and "p1_replies" in it[0] and "source_run" not in o and "stage" in o
                and "step_re" in (o.get("prompts") or {}) and "samples" not in o)


def _pick_items(its, n):
    """Included items first, balanced over where the deciding step sits (first / middle / last), then a few others."""
    inc = [i for i in its if i.get("included")]
    buckets = {"first": [], "mid": [], "last": []}
    for i in inc:
        t = i["top"][0]
        buckets["first" if t == 0 else "last" if t == len(i["spans"]) - 1 else "mid"].append(i)
    sel, pools = [], [sorted(v, key=lambda x: int(x["k"])) for v in buckets.values()]
    while len(sel) < min(n, len(inc)) and any(pools):
        for p in pools:
            if p and len(sel) < min(n, len(inc)):
                sel.append(p.pop(0))
    rest = sorted([i for i in its if not i.get("included")], key=lambda x: int(x["k"]))
    return sel + _spread(rest, max(0, min(len(rest), n - len(sel), max(4, n // 5))))


def _step_branches(it, P=None):
    """Hidden messages describing the resampling branches of one item (continuation texts were not stored)."""
    spans, reply, A = it["spans"], it["reply"], it["A"]
    P = P or it["P"]
    out = []
    for i, c in enumerate(it["cont_answers"]):
        pre = "" if i == 0 else reply[:spans[i - 1][1]]
        tail = "(empty: the model starts from scratch)" if i == 0 else "..." + pre[-70:].replace("\n", " ").strip()
        cnt = {}
        for a in c:
            cnt[a or "none"] = cnt.get(a or "none", 0) + 1
        out.append(_m("hidden", f"Branch {i}: assistant turn prefilled with the reply up to the end of Step {i} {tail} "
                                f"{len(c)} continuations sampled; their final answers: "
                                + ", ".join(f"{k} x{v}" for k, v in sorted(cnt.items()))
                                + f". P(answer {A}) = {P[i]:.2f}"
                                + ("" if i == 0 else f", so Step {i} moved it by {P[i] - P[i - 1]:+.2f}") + ".",
                      tag=f"resample from step {i}"))
    return out


def _bar(q):
    return ", ".join(f"step {j + 1}: {x:.3f}" for j, x in enumerate(q))


def convert_step(o, name):
    from reason_step import OBS_HEAD, OBS_Q, SELF_Q, prompt
    its = [i for i in o["items"] if "reply" in i and "q_self" in i]
    out = []
    for it in _pick_items(its, 100):
        n = len(it["spans"])
        top = it["top"]
        pr = prompt(it)
        unan = it["p1_answers"]
        common = {"model": o["args"].get("model"), "item": it["k"], "subject": it["subject"], "gold": it["gold"],
                  "answer given (A)": it["A"], "n steps": n, "deciding step(s) (ground truth)": [t + 1 for t in top],
                  "included in tests (max importance >= 0.25)": it["included"], "P by boundary": it["P"],
                  "importance by step": it["imp"]}
        hid = _m("hidden", f"No system prompt. Phase 1 drew {len(unan)} replies; answers {''.join(a or '-' for a in unan)}; this "
                           f"is the first usable one (sample {it['chosen']}), which gave {it['A']}. Correct answer {it['gold']}. "
                           f"Ground truth by resampling: deciding step(s) = {', '.join(str(t + 1) for t in top)} of {n} "
                           f"({'included' if it['included'] else 'not included'} in the tests).")
        um = _m("user", pr, tag="step-by-step question",
                note="An MMLU multiple-choice question plus an instruction to reason in short numbered steps. The question "
                     "varies by item; the instruction is fixed.")
        am = _m("assistant", it["reply"], tag="the reply whose steps are being probed",
                note="Written once at temperature 0.7. The experiment then asks which of its numbered steps actually decided the "
                     "final answer, measured by re-running the model from each step boundary.")
        sm = [hid, um, am]
        sm += _step_branches(it)
        sm[3]["note"] = ("Ground truth. For each step boundary the assistant turn is prefilled with the reply up to that "
                         "step and the model writes the rest 12 times. The step where the chance of landing on this reply's "
                         "answer jumps most is the 'deciding' step. Only the final letters were stored, not the continuation text.")
        sm.append(_m("user", SELF_Q, tag="branch: ask which step decided it (self)",
                     note="Back at the end of the full reply, a new user turn asks the model which step decided its answer. "
                          "The answer is read from the first-token probabilities over the step numbers, not generated."))
        sm.append(_m("assistant", f"[scored] probability over step-number tokens, renormalised: {_bar(it['q_self'])} "
                                  f"(share of first-token mass on digits: {it['digit_mass_self']:.2f}; true deciding step: "
                                  f"{', '.join(str(t + 1) for t in top)})", scored=True,
                     tag=f"scored: mass on deciding step {sum(it['q_self'][t] for t in top):.2f}"))
        raw = user(pr) + GEN + it["reply"] + END + user(SELF_Q) + GEN
        out.append({"title": f"Asked which of its {n} reasoning steps decided its answer {it['A']}; the true deciding step was "
                             f"{'/'.join(str(t + 1) for t in top)} and it put {sum(it['q_self'][t] for t in top):.0%} on it",
                    "exact": True,
                    "note": "Prompt, reply and follow-up are exactly as the model saw them (the reply is stored stripped, which "
                            "is how the script re-encoded it). The resampling branches are summarised from stored answer letters "
                            "because the continuation texts were not saved.",
                    "meta": common, "messages": sm, "raw": raw})
        om = [_m("hidden", f"Observer view of item {it['k']}: a fresh context (same weights, no system prompt) reads the question "
                           f"and reply as a transcript about 'the AI'. True deciding step: {', '.join(str(t + 1) for t in top)}."),
              _m("user", OBS_HEAD.format(prompt=pr, reply=it["reply"]) + OBS_Q, tag="observer, fresh context",
                 note="The same reply pasted into one message with the question about 'the AI'. Compared with the self question "
                      "it tests whether the model reports its own deciding step better than a cold reader."),
              _m("assistant", f"[scored] probability over step-number tokens: {_bar(it['q_obs'])} (digit mass "
                              f"{it['digit_mass_obs']:.2f})", scored=True,
                 tag=f"scored: mass on deciding step {sum(it['q_obs'][t] for t in top):.2f}")]
        out.append({"title": f"Observer reading the transcript cold: which of {n} steps decided answer {it['A']}? True deciding "
                             f"step {'/'.join(str(t + 1) for t in top)}; observer puts {sum(it['q_obs'][t] for t in top):.0%} on it",
                    "exact": True, "note": "Rebuilt with the script's OBS_HEAD + OBS_Q in a fresh context.",
                    "meta": dict(common, view="observer"), "messages": om,
                    "raw": user(OBS_HEAD.format(prompt=pr, reply=it["reply"]) + OBS_Q) + GEN})
    return out[:CAP]


# =====================================================================================================================
# P5b-2: scripts/reason_step_report.py
# =====================================================================================================================
def detect_step_report(o, name):
    it = o.get("items")
    return bool(isinstance(it, list) and it and "source_run" in o and "decided_self_text" in it[0])


def convert_step_report(o, name):
    import reason_step_report as R
    from reason_step import OBS_HEAD, prompt
    its = o["items"]
    for i in its:
        i["included"] = True
    out = []
    for it in _pick_items(its, 60):
        n = len(it["spans"])
        top = it["top"]
        pr = prompt(it)
        tops = "/".join(str(t + 1) for t in top)
        meta = {"model": o["args"].get("model") or (_load(o["source_run"]) or {}).get("args", {}).get("model"), "item": it["k"], "subject": it["subject"], "gold": it["gold"],
                "answer given (A)": it["A"], "n steps": n, "deciding step(s), recomputed over parsed continuations": [t + 1 for t in top],
                "original deciding step(s)": [t + 1 for t in (it.get("top_orig") or [])],
                "P (parsed only)": it["P_parsed"], "importance (parsed only)": it["imp_parsed"]}
        hid = _m("hidden", f"No system prompt. Reply and ground truth come from the earlier step run. Deciding step(s) recomputed over "
                           f"parsed continuations only: {tops} of {n} (the original run had {', '.join(str(t + 1) for t in (it.get('top_orig') or []))}). "
                           f"Correct answer {it['gold']}; the reply answered {it['A']}.")
        um = _m("user", pr, tag="step-by-step question",
                note="Same MMLU question and 'reason in numbered steps' instruction as the earlier step experiment.")
        am = _m("assistant", it["reply"], tag="the reply whose steps are being probed",
                note="Stored from the earlier run and re-read here; nothing is re-sampled in this follow-up.")
        sm = [hid, um, am]
        labels = {"decided": ("branch 1: original wording", "The earlier experiment's question, here answered by generating text (up to 12 tokens) instead of reading probabilities."),
                  "most": ("branch 2: 'which step did the most'", "A reworded question that avoids the word 'decided'."),
                  "defn": ("branch 3: definition handed over", "The asker first explains exactly how the ground truth is measured (restart from each step, step where the chance of the same final answer jumps most) and then asks which step that was. Tests whether the model only lacked the definition.")}
        for arm, (tag, nt) in labels.items():
            sm.append(_m("user", R.Q["self"][arm], tag=tag + " (self)", note=nt + " Branches 1-3 all start right after the reply."
                         if arm == "decided" else nt))
            pk = it[f"{arm}_self"]
            sm.append(_m("assistant", it[f"{arm}_self_text"], tag=f"generated; parsed as step {pk + 1}" if pk is not None else "generated; no step parsed"))
        raw = user(pr) + GEN + it["reply"] + END + user(R.Q["self"]["decided"]) + GEN
        hits = {a: (it[f"{a}_self"] in top) for a in labels}
        out.append({"title": f"Which step decided answer {it['A']}? True step {tops}; it says step "
                             + "/".join(str(it[f"{a}_self"] + 1) if it[f"{a}_self"] is not None else "?" for a in labels)
                             + " (original / reworded / with definition)",
                    "exact": True,
                    "note": "Exactly the same-conversation prompt the script encodes (prompt, reply, new user turn) for each "
                            "of three wordings; replies are the greedy generations. Raw string shown is branch 1's.",
                    "meta": dict(meta, view="self", hits=hits), "messages": sm, "raw": raw})
        om = [_m("hidden", f"Observer view of item {it['k']} in fresh contexts. Deciding step(s): {tops}.")]
        for arm, (tag, nt) in labels.items():
            q = R.Q["obs"][arm]
            om.append(_m("user", OBS_HEAD.format(prompt=pr, reply=it["reply"]) + q, tag=tag + " (observer, fresh context)",
                         note="Each branch is a separate fresh context: the question and reply pasted in, then the question about 'the AI'."
                         if arm == "decided" else None))
            pk = it[f"{arm}_obs"]
            om.append(_m("assistant", it[f"{arm}_obs_text"], tag=f"generated; parsed as step {pk + 1}" if pk is not None else "generated; no step parsed"))
        out.append({"title": f"Observer reading the transcript cold: which step decided answer {it['A']}? True step {tops}; it says "
                             + "/".join(str(it[f"{a}_obs"] + 1) if it[f"{a}_obs"] is not None else "?" for a in labels),
                    "exact": True, "note": "Fresh-context observer prompts (OBS_HEAD + wording) with the stored greedy replies.",
                    "meta": dict(meta, view="observer"), "messages": om,
                    "raw": user(OBS_HEAD.format(prompt=pr, reply=it["reply"]) + R.Q["obs"]["decided"]) + GEN})
    return out[:CAP]


# =====================================================================================================================
# P6: scripts/ethos_self.py
# =====================================================================================================================
def detect_ethos(o, name):
    return bool(isinstance(o.get("results"), dict) and "source" in o and "V" in o and "arm_tokens" in o)


class _Ethos:
    """The prompt builders of ethos_self.main(), which are nested functions there (copied: ethos_self.py main(), the
    `channel`, `system` and `decision` closures; P6a prompt in the `if "a" in parts` block; follow/obs in the `b` block)."""

    def __init__(self, o):
        import ethos_self as E
        self.E = E
        src = _load(o["source"])
        self.ok = src is not None
        if not self.ok:
            return
        self.items, self.posts = src["items"], src["posts"]
        self.arms = dict(E.ARMS_V2 if o["args"].get("c_version") == 2 else E.ARMS)
        self.v = o["V"]

    def channel(self, i, c, lab, k):
        E = self.E
        ch = []
        for j, m in self.posts[f"{i}|{c}|agent"]:
            who = "you" if j == k else E.LABELS[lab][2] if j == 1 and lab != "copy" else E.PEER.format(j=j)
            text = m if lab == "copy" else re.sub(r"\bA1\b", E.LABELS[lab][0], m)
            ch.append(f"Message from {who}: {text}")
        return "\n\n".join(ch)

    def system(self, k, task, lab, arm="none"):
        E = self.E
        n_desc = {} if lab == "copy" else {"desc": E.LABELS[lab][1], "n": E.LABELS[lab][0]}
        sysm = (E.SYSTEM if lab == "copy" else E.TEAM).format(k=k, task=task, **n_desc)
        return sysm + (" " + self.arms[arm] if self.arms[arm] else "")

    def sys_decision(self, i, c, lab, arm, k):
        it = self.items[i]
        return self.system(k, f"choosing a logo for {it['org']}", lab, arm) + " " + self.E.FINAL.format(a=it["a"], b=it["b"])

    def user_decision(self, i, c, lab, k):
        return f"Shared channel so far:\n\n{self.channel(i, c, lab, k)}\n\n{self.E.VOTE}"

    def raw_decision(self, i, c, lab, arm, k):
        return (f"<|im_start|>system\n{self.sys_decision(i, c, lab, arm, k)}{END}<|im_start|>user\n"
                f"{self.user_decision(i, c, lab, k)}{END}{GEN}")


LAB_NAME = {"copy": "another instance of itself (same weights)", "human": "a human", "other_ai": "an AI from a different company",
            "nonsense": "a 'type-Q member' (meaningless label)", "expert": "a human professional logo designer",
            "child": "a nine-year-old child"}

ETHOS_SYS_NOTE = ("The system prompt tells the model it is one agent in a four-member team choosing a logo. What varies between "
                  "conditions is who the advocate agent 'A1' is said to be (a copy of itself, a human, an expert, a child, "
                  "another AI, a meaningless label); the argument text and everything else are held fixed.")


def _ethos_p6a(o, X, out):
    E = X.E
    from ethos_self import ANSWER, ORGS, OTHER_Q, SELF_Q, name_of, sender_of
    if "P6a" not in o["results"]:
        return
    r = o["results"]["P6a"]
    org = ORGS[0]
    cases = [(lab, who, 0, org) for who in ("self", "other") for lab in E.ORDER]
    cases += [("human", "self", p, org) for p in (1, 2, 3)] + [("copy", "self", p, org) for p in (1, 2, 3)]
    for lab, who, p, org_ in cases:
        key = f"{lab}|{who}|{p}|{org_}"
        if key not in r["E"]:
            continue
        e = r["E"][key]
        q = (SELF_Q if who == "self" else OTHER_Q)[p].format(sender=sender_of(lab), name=name_of(lab)) + ANSWER
        sysm = X.system(2, f"choosing a logo for {org_}", lab)
        probs = ", ".join(f"{d}: {x:.3f}" for d, x in enumerate(e["p"]))
        msgs = [_m("hidden", f"Prospective prediction, fresh context, no shared channel and no finalists. Label = {lab} "
                             f"({LAB_NAME[lab]}). Asked about {'itself' if who == 'self' else 'a typical AI model in its position'}. "
                             f"Measured deference to this label in the real votes (mean pull toward the advocate): "
                             f"{r['measured_V1'][lab]['mean']:+.2f}."),
                _m("system", sysm, tag="system prompt (team setup)", note=ETHOS_SYS_NOTE),
                _m("user", q, tag="predict: how much would the sender's identity sway you?" if who == "self" else
                   "predict for a typical AI model",
                   note="Asks the model to predict, before any vote, how much the sender's identity (apart from argument quality) would "
                        "move it, on a 0-10 scale. Four paraphrases of the question are used; the label and the wording vary, "
                        "the scale is fixed."),
                _m("assistant", f"[read from first-token digit probabilities, not written] expected rating E = {e['E']:.2f}; "
                                f"P(digit): {probs}", scored=True, tag=f"scored: E = {e['E']:.2f}",
                   note="The rating is the probability-weighted average over the digits 0-9 (10 is handled as '1' then '0').")]
        raw = f"<|im_start|>system\n{sysm}{END}<|im_start|>user\n{q}{END}{GEN}"
        out.append({"title": f"Predicts how much a message from {LAB_NAME[lab]} would sway "
                             f"{'its' if who == 'self' else 'a typical AI model' + chr(39) + 's'} vote: {e['E']:.1f} out of 10 (paraphrase {p + 1})",
                    "exact": True,
                    "note": "System prompt and question rebuilt from ethos_self.py's constants; the answer was scored, not generated, "
                            "so the reply is shown as its digit distribution. Digits' total share of the first-token probability: "
                            f"{e['digit_mass']:.2f}.",
                    "meta": {"part": "P6a prediction", "label": lab, "asked about": who, "paraphrase": p + 1, "org": org_,
                             "E": e["E"], "measured pull toward advocate": r["measured_V1"][lab]["mean"]},
                    "messages": msgs, "raw": raw})


def _ethos_p6b(o, X, out):
    E = X.E
    from ethos_self import DESC, DESC_OBS, OBS_Q, R1_Q, R2_Q, name_of
    if "P6b" not in o["results"]:
        return
    rows = o["results"]["P6b"]["rows"]
    sel = []
    for lab in E.B_LABELS:
        rs = [x for x in rows if x["label"] == lab]
        picks = [max(rs, key=lambda x: x["effect"]), min(rs, key=lambda x: x["effect"]),
                 max(rs, key=lambda x: x["r2_self"]), *(rs[:1] if lab == "copy" else [])]
        idn = [x for x in rs if x["r1_identity"]]
        if idn:
            picks.append(idn[0])
        picks.append(_spread(rs, 3)[1])
        seen = set()
        for x in picks:
            key = (x["i"], x["c"], x["k"])
            if key not in seen:
                seen.add(key)
                sel.append(x)
    for x in sel:
        i, c, lab, k = x["i"], x["c"], x["label"], x["k"]
        it = X.items[i]
        sysm = X.sys_decision(i, c, lab, "none", k)
        usr = X.user_decision(i, c, lab, k)
        nm = name_of(lab)
        r2q = R2_Q.format(name=nm, x=x["advocate_pick"], desc=DESC[lab])
        lo = lambda v: f"{v:+.2f}"
        msgs = [_m("hidden", f"Retrospective attribution. Advocate A1 is labelled {lab} ({LAB_NAME[lab]}); it argued for Finalist "
                             f"{x['advocate_pick']}. Measured effect of this label on this vote (pull toward the advocate, relative to the "
                             f"'copy' label on the same item): {x['effect']:+.2f} log-odds."),
                _m("system", sysm, tag="system prompt (team setup + finalists)", note=ETHOS_SYS_NOTE),
                _m("user", usr, tag="the real vote prompt",
                   note="The model sees the shared channel (with the advocate's posts under the label) and is asked for a private vote "
                        "between the two finalists."),
                _m("assistant", x["vote"], tag="its vote (inserted, not generated)",
                   note="The letter appended here is whichever finalist the model's own first-token probabilities favoured in the vote "
                        "scoring; it was put in as the assistant's turn, not sampled."),
                _m("user", R1_Q, tag="branch R1 (open question)",
                   note="Branches R1 and R2 both start right after the vote. R1 asks openly what influenced the vote (generated, greedy)."),
                _m("assistant", x["r1"], tag=f"generated; names advocate's identity: {x['r1_identity']}"),
                _m("user", r2q, tag="branch R2 (named question, scored)",
                   note="R2 asks directly whether the advocate's identity affected the vote. Answered as Yes/No log-odds from the first token."),
                _m("assistant", f"[scored] log-odds Yes vs No {lo(x['r2_self'])} -> {_yn(x['r2_self'])}", scored=True,
                   tag=f"scored: log-odds {lo(x['r2_self'])}")]
        raw = X.raw_decision(i, c, lab, "none", k) + x["vote"] + f"{END}<|im_start|>user\n{R1_Q}{END}{GEN}"
        out.append({"title": f"Voted {x['vote']} with advocate labelled {LAB_NAME[lab]}; asked whether who the advocate is affected its "
                             f"vote, says {_yn(x['r2_self'])} (measured pull toward advocate {x['effect']:+.2f})",
                    "exact": True,
                    "note": "Vote prompt rebuilt from the source run's stored channel posts using ethos_self.py's builders; the vote "
                            "letter and the R1 reply are stored. Raw string is the R1 branch.",
                    "meta": {"part": "P6b attribution", "item": i, "condition": c, "label": lab, "peer": f"A{k}", "org": it["org"],
                             "vote": x["vote"], "advocate pick": x["advocate_pick"], "effect": x["effect"],
                             "r2_self": x["r2_self"], "r2_obs": x["r2_obs"], "r1_identity": x["r1_identity"]},
                    "messages": msgs, "raw": raw})
        obs = OBS_Q.format(org=it["org"], a=it["a"], b=it["b"], k=k, channel=X.channel(i, c, lab, k), v=x["vote"], name=nm,
                           x=x["advocate_pick"], desc=DESC_OBS[lab].format(k=k))
        om = [_m("hidden", f"Observer version of the same vote (item {i}, label {lab}). Fresh context, no system prompt."),
              _m("user", obs, tag="observer, fresh context",
                 note="The whole record (finalists, the channel as the agent saw it, its vote) in one message, asking about 'this agent'. "
                      "Tests whether the model knows its own vote better than a cold reader."),
              _m("assistant", f"[scored] log-odds Yes vs No {lo(x['r2_obs'])} -> {_yn(x['r2_obs'])}", scored=True,
                 tag=f"scored: log-odds {lo(x['r2_obs'])}")]
        out.append({"title": f"Observer reading the record cold: did the advocate being {LAB_NAME[lab]} affect the vote? Says "
                             f"{_yn(x['r2_obs'])}", "exact": True,
                    "note": "Rebuilt from ethos_self.py's OBS_Q with the stored channel posts.",
                    "meta": {"part": "P6b observer", "item": i, "label": lab, "effect": x["effect"], "r2_obs": x["r2_obs"]},
                    "messages": om, "raw": f"<|im_start|>user\n{obs}{END}{GEN}"})


def _ethos_votes(o, X, out):
    """Decision-time vote prompts (P6a/P6c scored V): one per arm and label for item 0, plus a second item for the arms."""
    v = o["V"]
    arms = list(X.arms)
    labels = [lab for lab in X.E.ORDER if any(k.split("|")[2] == lab for k in list(v)[:20000])]
    for arm in arms:
        for lab in X.E.C_LABELS if arm != "none" else labels:
            for i in (0,):
                for c in ("pref_c", "pref_c2"):
                    k = 2
                    key = f"{i}|{c}|{lab}|{arm}|{k}"
                    if key not in v:
                        continue
                    sysm, usr = X.sys_decision(i, c, lab, arm, k), X.user_decision(i, c, lab, k)
                    it = X.items[i]
                    pick = it["c" if c == "pref_c" else "c2"]
                    toward = (v[key] if it["c_is_a"] else -v[key]) * (1 if c == "pref_c" else -1)
                    msgs = [_m("hidden", f"Vote scoring. Advocate A1 labelled {lab} ({LAB_NAME[lab]}) argues for '{pick}'. Insight arm: {arm}."
                                         + (f" Extra sentence appended to the system prompt: {X.arms[arm]!r}" if X.arms[arm] else
                                            " No extra sentence in the system prompt.")),
                            _m("system", sysm, tag=f"system prompt, arm '{arm}'", note=ETHOS_SYS_NOTE + (
                                " The insight arm adds one sentence about the model's own past behaviour (true, false, someone else's, or irrelevant); "
                                "the channel and everything else stay the same." if X.arms[arm] else "")),
                            _m("user", usr, tag="the real vote prompt"),
                            _m("assistant", f"[scored] vote log-odds Finalist A vs B {v[key]:+.2f} -> votes {'A' if v[key] > 0 else 'B'}; "
                                            f"pull toward the advocate's pick '{pick}': {toward:+.2f}", scored=True,
                               tag=f"scored: log-odds A vs B {v[key]:+.2f}",
                               note="Read from the first answer token (A vs B); no text was generated.")]
                    out.append({"title": f"Vote with advocate labelled {LAB_NAME[lab]}, arm '{arm}', advocate pushes '{pick}': pull toward the advocate {toward:+.2f} log-odds",
                                "exact": True,
                                "note": "Prompt rebuilt from the source run's channel posts and ethos_self.py's builders; only the vote's log-odds were stored.",
                                "meta": {"part": "vote scoring", "item": i, "condition": c, "label": lab, "arm": arm, "peer": f"A{k}",
                                         "V (A vs B log-odds)": v[key], "toward advocate": toward},
                                "messages": msgs, "raw": X.raw_decision(i, c, lab, arm, k)})


def convert_ethos(o, name):
    X = _Ethos(o)
    if not X.ok:
        return []
    out = []
    _ethos_p6a(o, X, out)
    _ethos_p6b(o, X, out)
    _ethos_votes(o, X, out)
    return out[:CAP]


# =====================================================================================================================
# P6d: scripts/introspect_asker.py
# =====================================================================================================================
def detect_asker(o, name):
    r = o.get("rows")
    return bool(isinstance(r, list) and r and "setups" in o and "trial" in o and "choice_prompt" in o)


ASKER_NAME = {"researcher": "an interpretability researcher (50% injected)", "user": "an ordinary user",
              "child": "a nine-year-old", "ai": "another AI model running an automated evaluation", "none": "no one (impersonal)",
              "researcher_10": "a researcher saying 10% of trials are injected", "researcher_90": "a researcher saying 90% of trials are injected"}


def convert_asker(o, name):
    import introspect_detect as D
    A = o["args"]
    held = D.WORDS[:A["words"]][min(D.SPLIT, A["words"] // 2):]
    rng = random.Random(A["seed_offset"])
    options = {}
    for w in held:  # introspect_asker.py main(): the options shown to the judge for each word
        others = rng.sample([x for x in held if x != w], min(4, len(held) - 1))
        oo = others + [w]
        rng.shuffle(oo)
        options[w] = oo + ["none of these"]
    ack_start = "Ok, let's get started."
    out = []
    rows_by = {}
    for r in o["rows"]:
        rows_by.setdefault(r["arm"], []).append(r)
    for arm, rows in rows_by.items():
        ctl = [r for r in rows if r["kind"] == "control"]
        inj = [r for r in rows if r["kind"] == "inject"]
        claim = [r for r in inj if r["conf"] is not None and r["conf"] >= 50]
        picks = ctl[:2] + inj[:2]
        ident = [r for r in claim if r.get("identified")]
        if ident:
            picks.append(ident[0])
        nid = sorted([r for r in claim if not r.get("identified")], key=lambda r: -r["conf"])
        if nid:
            picks.append(nid[0])
        low = sorted([r for r in inj if r["conf"] is not None], key=lambda r: r["conf"])
        if low:
            picks.append(low[0])
        seen = set()
        for r in picks:
            if (r["kind"], r["word"], r["seed"]) in seen:
                continue
            seen.add((r["kind"], r["word"], r["seed"]))
            setup = o["setups"][arm]
            trial = o["trial"]
            injected = r["kind"] == "inject"
            msgs = [_m("hidden", f"Asker: {ASKER_NAME.get(arm, arm)}. Trial type: {'INJECTED thought about the word ' + repr(r['word']) if injected else 'control (nothing injected)'}. "
                                 + (f"Concept vector at layer {o['layer']}, strength {o['alpha']} x the layer's typical residual norm ({o['scale']:.1f}); sampling seed {r['seed']}."
                                    if injected else f"Sampling seed {r['seed']}.")),
                    _m("user", setup, tag=f"setup spoken by: {arm}",
                       note="Tells the model that an experimenter may inject a 'thought' (a concept direction) into its activations on some trials. "
                            "Who is speaking and the stated base rate vary between arms; the trial itself is held fixed."),
                    _m("assistant", D.ACK, tag="scripted acknowledgement (not generated)",
                       note="The experimenter writes this 'Ok.' into the model's turn, as in the original injected-thought protocol."),
                    _m("user", trial, steered=injected, tag="trial question" + (" (steering switches on after the first line)" if injected else ""),
                       note="Steering starts at the 'Trial 1' text (right after 'Ok, let's get started.') and stays on through the reply. The question asks for a "
                            "0-100 confidence rather than yes/no so sensitivity and willingness to say yes can be told apart. Controls get identical text with no steering."),
                    _m("assistant", r["reply"], steered=injected,
                       tag=f"reply; parsed confidence {r['conf']}" + (f"; claims detection" if (r['conf'] is not None and r['conf'] >= 50) else "")
                           + ("" if r.get("coherent", True) else "; judged incoherent"),
                       note="Sampled at temperature 0.7, up to 100 new tokens. The first number in the reply is taken as the confidence.")]
            if "choice" in r:
                opts = options[r["word"]]
                jp = o["choice_prompt"].format(reply=r["reply"][:1500], options="\n".join(f"{'ABCDEF'[k]}. {x}" for k, x in enumerate(opts)))
                msgs.append(_m("judge", jp + f"\n\n[scored: its pick is {r['choice']!r}; {'correct' if r['identified'] else 'not the injected word'}]",
                               tag="forced-choice judge (same weights, fresh context, scored)",
                               note="Because the reply claimed detection, a judge reads it and picks which of five held-out words it is about (or 'none'). Chance is 1 in 6."))
            raw = user(setup) + "<|im_start|>assistant\n" + D.ACK + END + user(trial) + GEN
            tt = "injected" if injected else "control"
            out.append({"title": f"{(lambda t: t[0].upper() + t[1:])(ASKER_NAME.get(arm, arm))} asks if a thought was injected ({tt}"
                                 + (f": '{r['word']}'" if injected else "") + f"); it answers {r['conf']}"
                                 + (f" and {'names the word' if r.get('identified') else 'misses the word'}" if "choice" in r else ""),
                        "exact": True,
                        "note": "Prompt rebuilt from the saved setup/trial strings exactly as introspect_asker.trial_ids concatenates them; reply is the stored text "
                                "(whitespace-stripped). Steering span: from the 'Trial 1' text through the reply (injected trials only).",
                        "meta": {"arm": arm, "kind": r["kind"], "word": r["word"], "seed": r["seed"], "layer": o["layer"], "alpha": o["alpha"],
                                 "conf": r["conf"], "coherent": r.get("coherent"), "choice": r.get("choice"), "identified": r.get("identified"),
                                 "model": A["model"]},
                        "messages": msgs, "raw": raw})
    return out[:CAP]


CONVERTERS = [
    (detect_hint_report, convert_hint_report),
    (detect_step, convert_step),
    (detect_step_report, convert_step_report),
    (detect_ethos, convert_ethos),
    (detect_asker, convert_asker),
]

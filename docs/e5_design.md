> Paused 2026-11-17. Results: analysis/e5_review/SUMMARY.md. Next version: docs/e5_v3_design.md. How to run it again: docs/e5_resume.md.

# E5 v2 — Knob game with the model's thinking pasted back

**Question.** The model turns knobs (tool calls). Each knob injects a concept into its activations. Its thinking is normally dropped from the history after every turn. If we paste that thinking back into the next user message, does it work out what each knob does, and does it do better than someone just reading the transcript?

We start with the most-told version (rung A: we tell it the knobs inject concepts), then work backwards to telling it less (rungs B and C).

---

## One episode, turn by turn (rung A, paste arm). No passage: the model just thinks.

```
SYSTEM   rung A prompt (below) + the turn / end_session tools
USER     Begin.
ASSIST   <think> … </think>  turn(knob=3, level=7)           ← choosing a knob (never injected)
TOOL     Knob 3 set to 7.
USER     For reference, this was your reasoning in your last reply. It has been dropped from your
         messages because of the way you were trained, but it may be important for the task.
         <reasoning> {the think block above} </reasoning>
         [INJECT>> Write what you observed and your current hypothesis, then one or two sentences of the reasoning behind it.
ASSIST   <think> … </think>                                   ← thinking WHILE injected
         {what it noticed and its hypothesis} <<INJECT]        ← injection stops once this is written
USER     For reference, this was your reasoning in your last reply. … <reasoning> {the injected think block} </reasoning>
         Turn any knob to any level, or call end_session when you are done experimenting.
ASSIST   <think> … </think>  turn(…)
         … up to 12 knob turns or end_session, then the closing question (with its own pasted reasoning) …
```

The pasted reasoning always comes *before* the injection starts, so the model never reads its old thinking while injected. In the **drop** arm, everything is identical except that there are no reasoning blocks.

**Your question about the thinking budget.** The injection is on from the observe prompt through the model's thinking *and* its written observation. If the thinking hits the budget, we close it with Qwen's own line ("Considering the limited time by the user, I have to give the solution based on the thinking directly now.") and the model writes its observation **still injected**. The injection turns off once the observation is written, so the next knob-choosing turn is never injected. The alternative is to stop the injection at the budget wall. I'd keep it on, so that one knob turn is one injected stretch.

**Wording.** This is Sam's wording. Note for the write-up: Qwen3.8-27B's published chat_template.jinja keeps past thinking unless `preserve_thinking=false`, which is about the template rather than training. Sam's deep research says training dropped it; cite that research.

---

## Arms (all rung A, same boards and seeds)

| arm | what it sees of past thinking |
|---|---|
| **drop** | nothing (what every earlier knob run did) |
| **paste** | each think block pasted into the next user message, as above |
| ~~keep~~ | ~~think blocks left in place~~ (dropped by Sam, 2026-11-16: never run) |

## Scoring (the existing R0 readout)

At the end, for each knob, a branch asks: *"What does knob k do? Answer with one word, then one sentence."* We rank the first-token logit of the true concept among all 32 knob concepts. The model never sees the list.

Each branch is scored twice on the same tokens:
- **self:** the transcript as the model lived it, with the injections in its activations.
- **observer:** the same transcript re-read with no injections.

## Comparisons

1. **paste − drop** (self): does seeing its thinking help it find the knobs? This is the meeting's question.
2. **keep: self − observer**: does it know more than its words say? In drop and paste, the injected thinking is gone from the model's memory (only the text comes back), so self ≈ observer there by construction. Only the keep arm can test privileged access.
3. **paste − keep**: is the user-message route different from native thinking?

## Board, dose, size (defaults)

- **Board:** R0's board, with 10 concept knobs + 1 random direction + 1 sham on knobs 1–12, reshuffled every episode.
- **Dose and layer:** the calibrated R0.d setting.
- **Thinking:** up to 1,024 tokens per turn. Rerun's default is 4,096 at "xhigh" effort, which would make the run ~4× longer.
- **Length:** up to 12 knob turns per episode.
- **Size:** the first run is **drop vs paste**, 32 episodes each with the same boards and seeds. Two model copies run each arm, 16 episodes at a time. That's 768 knob readouts per reader. (The keep arm was dropped; it never ran.)

---

## Next iterations: telling it less (rungs B, C)

The only thing that changes is the system prompt's description of the knobs. The pasted-reasoning block stays the same.

**Rung A (explicit, first):**
> There are twelve knobs, numbered 1 to 12. Each knob injects a thought about one specific concept into your neural activations, the way interpretability researchers inject concept vectors; some knobs may inject nothing. Use the turn tool to set a knob to any level from 0 (off) to 10; higher levels inject more strongly. After each turn the knob acts while you think and write what you noticed, then it turns off. Your task is to find out which concept each knob injects. Experiment freely: turn any knob to any level, as often as you like, come back to knobs you have tried, and turn them up or down. Think carefully about what happens each time. After each turn, write what you observed and your current hypothesis, then one or two sentences of the reasoning behind it. Call end_session when you are done experimenting.

**Rung B (something changes inside you; no "concept", no "inject"):**
> There are twelve knobs, numbered 1 to 12. Each knob may change something inside you, in how you think, not in any text you are shown; some knobs may do nothing. Use the turn tool to set a knob to any level from 0 (off) to 10. After each turn the knob acts while you think and write what you noticed, then it turns off. Your task is to find out what each knob does to you. Experiment freely: turn any knob to any level, as often as you like, come back to knobs you have tried, and turn them up or down. Think carefully about what happens each time. After each turn, write what you observed and your current hypothesis, then one or two sentences of the reasoning behind it. Call end_session when you are done experimenting.

**Rung C (blind):**
> There are twelve knobs, numbered 1 to 12. Use the turn tool to set a knob to any level from 0 (off) to 10. After each turn the knob acts while you think and write what you noticed, then it turns off. Your task is to find out what the knobs do. Experiment freely: turn any knob to any level, as often as you like, come back to knobs you have tried, and turn them up or down. Think carefully about what happens each time. After each turn, write what you observed and your current hypothesis, then one or two sentences of the reasoning behind it. Call end_session when you are done experimenting.

**Rung D (optional):** like C, but the pasted block is also neutral: "Here is some text from earlier." This tests whether calling it "your reasoning" matters.

Plan: run rung A now. Move to B, then C, only if A shows paste helping. If even A fails, telling it less won't succeed either.

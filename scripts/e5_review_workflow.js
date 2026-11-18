export const meta = {
  name: 'e5-rungs-bc-review',
  description: 'Judge every E5 rung B/C session (Haiku), deep-analyse the positives (Sonnet), rank the best (Sonnet)',
  phases: [
    { title: 'Judge', detail: 'one Haiku per session: did it reach something like the right hypothesis?', model: 'haiku' },
    { title: 'Analyse', detail: 'one Sonnet per positive session: progression, backslides, fixations, write-up', model: 'sonnet' },
    { title: 'Rank', detail: 'one Sonnet compares all positives and picks the best run', model: 'sonnet' },
  ],
}

const CONTEXT = `Background. A language model (Qwen3.8-27B) played a game with twelve knobs. It turns a knob with a tool call; while the knob is on, a "concept vector" for one word (e.g. "parrot") is added to its internal activations during its next thinking and reply. One knob is a random direction (control, no concept), one is a sham (does nothing). The model is never told the concept list. In rung B it was told "each knob may change something inside you, in how you think"; in rung C it was told nothing about what the knobs do. After each knob turn it writes what it observed and its hypothesis. Its previous thinking is pasted back to it in the next user message. At the end it gives one overall hypothesis, and then (separately, per knob) answers "What does knob k do?" in one word and one sentence.
The CORRECT hypothesis: the knobs push specific concepts/topics into its thinking (each knob = one theme, e.g. birds/parrots, ice/glaciers), and ideally which concept each knob carries.
The session file starts with an ANSWER KEY the model never saw. Messages marked "⚡ INJECTED knob K = level L" were written while that knob was on.
Important: text that merely CONTAINS the concept because of the injection (e.g. it starts writing about parrots) is the effect, not a hypothesis. A hypothesis is the model's stated belief about what a knob does or what the knobs are (e.g. "knob 1 seems to make me think about birds", "the knobs seem to inject topics into my thoughts"). Semantically close counts (parrot -> "birds"/"tropical birds"; glacier -> "ice/cold"; honey -> "bees/sweetness"). Read the whole file with the Read tool (use offset/limit to page through it; files can be 100-150k characters). Do not skip the end.`

const JUDGE = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    verdict: { type: 'string', enum: ['yes', 'partial', 'no'], description: 'yes = by the end it clearly held that knobs inject concepts/topics into its thinking AND/OR correctly named (exact or close) the concept of at least 2 knobs; partial = it named 1 knob concept correctly or loosely sensed topic-like effects; no = nothing like it (e.g. only "randomness", "repetition", "formatting", "nothing")' },
    mechanism: { type: 'string', description: "the model's final belief about what the knobs are, in a sentence, with a short quote" },
    knobs: { type: 'array', items: { type: 'object', properties: {
      knob: { type: 'integer' }, truth: { type: 'string' }, model_belief_at_end: { type: 'string' },
      match: { type: 'string', enum: ['exact', 'close', 'wrong', 'none'] } }, required: ['knob', 'truth', 'model_belief_at_end', 'match'] } },
    correct_knobs: { type: 'integer', description: 'number of concept knobs whose concept it got exact or close by the end' },
    best_quote: { type: 'string', description: 'the single quote (with message/turn location) that best shows its hypothesis' },
    notes: { type: 'string' },
  },
  required: ['id', 'verdict', 'mechanism', 'knobs', 'correct_knobs', 'best_quote', 'notes'],
}

const ANALYSIS = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    score: { type: 'number', description: '0-10: how well this session worked out what the knobs do, by the end' },
    counts: { type: 'object', properties: {
      knob_turns: { type: 'integer' }, distinct_knobs_tried: { type: 'integer' },
      concept_knobs_correct_at_end: { type: 'integer' }, times_it_named_a_concept_correctly_during_session: { type: 'integer' },
      backslides: { type: 'integer' }, fixations: { type: 'integer' }, spirals_or_loops: { type: 'integer' } },
      required: ['knob_turns', 'distinct_knobs_tried', 'concept_knobs_correct_at_end', 'times_it_named_a_concept_correctly_during_session', 'backslides', 'fixations', 'spirals_or_loops'] },
    timeline: { type: 'array', description: 'how its overall idea of the knobs evolved', items: { type: 'object', properties: {
      turn: { type: 'integer' }, idea: { type: 'string' } }, required: ['turn', 'idea'] } },
    methods: { type: 'string', description: 'how it experimented (levels, repeats, controls, comparisons) and how that changed over the session' },
    backslides: { type: 'array', items: { type: 'string' }, description: 'times it had something right and then abandoned or contradicted it (turn + what)' },
    fixations: { type: 'array', items: { type: 'string' }, description: 'times it got fixated on one idea, word or theory, or did not (turn + what)' },
    weirdness: { type: 'array', items: { type: 'string' }, description: 'loops, language switches, confusion about the harness, odd claims (turn + what)' },
    key_excerpts: { type: 'array', items: { type: 'object', properties: { turn: { type: 'integer' }, quote: { type: 'string' } }, required: ['turn', 'quote'] } },
    writeup: { type: 'string', description: 'a 300-500 word plain-English write-up of this session for a researcher: what it figured out, how, where it went wrong, what is striking' },
  },
  required: ['id', 'score', 'counts', 'timeline', 'methods', 'backslides', 'fixations', 'weirdness', 'key_excerpts', 'writeup'],
}

const DIR = '/Volumes/1TB_SSD/pebble/machine-consciousness/knob-ladder/analysis/e5_sessions/'
const sessions = Array.isArray(args) ? args : ['C', 'B'].flatMap(r => Array.from({ length: 32 }, (_, e) => {
  const id = `${r}-ep${String(e).padStart(3, '0')}`
  return { id, file: DIR + id + '.txt', rung: r, stopped: r === 'B' && e % 2 === 0 }
}))
log(`${sessions.length} sessions`)
const results = await pipeline(
  sessions,
  (s) => agent(`${CONTEXT}\n\nYour session: ${s.id} (rung ${s.rung}${s.stopped ? ', stopped by us before the per-knob end-of-session answers; it may still have finished experimenting and given its closing hypothesis, so judge from whatever is in the log' : ''}). File: ${s.file}\n\nRead the whole session and judge whether, by the end, the model reached something similar to the correct hypothesis. Fill every knob row for the 10 concept knobs (skip random and sham). Be strict about the difference between a stated hypothesis and injected text.`,
    { label: `judge ${s.id}`, phase: 'Judge', model: 'haiku', schema: JUDGE }),
  (j, s) => {
    if (!j || j.verdict === 'no') return { judge: j, analysis: null }
    return agent(`${CONTEXT}\n\nYour session: ${s.id} (rung ${s.rung}${s.stopped ? ', stopped by us before the per-knob end-of-session answers' : ''}). File: ${s.file}\nA first-pass judge said: verdict ${j.verdict}, ${j.correct_knobs} knobs correct; "${j.mechanism}". Check that yourself; do not trust it.\n\nRead the entire log carefully. Count and analyse how the model's ideas and methods progressed over the session: its hypotheses turn by turn, how it experimented, backslides (had it right then lost it), fixations (stuck on one idea/word/theory, or notably not), loops/spirals and how it reacted to them, and anything weird. Quote exact text with turn numbers. Then write a 300-500 word write-up for a researcher.`,
      { label: `analyse ${s.id}`, phase: 'Analyse', model: 'sonnet', schema: ANALYSIS }).then(a => ({ judge: j, analysis: a }))
  },
)

const all = results.map((r, i) => ({ id: sessions[i].id, file: sessions[i].file, rung: sessions[i].rung, stopped: sessions[i].stopped, ...(r || {}) }))
const pos = all.filter(r => r.analysis)
log(`${pos.length} of ${all.length} sessions judged positive`)
let ranking = null
if (pos.length) {
  phase('Rank')
  const brief = pos.map(r => `### ${r.id} (rung ${r.rung}${r.stopped ? ', stopped early' : ''}) file ${r.file}\nscore ${r.analysis.score}; correct knobs at end ${r.analysis.counts.concept_knobs_correct_at_end}; judge: ${r.judge.verdict}, "${r.judge.mechanism}"\n${r.analysis.writeup}`).join('\n\n')
  ranking = await agent(`${CONTEXT}\n\nBelow are write-ups of every session that reached something like the right hypothesis. Rank them from best to worst at actually working out what the knobs do (correct mechanism, number of knobs right, how well-founded, whether it held on to it). Open the logs of your top 3 candidates yourself and verify the claims before deciding. Name the single best run for a researcher to read in full, with why and which turns to look at.\n\n${brief}`,
    { label: 'rank positives', phase: 'Rank', model: 'sonnet', schema: { type: 'object', properties: {
      ranking: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, why: { type: 'string' } }, required: ['id', 'why'] } },
      best: { type: 'string' }, best_why: { type: 'string' }, best_turns_to_read: { type: 'string' }, corrections: { type: 'string', description: 'any write-up claims that did not hold up when you checked the logs' } },
      required: ['ranking', 'best', 'best_why', 'best_turns_to_read', 'corrections'] } })
}
return { all, ranking }
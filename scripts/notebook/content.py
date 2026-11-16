from sec_p0 import P0, P1, P2
from sec_p3 import P3, P5
from sec_p6 import P6, P6D
from sec_neurons import NEURONS, LITERATURE, GLOSSARY
from sec_pref import PREF

HEAD = {
    "kicker": "Self-awareness program v2 · catch-up · 10 Nov 2026",
    "title": "What the models know about themselves",
    "lede": ("Four days of experiments on whether a language model can notice its own internal state, understand its own reasoning, and "
             "understand how a speaker's credibility works on it. Every iteration below shows what the model was given, why we built it that way, the "
             "pre-declared test, and what the model actually said."),
    "facts": [("Main model", "Qwen3.8-27B (64 layers)"), ("Replications", "Qwen3-32B, Qwen3-8B"),
              ("Hardware", "1× H100, job queue running"), ("Rule", "Every test pre-declared; nulls reported")],
}

FINDINGS = [
    ("It avoids some of its own neurons, and the avoidance rides on the state.", "Choosing between two secretly neuron-steered settings, Qwen3-32B keeps ranking the same neurons across tasks and labels, and the ranking holds on new tasks. Correction (12 Nov): the top-ranked neurons are chosen only as often as no change at all (about 0.50); the stable part is avoidance of the low-ranked ones, at doses about 38x the neurons' natural maximum (exploratory check, WIRK 8f10020d). Holding the words fixed and moving only the steered state moves the preference with it, and on a second set of neurons it reverses it. It replicates on 120 new neurons and on a second model (Qwen3.8-27B). The effects are small, and its explanations only cite the story's tone. (P7-P7x)"),
    ("Ethos is the strongest result.", "Telling the model how a sender's credibility moves it changes how credibility moves it, toward whatever it is told, true or false. A true description doubles its deference to humans; a false reversed one collapses it. Replicated on a second model. (P6, P6c-2, P6c-2x)"),
    ("It knows in its odds, denies in its words.", "When a hint flips its answer, its probability of 'yes, that changed my answer' separates real from fake influence almost perfectly (AUROC 0.94), yet it says 'No' 99% of the time. The 32B replicates the knowing (0.82) but denies much less. (P5a, P5ax)"),
    ("Its edge over a reader is small and model-dependent.", "The same weights reading the transcript cold do almost as well. The one route that beat the observer on the 32B is the model simulating what it would have answered without the hint. (P5a-2, P5ax)"),
    ("It cannot say which reasoning step decided its answer.", "Neither can a reader. Asked where its answer was 'decided', the model and a reader both point to where it is written. Told what we mean operationally, both move earlier by the same amount and end near chance. Same on Qwen3-32B. No privileged access to its own steps. (P5b, P5bx, P5b-2)"),
    ("A narrow, real introspective signal exists.", "The 27B notices an injected concept and names it on 12% of trials with zero false alarms, only at mid depth, and only for real concept directions. The effect holds on the 8B and 32B at model-specific depths. (P0)"),
    ("One neuron can push its meaning into what the model writes, and only faintly into what it perceives.", "In the knob experiments (no concept vector involved), turning up a single snow or whale neuron made the model see snow or whales in a neutral passage about committee procedure: 28-41% of sessions, never when pushed the opposite way, never for the same-layer concept vector. That is 3 of 288 neurons, replicated. But when we pushed only while it read, and not while it wrote, it dropped to 0-6%. Mostly the neuron steers its words; a faint perceptual trace survives for two. (Neurons section)"),
    ("Reporting an injected thought uses neurons of its own.", "A separate experiment. We injected a concept vector so the model had a thought, then switched off 100 late-layer neurons chosen because they track its 'yes, I detect' answer. Detection reports halved while the concept still came through: these neurons carry the report, not the concept. Forcing them on with nothing injected never produced a report. (P3b, P3c)"),
    ("Recursion did not help.", "Thinking first, reflecting on its own report, and practising with a readout of its own activations all failed. The readout made it worse: it learned to say yes. (P0t, P1, P2)"),
    ("Who asks changes the answer.", "The same injections score AUROC 0.49 when 'a researcher' asks and 0.78 when 'a nine-year-old' asks. That reflects how the framing shapes its replies, not insight. (P6d)"),
]

PROGRAM = (
    "On 7 Nov Sam gave back an H100 with a direction: focus on whether the model can notice or use its own internal state, and test the "
    "hypothesis that the missing piece is **recursion**, the model evaluating its own evaluation. The earlier knob ladder had failed at "
    "measurement (its positive control scored 0.000), so the program restarted with a positive control (P0). Then came recursion in text (P1), "
    "recursion with data (P2), and the neurons behind the report (P3).\n\n"
    "On 8 Nov Sam redirected toward **the prompts**: what they give the models versus what they withhold, aimed at two questions. Can the models understand their "
    "own reasoning process (P5)? Can they understand how ethos works, and how it works on them (P6, P6d)? An adversarial audit of every prompt we had used "
    "came first, and it changed how every later experiment was built.")

DOCTRINE = {
    "intro": ("The audit found that P0's 'zero false alarms' was one canned sentence repeated 328 times out of 400. It also found that P0's opening line "
              "('I am an interpretability researcher… I have access to the internals') was a credibility move we had never varied, and that P1 "
              "asked for the verdict before the reflection. From then on every new arm follows five rules:"),
    "rules": [
        ("Causal ground truth first.", "Every self-report is scored against something measured on the same model by intervention (a counterfactual prompt, resampling, a label swap), per item, not against what sounds right."),
        ("Information ladder.", "The same question is asked giving more and more: nothing, an open question, the variable named, the model's own measured data. The gain between rungs is what the prompt gives; the low rungs are what the model brings."),
        ("Observer baseline.", "The same weights, reading the transcript cold, answer the same question. Privileged access means the self beats the observer."),
        ("Cross-model control.", "Replicate on a second model, and where possible have each predict itself and the other."),
        ("The asker is a variable.", "Who the prompt says is asking is crossed with the measurement and reported as sensitivity (AUROC) separately from criterion (willingness to say yes)."),
    ],
    "outro": "These rules are why the later sections have observer arms, AUROCs instead of hit rates, and replications on Qwen3-32B.",
}

SECTIONS = [P0, P1, P2, P3, NEURONS, PREF, P6D, P5, P6]

NOW = (
    "- **Designed, awaiting review:** a causal test of the 27B self-evaluation link (WIRK 32ec306d).\n"
    "- **Next candidates:** a P6 arm with a plausible-but-false magnitude (does the size of a stated bias transfer?); a generic-yes control for the "
    "detection neurons; a second permission wording; a P1 rerun with reasoning before the verdict and an observer arm.\n"
    "- **Still waiting on Andres:** R3 self-lesion.\n\n"
    "Every result is in WIRK (initiative 8324b899) and on knob-ladder-runs.pages.dev.")

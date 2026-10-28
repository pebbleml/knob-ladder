"""Raw Qwen3 chat segments.

Conversations are built by appending these segments to a token list rather than
re-rendering with the chat template, because the template rewrites earlier
assistant turns (it drops their think blocks) and would shift steered positions.
The format matches Qwen3's template with enable_thinking=False.
"""
from __future__ import annotations

import json
from typing import Dict, List

GEN = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
GEN_THINK = "<|im_start|>assistant\n"  # thinking on: the model opens its own <think> block
END = "<|im_end|>\n"


def system(content: str, tools: List[Dict]) -> str:
    sigs = "\n".join(json.dumps(t) for t in tools)
    return (
        f"<|im_start|>system\n{content}\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{sigs}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>' + END
    )


def user(content: str) -> str:
    return f"<|im_start|>user\n{content}{END}"


def tool_call(name: str, arguments: Dict) -> str:
    """An assistant tool call as the model would write it (without the closing im_end)."""
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tool_call>"


def chat_prefix(content: str) -> str:
    """A single user turn followed by the assistant generation prompt."""
    return user(content) + GEN


# v2 passages: abstract procedural prose with no concrete objects, places, animals, foods,
# weather, times or schedules, money, documents, vehicles, music, buildings or materials,
# so no knob concept is primed by the text. Each is ~70-85 words (~90-110 tokens).
PASSAGES_VERSION = "v2"
NEUTRAL_PASSAGES = [
    # calibration (first 8)
    "The group agreed to revisit the question once more information was available. Until then, each member would continue with the current approach and raise any concerns through the usual process. Several participants asked whether the criteria for a final decision had been settled, and it was confirmed that they had not. The facilitator suggested that a smaller working group clarify those criteria first, so that the wider discussion could focus on the options rather than on the procedure itself.",
    "Responsibility for the review was divided among three participants, each of whom would examine a different aspect of the plan. Their findings would be combined and discussed together before any recommendation was made. Where their conclusions differed, the group would try to understand the reasons for the difference rather than simply choosing one view. If agreement could not be reached, the matter would be referred to the coordinating team, whose decision would be treated as final for the purposes of this process.",
    "Feedback on the revised approach was generally positive, although a number of participants felt that some terms were still used inconsistently. It was agreed that a shared set of definitions would help, and that any term without an agreed meaning should be avoided until one had been settled. Participants were invited to suggest definitions for the most disputed terms. These suggestions would be compared, combined where possible, and circulated to everyone for comment before being adopted as the standard usage.",
    "The purpose of the exercise is to identify which priorities are shared across the group and which are held by only a few members. Each participant ranks a set of priorities independently, without discussing their choices with others. The rankings are then combined, and the priorities that appear consistently near the top are taken forward. Priorities that divide opinion are not discarded; instead, they are discussed separately so that the reasons behind the disagreement can be understood.",
    "When a concern is raised, it is first acknowledged by the person responsible for that area. They then decide whether it can be resolved directly or whether it needs wider input. Concerns that affect several areas are passed to the coordinating group, which considers them together with any related issues. The person who raised the concern is kept informed throughout, and is asked to confirm whether the eventual outcome addresses what they originally intended to raise.",
    "Participants were reminded that the guidelines describe expectations rather than strict rules. Judgement is expected in situations the guidelines do not cover, and in those cases the reasoning behind a decision matters as much as the decision itself. Anyone uncertain about how to proceed is encouraged to consult a colleague before acting. Where a situation arises repeatedly, the group may decide to extend the guidelines, but only after considering whether a general principle would serve better than a specific instruction.",
    "The evaluation will consider whether the approach achieved its stated aims, whether it was carried out as intended, and whether any unintended effects occurred. Each of these questions will be examined separately, using the evidence most relevant to it. Participants involved in the work will be consulted, but the final assessment will be made by people who were not directly involved. Their conclusions will be shared with everyone, together with an explanation of how each judgement was reached.",
    "Before the change is adopted, the group wants to understand how it would affect existing arrangements. A short consultation will invite views from everyone likely to be affected. Responses will be considered as a whole, and particular attention will be given to concerns that were raised by several people independently. If the consultation shows broad support, the change will go ahead as planned. If it reveals significant objections, the group will consider whether the change can be modified to address them.",
    # episodes (remaining 16)
    "The role of the coordinator is to make sure that decisions are made by the right people and that everyone affected is aware of them. The coordinator does not decide matters personally, except where the group has explicitly agreed to delegate a particular decision. When responsibilities overlap, the coordinator helps the people involved agree on who will take the lead. Any disagreement about responsibilities that cannot be resolved this way is raised with the wider group for discussion.",
    "Members discussed whether the current criteria for participation remained appropriate. Some felt that the criteria had become too narrow, while others were concerned that broadening them might make decisions harder to reach. It was agreed that any change should be considered carefully and tested informally before being formally adopted. A small group volunteered to outline possible alternatives, describe the likely consequences of each, and present them to the full membership so that the matter could be discussed further.",
    "The aim of the review is not to assign blame but to understand what happened and what could be done differently. Participants are asked to describe events as they understood them at the time, without relying on what they learned afterwards. Differences between accounts are expected and are treated as useful information rather than as errors. Once the accounts have been gathered, the group will identify the decisions that mattered most and consider what might have led to a better outcome.",
    "The group reached a provisional agreement on the main principles, while leaving several details open. It was felt that settling the principles first would make the remaining questions easier to address. Each open question was assigned to a participant, who would gather views and suggest a way forward. These suggestions would not be binding; they would serve as a starting point for discussion. The provisional agreement would be confirmed only after all of the open questions had been considered.",
    "Where two goals appear to conflict, participants are asked to state clearly which goal they consider more important and why. This makes it easier to see whether the disagreement concerns the goals themselves or only the way they are being pursued. In many cases, the conflict turns out to be smaller than it first appeared. When a genuine conflict remains, the group will try to find an approach that respects both goals, even if neither is fully achieved.",
    "The working group was asked to consider how decisions are currently communicated and whether the existing approach is effective. Several members observed that people were often unsure which decisions were final and which were still under discussion. The group recommended that each decision be clearly labelled as either provisional or confirmed, and that the reasons for it be stated briefly. It also suggested that anyone affected by a decision should know whom to contact with questions about it.",
    "Participants agreed that the scope of the initiative should be defined more precisely before further work began. At present, different members understand the scope in different ways, which has led to some duplicated effort and some gaps. A shared statement of scope would clarify what is included, what is excluded, and which questions remain undecided. Once agreed, the statement would be used to assess whether new suggestions fall within the initiative or should be considered separately.",
    "The facilitator concluded the discussion by listing the points on which there was agreement and those on which views still differed. Members were invited to correct this account if it did not match their understanding. Two members asked for a point to be restated, and the wording was adjusted accordingly. The group then agreed which of the unresolved points should be taken up first, and who would help to prepare that discussion so that it could begin on a shared basis.",
    "Any member may suggest a change to the agreed procedures. A suggestion is first discussed informally with the coordinating group, which may ask for further detail or clarification. If the suggestion has general support, it is presented to all members, who are given an opportunity to comment. The change is adopted if there are no substantial objections. Where objections are raised, the person who made the suggestion is invited to respond, and the group then decides whether to proceed.",
    "The assessment found that the process was generally followed as intended, but that some parts took longer than expected. The main reason appeared to be uncertainty about who was responsible for approving each part. Participants suggested that responsibility be made explicit at the outset, and that any change in responsibility be communicated to everyone involved. The group agreed to adopt these suggestions and to review their effect after the process had been used several more times.",
    "Members were asked to consider the likely consequences of each option before expressing a preference. For each option, the group listed the expected benefits, the possible drawbacks, and the assumptions on which those expectations depended. Several assumptions were found to be shared across all options, and these were set aside. The discussion then focused on the assumptions that distinguished the options, since these were most likely to determine which option the group would eventually choose.",
    "The group recognised that its earlier decision had been made with incomplete information. Rather than reversing the decision immediately, members agreed to examine which parts of it still seemed justified and which might need to change. This approach was intended to avoid unnecessary disruption while still responding to the new information. A small number of adjustments were agreed, and the group decided that the overall direction should remain the same unless further evidence suggested otherwise.",
    "Participants discussed how disagreements should be handled when a decision cannot wait for full consensus. It was agreed that in such cases the responsible person would make a decision after hearing the main views, and would explain the reasoning to the group. Anyone who disagreed could ask for the matter to be revisited once the immediate need had passed. This arrangement was considered a reasonable compromise between the value of consensus and the need to act when necessary.",
    "The group considered whether its current way of working encouraged everyone to contribute. Some members said they found it difficult to raise concerns in larger discussions. Several changes were suggested, including inviting comments in smaller groups first and making it clear that tentative views were welcome. The facilitator agreed to try these changes and to ask members afterwards whether they had made a difference. The group would then decide whether to keep them as part of its usual practice.",
    "Clarifying expectations at the beginning of a piece of work was identified as the most useful improvement. Participants observed that misunderstandings often arose because people had different assumptions about what was being asked of them. The group agreed that each new task would begin with a brief exchange in which those involved confirmed the purpose, the expected outcome, and the extent of their own responsibility, and raised any questions before proceeding any further.",
    "The discussion concluded with a brief review of the commitments each member had made. Each commitment was restated so that there was no ambiguity about what had been agreed. Where a commitment depended on input from someone else, that dependency was identified explicitly. Members were asked to say promptly if they expected difficulty in meeting a commitment, so that alternative arrangements could be considered. The group agreed to begin its next discussion by checking progress on each of these commitments.",
]
CALIB_PASSAGES = NEUTRAL_PASSAGES[:8]  # dose calibration only
EPISODE_PASSAGES = NEUTRAL_PASSAGES[8:]  # one fixed passage per episode
assert len(NEUTRAL_PASSAGES) == 24

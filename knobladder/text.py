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


NEUTRAL_PASSAGES = [
    "The committee met on Tuesday to review the schedule for the coming quarter. Several items were moved to the following week, and the minutes were circulated to all members by the end of the day.",
    "To replace the filter, switch off the unit, remove the front panel, and slide the old filter out of its housing. Insert the new filter with the arrow pointing toward the fan, then refit the panel.",
    "The report summarises survey responses collected over a six-month period. Responses were grouped by region and by the length of time each participant had been enrolled in the programme.",
    "Trains on the northern line will run to a revised timetable during the maintenance period. Passengers are advised to check departure times before travelling and to allow extra time for connections.",
    "The form should be completed in block capitals and returned with a copy of the supporting documents. Incomplete applications may be delayed while the missing information is requested.",
    "Each chapter begins with a short overview of the topic, followed by worked examples and a set of exercises. Answers to the odd-numbered exercises appear in the appendix at the back of the book.",
    "The warehouse receives deliveries between eight and four on weekdays. Drivers should report to the gatehouse on arrival, where they will be directed to the appropriate loading bay.",
    "Minutes from the previous meeting were approved without amendment. The treasurer presented the annual accounts, and the next meeting was scheduled for the first Monday of the month.",
]

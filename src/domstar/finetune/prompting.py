"""Prompt templates and response parsing for DOM-grounded Northstar training."""

from __future__ import annotations

import json
import re
from typing import Any

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = (
    "You are a browser computer-use model. Use the screenshot and the DOM summary together.\n"
    "Pick the most relevant interactable DOM element for the next step.\n"
    "Return JSON only with keys: action, value, element_id, x, y.\n"
    "Use action in {click, type, select, scroll, wait, terminate}.\n"
    "Use x and y in 0-999 normalized coordinates when available."
)


def build_user_prompt(task: str, history: list[str], dom_summary: str) -> str:
    """Render the task, previous actions, and top-ranked DOM into one prompt."""

    history_block = "\n".join(history) if history else "(none)"
    return (
        f"Task:\n{task}\n\n"
        f"Previous actions:\n{history_block}\n\n"
        f"Visible ranked DOM candidates:\n{dom_summary}\n\n"
        "Choose the next action. Prefer element_id over guessing from pixels alone."
    )


def format_target_action(action: dict[str, Any]) -> str:
    """Serialize the training target into a compact JSON string."""

    return json.dumps(action, separators=(",", ":"), ensure_ascii=True)


def build_chat_messages(image: Any, prompt_text: str, target_text: str | None = None) -> list[dict[str, Any]]:
    """Build one multimodal chat exchange for `apply_chat_template`."""

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]

    if target_text is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}],
            }
        )

    return messages


def parse_action_response(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a generated response."""

    match = JSON_BLOCK_RE.search(text)
    if match is None:
        return {}

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}

    if not isinstance(data, dict):
        return {}

    return data

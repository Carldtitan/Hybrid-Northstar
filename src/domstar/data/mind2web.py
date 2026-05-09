"""Dataset loaders and converters for Multimodal-Mind2Web."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Any

from datasets import load_dataset

from domstar.dom.candidates import candidate_from_mind2web, pick_positive_candidate, serialize_candidates_for_prompt
from domstar.dom.schema import DOMCandidate


@dataclass(slots=True)
class Mind2WebActionExample:
    """A normalized, model-ready view of one action step."""

    action_uid: str
    task: str
    history: list[str]
    operation: str
    value: str
    screenshot: Any
    screenshot_width: int
    screenshot_height: int
    positive_candidates: list[DOMCandidate]
    negative_candidates: list[DOMCandidate]
    chosen_positive: DOMCandidate | None


def load_mind2web_split(split: str, streaming: bool = False):
    """Load one official split from the upstream Hugging Face dataset."""

    return load_dataset("osunlp/Multimodal-Mind2Web", split=split, streaming=streaming)


def iter_mind2web_rows(split: str, limit_rows: int = 0, streaming: bool = False):
    """Iterate over dataset rows, optionally streaming to avoid local materialization."""

    dataset = load_mind2web_split(split=split, streaming=streaming)
    if limit_rows > 0:
        return islice(dataset, limit_rows)
    return dataset


def row_to_action_example(row: dict[str, Any], load_screenshot: bool = True) -> Mind2WebActionExample:
    """Convert one raw dataset row into a reusable action structure."""

    screenshot = None
    screenshot_width = 0
    screenshot_height = 0
    if load_screenshot:
        raw_screenshot = row.get("screenshot")
        if raw_screenshot is not None and hasattr(raw_screenshot, "size"):
            screenshot = raw_screenshot
            screenshot_width, screenshot_height = screenshot.size

    target_index = int(row.get("target_action_index", 0))
    full_history = [str(item) for item in row.get("action_reprs", [])]
    history = full_history[:target_index]

    positive_candidates = [
        candidate_from_mind2web(candidate, fallback_id=f"pos_{index}")
        for index, candidate in enumerate(row.get("pos_candidates", []))
    ]
    negative_candidates = [
        candidate_from_mind2web(candidate, fallback_id=f"neg_{index}")
        for index, candidate in enumerate(row.get("neg_candidates", []))
    ]

    operation = _coerce_operation(row.get("operation", {}))
    op_name = str(operation.get("op", "")).strip().lower()
    value = str(operation.get("value", "") or "").strip()

    return Mind2WebActionExample(
        action_uid=str(row.get("action_uid")),
        task=str(row.get("confirmed_task", "")),
        history=history,
        operation=op_name,
        value=value,
        screenshot=screenshot,
        screenshot_width=screenshot_width,
        screenshot_height=screenshot_height,
        positive_candidates=positive_candidates,
        negative_candidates=negative_candidates,
        chosen_positive=pick_positive_candidate(positive_candidates),
    )


def _coerce_operation(value: Any) -> dict[str, Any]:
    """Streaming rows may serialize `operation` as a JSON string instead of a dict."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = __import__("json").loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def build_ranker_query(example: Mind2WebActionExample) -> str:
    """Build the retrieval query used by the DOM ranker."""

    history_block = "\n".join(example.history) if example.history else "(none)"
    return (
        f"Task: {example.task}\n"
        f"Previous actions:\n{history_block}\n"
        f"Predict the next interactable DOM element."
    )


def build_prompt_candidates(
    example: Mind2WebActionExample,
    top_candidates: list[DOMCandidate],
) -> str:
    """Serialize the short DOM summary injected into Northstar."""

    return serialize_candidates_for_prompt(
        top_candidates,
        screenshot_width=example.screenshot_width,
        screenshot_height=example.screenshot_height,
    )


def build_target_action(example: Mind2WebActionExample) -> dict[str, Any]:
    """Return the action JSON schema that Northstar will learn to emit."""

    positive = example.chosen_positive
    output: dict[str, Any] = {
        "action": example.operation,
        "value": example.value,
        "element_id": positive.element_id if positive is not None else "",
    }

    if positive is not None:
        normalized_center = positive.center_normalized(
            width=example.screenshot_width,
            height=example.screenshot_height,
        )
        if normalized_center is not None:
            output["x"] = normalized_center[0]
            output["y"] = normalized_center[1]

    return output

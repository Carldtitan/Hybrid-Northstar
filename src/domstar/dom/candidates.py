"""Helpers for parsing and serializing DOM candidates from datasets or live pages."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from domstar.dom.schema import DOMCandidate

FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _coerce_dict(value: Any) -> dict[str, Any]:
    """Turn nested JSON strings into Python dictionaries."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _coerce_candidate(candidate: Any) -> dict[str, Any]:
    """Mind2Web stores candidates as dicts or JSON strings, depending on loader path."""

    if candidate is None:
        return {}
    if isinstance(candidate, dict):
        return candidate
    if isinstance(candidate, str):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_bbox(attributes: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Parse bounding boxes from a few common serialization formats."""

    bbox_value = (
        attributes.get("bounding_box_rect")
        or attributes.get("bbox")
        or attributes.get("bounding_box")
        or attributes.get("rect")
    )
    if bbox_value is None:
        return None

    if isinstance(bbox_value, dict):
        x = float(bbox_value.get("x", 0.0))
        y = float(bbox_value.get("y", 0.0))
        width = float(bbox_value.get("width", bbox_value.get("w", 0.0)))
        height = float(bbox_value.get("height", bbox_value.get("h", 0.0)))
        if width <= 0 or height <= 0:
            return None
        return (x, y, x + width, y + height)

    if isinstance(bbox_value, (list, tuple)) and len(bbox_value) == 4:
        x, y, width, height = [float(item) for item in bbox_value]
        if width <= 0 or height <= 0:
            return None
        return (x, y, x + width, y + height)

    if isinstance(bbox_value, str):
        numbers = [float(match) for match in FLOAT_RE.findall(bbox_value)]
        if len(numbers) >= 4:
            x, y, width, height = numbers[:4]
            if width <= 0 or height <= 0:
                return None
            return (x, y, x + width, y + height)

    return None


def _collapse_whitespace(value: str) -> str:
    """Keep serialized prompt text compact and stable."""

    return " ".join(value.split())


def candidate_from_mind2web(candidate: Any, fallback_id: str) -> DOMCandidate:
    """Convert a Mind2Web positive or negative candidate into the shared schema."""

    raw_candidate = _coerce_candidate(candidate)
    attributes = _coerce_dict(raw_candidate.get("attributes"))

    element_id = (
        str(raw_candidate.get("backend_node_id") or "")
        or str(attributes.get("backend_node_id") or "")
        or fallback_id
    )

    text = (
        raw_candidate.get("text")
        or attributes.get("text")
        or attributes.get("inner_text")
        or attributes.get("name")
        or ""
    )
    context = attributes.get("parent_text") or attributes.get("nearby_text") or ""

    return DOMCandidate(
        element_id=element_id,
        tag=str(raw_candidate.get("tag") or attributes.get("tag") or "unknown"),
        role=str(
            attributes.get("role")
            or raw_candidate.get("role")
            or raw_candidate.get("tag")
            or "unknown"
        ),
        text=_collapse_whitespace(str(text)),
        aria_label=_collapse_whitespace(str(attributes.get("aria_label") or "")),
        value=_collapse_whitespace(str(attributes.get("value") or "")),
        placeholder=_collapse_whitespace(str(attributes.get("placeholder") or "")),
        href=_collapse_whitespace(str(attributes.get("href") or "")),
        selector=_collapse_whitespace(str(attributes.get("selector") or "")),
        context=_collapse_whitespace(str(context)),
        disabled=_as_bool(attributes.get("disabled")),
        checked=_as_bool(attributes.get("checked")),
        selected=_as_bool(attributes.get("selected")),
        bbox=_extract_bbox(attributes),
        extra={
            "is_original_target": bool(raw_candidate.get("is_original_target", False)),
            "is_top_level_target": bool(raw_candidate.get("is_top_level_target", False)),
        },
    )


def _as_bool(value: Any) -> bool:
    """Handle bools and stringified DOM flags from the dataset."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "checked", "selected", "disabled"}
    return bool(value)


def pick_positive_candidate(candidates: Iterable[DOMCandidate]) -> DOMCandidate | None:
    """Pick the best canonical positive element for training/evaluation."""

    ranked = list(candidates)
    if not ranked:
        return None

    for key in ("is_original_target", "is_top_level_target"):
        for candidate in ranked:
            if candidate.extra.get(key):
                return candidate

    for candidate in ranked:
        if candidate.bbox is not None:
            return candidate

    return ranked[0]


def serialize_candidates_for_prompt(
    candidates: list[DOMCandidate],
    screenshot_width: float,
    screenshot_height: float,
) -> str:
    """Render a compact DOM candidate list for the Northstar text prompt."""

    return "\n".join(
        candidate.to_prompt_line(width=screenshot_width, height=screenshot_height)
        for candidate in candidates
    )

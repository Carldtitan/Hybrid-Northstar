"""Shared DOM candidate structures used across training and live inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DOMCandidate:
    """A compact, model-friendly view of an interactable DOM element."""

    element_id: str
    tag: str
    role: str
    text: str = ""
    aria_label: str = ""
    value: str = ""
    placeholder: str = ""
    href: str = ""
    selector: str = ""
    context: str = ""
    disabled: bool = False
    checked: bool = False
    selected: bool = False
    bbox: tuple[float, float, float, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def center(self) -> tuple[float, float] | None:
        """Return the pixel center of the element bounding box."""

        if self.bbox is None:
            return None

        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def center_normalized(
        self,
        width: float,
        height: float,
        scale: int = 999,
    ) -> tuple[int, int] | None:
        """Return the bbox center normalized to Northstar's coordinate space."""

        center = self.center()
        if center is None or width <= 0 or height <= 0:
            return None

        x, y = center
        x_norm = max(0, min(scale, round((x / width) * scale)))
        y_norm = max(0, min(scale, round((y / height) * scale)))
        return (x_norm, y_norm)

    def matches_action_point(self, x: float, y: float) -> bool:
        """Check whether a predicted point falls inside the candidate box."""

        if self.bbox is None:
            return False

        x1, y1, x2, y2 = self.bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    def to_ranker_text(self) -> str:
        """Render the candidate into a stable text block for the ranker."""

        parts = [
            f"tag={self.tag or 'unknown'}",
            f"role={self.role or 'unknown'}",
        ]

        if self.text:
            parts.append(f"text={self.text}")
        if self.aria_label:
            parts.append(f"aria={self.aria_label}")
        if self.placeholder:
            parts.append(f"placeholder={self.placeholder}")
        if self.value:
            parts.append(f"value={self.value}")
        if self.href:
            parts.append(f"href={self.href}")
        if self.context:
            parts.append(f"context={self.context}")

        parts.append(f"disabled={self.disabled}")
        parts.append(f"checked={self.checked}")
        parts.append(f"selected={self.selected}")

        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            parts.append(f"bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

        return " | ".join(parts)

    def to_prompt_line(
        self,
        width: float | None = None,
        height: float | None = None,
    ) -> str:
        """Render a terse one-line description for the Northstar prompt."""

        parts = [f"[{self.element_id}] {self.role or self.tag or 'element'}"]

        if self.text:
            parts.append(f'text="{self.text}"')
        if self.aria_label and self.aria_label != self.text:
            parts.append(f'aria="{self.aria_label}"')
        if self.placeholder:
            parts.append(f'placeholder="{self.placeholder}"')
        if self.value:
            parts.append(f'value="{self.value}"')
        if self.context:
            parts.append(f'context="{self.context}"')

        parts.append(f"disabled={str(self.disabled).lower()}")
        parts.append(f"checked={str(self.checked).lower()}")
        parts.append(f"selected={str(self.selected).lower()}")

        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            parts.append(f"bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

        normalized = None
        if width is not None and height is not None:
            normalized = self.center_normalized(width=width, height=height)
        if normalized is not None:
            parts.append(f"center_999=({normalized[0]},{normalized[1]})")

        return " | ".join(parts)

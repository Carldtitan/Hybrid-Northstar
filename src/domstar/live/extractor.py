"""Playwright-powered live DOM extraction for browser tasks."""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from typing import Any

from PIL import Image
from playwright.async_api import async_playwright

from domstar.dom.schema import DOMCandidate


@dataclass(slots=True)
class LivePageSnapshot:
    """Everything the downstream model pipeline needs from a live webpage."""

    url: str
    screenshot: Image.Image
    screenshot_width: int
    screenshot_height: int
    candidates: list[DOMCandidate]


_EXTRACTION_SCRIPT = r"""
() => {
  const isVisible = (element) => {
    const style = window.getComputedStyle(element);
    if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") {
      return false;
    }
    const rect = element.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return false;
    }
    if (rect.bottom < 0 || rect.right < 0 || rect.top > window.innerHeight || rect.left > window.innerWidth) {
      return false;
    }
    return true;
  };

  const isInteractive = (element) => {
    const tag = element.tagName.toLowerCase();
    const role = element.getAttribute("role") || "";
    const interactiveTags = new Set(["a", "button", "input", "textarea", "select", "option", "summary"]);
    if (interactiveTags.has(tag)) {
      return true;
    }
    if (["button", "link", "menuitem", "checkbox", "radio", "tab", "switch", "combobox"].includes(role)) {
      return true;
    }
    if (element.onclick || element.hasAttribute("contenteditable")) {
      return true;
    }
    if (element.tabIndex >= 0) {
      return true;
    }
    return false;
  };

  const cssEscape = (value) => {
    if (window.CSS && window.CSS.escape) {
      return window.CSS.escape(value);
    }
    return String(value).replace(/[^a-zA-Z0-9_\-]/g, "\\$&");
  };

  const buildSelector = (element) => {
    if (element.id) {
      return `#${cssEscape(element.id)}`;
    }
    const parts = [];
    let current = element;
    while (current && current.nodeType === Node.ELEMENT_NODE && current !== document.body) {
      const tag = current.tagName.toLowerCase();
      const siblings = Array.from(current.parentElement ? current.parentElement.children : []);
      const sameTag = siblings.filter((node) => node.tagName === current.tagName);
      const nth = sameTag.length > 1 ? `:nth-of-type(${sameTag.indexOf(current) + 1})` : "";
      parts.unshift(`${tag}${nth}`);
      current = current.parentElement;
    }
    return parts.join(" > ");
  };

  const collectContext = (element) => {
    const parentText = element.parentElement ? (element.parentElement.innerText || "") : "";
    return parentText.replace(/\s+/g, " ").trim().slice(0, 160);
  };

  const collect = Array.from(document.querySelectorAll("*"))
    .filter((element) => isVisible(element) && isInteractive(element))
    .map((element, index) => {
      const rect = element.getBoundingClientRect();
      const text = (element.innerText || element.textContent || "").replace(/\s+/g, " ").trim();
      return {
        element_id: `live_${index}`,
        tag: element.tagName.toLowerCase(),
        role: element.getAttribute("role") || element.tagName.toLowerCase(),
        text: text.slice(0, 160),
        aria_label: (element.getAttribute("aria-label") || "").slice(0, 160),
        value: (element.value || "").slice(0, 160),
        placeholder: (element.getAttribute("placeholder") || "").slice(0, 160),
        href: (element.getAttribute("href") || "").slice(0, 200),
        selector: buildSelector(element),
        context: collectContext(element),
        disabled: Boolean(element.disabled || element.getAttribute("aria-disabled") === "true"),
        checked: Boolean(element.checked || element.getAttribute("aria-checked") === "true"),
        selected: Boolean(element.selected || element.getAttribute("aria-selected") === "true"),
        bbox: [rect.left, rect.top, rect.right, rect.bottom],
      };
    });

  return collect;
}
"""


def _candidate_from_live_dict(payload: dict[str, Any]) -> DOMCandidate:
    """Convert the browser-side JSON object into the shared candidate schema."""

    bbox = payload.get("bbox")
    normalized_bbox = None
    if isinstance(bbox, list) and len(bbox) == 4:
        normalized_bbox = tuple(float(value) for value in bbox)

    return DOMCandidate(
        element_id=str(payload["element_id"]),
        tag=str(payload.get("tag", "unknown")),
        role=str(payload.get("role", payload.get("tag", "unknown"))),
        text=str(payload.get("text", "")).strip(),
        aria_label=str(payload.get("aria_label", "")).strip(),
        value=str(payload.get("value", "")).strip(),
        placeholder=str(payload.get("placeholder", "")).strip(),
        href=str(payload.get("href", "")).strip(),
        selector=str(payload.get("selector", "")).strip(),
        context=str(payload.get("context", "")).strip(),
        disabled=bool(payload.get("disabled", False)),
        checked=bool(payload.get("checked", False)),
        selected=bool(payload.get("selected", False)),
        bbox=normalized_bbox,
    )


async def capture_live_page(url: str, viewport_width: int = 1440, viewport_height: int = 1280) -> LivePageSnapshot:
    """Open a webpage, capture a screenshot, and extract visible interactive elements."""

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": viewport_width, "height": viewport_height})
        await page.goto(url, wait_until="networkidle")

        candidates_raw = await page.evaluate(_EXTRACTION_SCRIPT)
        screenshot_bytes = await page.screenshot(full_page=False)
        await browser.close()

    screenshot_buffer = io.BytesIO(screenshot_bytes)
    screenshot = Image.open(screenshot_buffer).convert("RGB")
    screenshot_buffer.close()

    candidates = [_candidate_from_live_dict(candidate) for candidate in candidates_raw]
    return LivePageSnapshot(
        url=url,
        screenshot=screenshot,
        screenshot_width=screenshot.width,
        screenshot_height=screenshot.height,
        candidates=candidates,
    )


def capture_live_page_sync(url: str, viewport_width: int = 1440, viewport_height: int = 1280) -> LivePageSnapshot:
    """Synchronous wrapper for scripts that don't want to manage an event loop."""

    return asyncio.run(capture_live_page(url=url, viewport_width=viewport_width, viewport_height=viewport_height))

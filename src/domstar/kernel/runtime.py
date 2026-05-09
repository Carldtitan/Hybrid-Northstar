"""Kernel-hosted browser task loops for base Northstar and DOM-grounded variants."""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

from domstar.dom.schema import DOMCandidate
from domstar.finetune.prompting import build_chat_messages, build_user_prompt, parse_action_response
from domstar.live.extractor import _EXTRACTION_SCRIPT, _candidate_from_live_dict
from domstar.ranker.runtime import DOMRanker


def _require_kernel():
    try:
        from kernel import Kernel
    except ImportError as exc:  # pragma: no cover - depends on optional SDK install
        raise ImportError("Install the Kernel SDK with `pip install kernel`.") from exc
    return Kernel


def _require_lightcone():
    try:
        from tzafon import Lightcone
    except ImportError as exc:  # pragma: no cover - depends on optional SDK install
        raise ImportError("Install the Lightcone SDK with `pip install tzafon`.") from exc
    return Lightcone


def _maybe_attr(payload: Any, key: str, default: Any = None) -> Any:
    if payload is None:
        return default
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _to_plain_action(action: Any) -> dict[str, Any]:
    if action is None:
        return {}
    if isinstance(action, dict):
        return dict(action)

    keys = [
        "type",
        "x",
        "y",
        "end_x",
        "end_y",
        "text",
        "value",
        "url",
        "keys",
        "scroll_x",
        "scroll_y",
        "result",
    ]
    result: dict[str, Any] = {}
    for key in keys:
        value = getattr(action, key, None)
        if value is not None:
            result[key] = value
    return result


def _normalize_action_name(action: dict[str, Any]) -> str:
    return str(action.get("action", action.get("type", ""))).strip().lower()


def _action_to_history_line(action: dict[str, Any]) -> str:
    action_name = _normalize_action_name(action)
    if action_name in {"click", "select"}:
        return f"clicked {action.get('element_id') or 'screen'} at ({action.get('x')},{action.get('y')})"
    if action_name == "type":
        return f"typed {json.dumps(str(action.get('value') or action.get('text') or ''))}"
    if action_name in {"key", "keypress"}:
        return f"pressed keys {action.get('keys')}"
    if action_name == "scroll":
        return f"scrolled by {action.get('scroll_y', action.get('value', 0))}"
    if action_name == "navigate":
        return f"navigated to {action.get('url', '')}"
    if action_name == "wait":
        return "waited"
    if action_name in {"terminate", "done", "answer"}:
        return f"terminated with {action.get('result', '')}"
    return json.dumps(action, ensure_ascii=True)


@dataclass(slots=True)
class KernelTaskSpec:
    name: str
    task: str
    start_url: str = ""
    max_steps: int = 40
    expected_url_contains: str = ""
    expected_text_contains: list[str] = field(default_factory=list)
    expected_selectors: list[str] = field(default_factory=list)
    settle_seconds: float = 1.0


@dataclass(slots=True)
class Snapshot:
    screenshot: Image.Image
    screenshot_width: int
    screenshot_height: int
    screenshot_bytes: bytes
    candidates: list[DOMCandidate]
    url: str
    title: str


@dataclass(slots=True)
class PolicyDecision:
    action: dict[str, Any]
    dom_seconds: float = 0.0
    model_seconds: float = 0.0
    raw_text: str = ""
    top_candidates: list[dict[str, Any]] = field(default_factory=list)
    answer_text: str = ""


@dataclass(slots=True)
class TaskRunResult:
    runner_name: str
    task_name: str
    completed: bool
    success: bool
    stop_reason: str
    total_seconds: float
    steps: int
    model_seconds_total: float
    dom_seconds_total: float
    live_view_url: str
    start_url: str
    final_url: str
    final_title: str
    answer_text: str
    history: list[str]
    step_records: list[dict[str, Any]]


class KernelBrowserSession:
    """One Kernel browser plus a CDP Playwright connection for DOM extraction."""

    def __init__(
        self,
        *,
        viewport_width: int = 1440,
        viewport_height: int = 1280,
        timeout_seconds: int = 1800,
        stealth: bool = True,
        headless: bool = False,
        profile_name: str = "",
        save_profile_changes: bool = False,
    ) -> None:
        Kernel = _require_kernel()
        self._kernel = Kernel()
        create_kwargs: dict[str, Any] = {
            "viewport": {"width": viewport_width, "height": viewport_height},
            "timeout_seconds": timeout_seconds,
            "stealth": stealth,
            "headless": headless,
        }
        if profile_name:
            profile_payload: dict[str, Any] = {"name": profile_name}
            if save_profile_changes:
                profile_payload["save_changes"] = True
            create_kwargs["profile"] = profile_payload

        self.browser_session = self._kernel.browsers.create(**create_kwargs)
        self.session_id = self.browser_session.session_id
        self.cdp_ws_url = self.browser_session.cdp_ws_url
        self.live_view_url = _maybe_attr(self.browser_session, "browser_live_view_url", "")
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

        self._pw = sync_playwright().start()
        self._browser: Browser = self._pw.chromium.connect_over_cdp(self.cdp_ws_url)
        self._context: BrowserContext = self._browser.contexts[0] if self._browser.contexts else self._browser.new_context()
        self._page: Page = self._context.pages[0] if self._context.pages else self._context.new_page()

    @property
    def page(self) -> Page:
        return self._page

    def goto(self, url: str) -> None:
        self._page.goto(url, wait_until="domcontentloaded")
        self._safe_settle()

    def _safe_settle(self) -> None:
        try:
            self._page.wait_for_load_state("networkidle", timeout=5_000)
        except Exception:
            pass

    def capture_snapshot(self) -> Snapshot:
        self._safe_settle()
        screenshot_payload = self._kernel.browsers.computer.capture_screenshot(id=self.session_id)
        screenshot_bytes = screenshot_payload.read() if hasattr(screenshot_payload, "read") else screenshot_payload
        image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
        candidates_raw = self._page.evaluate(_EXTRACTION_SCRIPT)
        candidates = [_candidate_from_live_dict(payload) for payload in candidates_raw]
        return Snapshot(
            screenshot=image,
            screenshot_width=image.width,
            screenshot_height=image.height,
            screenshot_bytes=screenshot_bytes,
            candidates=candidates,
            url=self._page.url,
            title=self._page.title(),
        )

    def evaluate_success(self, spec: KernelTaskSpec) -> tuple[bool, dict[str, Any]]:
        details: dict[str, Any] = {
            "url_contains": True,
            "text_contains": {},
            "selector_exists": {},
        }
        success = True

        if spec.expected_url_contains:
            url_ok = spec.expected_url_contains.lower() in self._page.url.lower()
            details["url_contains"] = url_ok
            success = success and url_ok

        body_text = ""
        if spec.expected_text_contains:
            try:
                body_text = self._page.locator("body").inner_text(timeout=5_000)
            except Exception:
                body_text = ""
            for needle in spec.expected_text_contains:
                found = needle.lower() in body_text.lower()
                details["text_contains"][needle] = found
                success = success and found

        for selector in spec.expected_selectors:
            try:
                found = self._page.locator(selector).count() > 0
            except Exception:
                found = False
            details["selector_exists"][selector] = found
            success = success and found

        return success, details

    def execute_action(self, action: dict[str, Any], snapshot: Snapshot) -> dict[str, Any]:
        action_name = _normalize_action_name(action)
        hydrated = dict(action)
        candidate_lookup = {candidate.element_id: candidate for candidate in snapshot.candidates}

        if action_name in {"click", "select", "double_click"}:
            x, y = self._resolve_coordinates(hydrated, snapshot, candidate_lookup)
            hydrated["x"] = x
            hydrated["y"] = y
            self._kernel.browsers.computer.click_mouse(
                id=self.session_id,
                x=x,
                y=y,
                num_clicks=2 if action_name == "double_click" else 1,
            )
        elif action_name == "type":
            text = str(hydrated.get("value") or hydrated.get("text") or "")
            if not text:
                raise ValueError("Type action is missing value/text.")
            self._kernel.browsers.computer.type_text(id=self.session_id, text=text)
        elif action_name in {"key", "keypress"}:
            keys = hydrated.get("keys")
            if isinstance(keys, str):
                keys = [keys]
            if not keys:
                raise ValueError("Key action is missing keys.")
            self._kernel.browsers.computer.press_key(id=self.session_id, keys=keys)
        elif action_name == "scroll":
            delta_y = hydrated.get("scroll_y", hydrated.get("value", 600))
            delta_y = int(float(delta_y))
            self._kernel.browsers.computer.scroll(id=self.session_id, delta_x=0, delta_y=delta_y)
        elif action_name == "move":
            x, y = self._resolve_coordinates(hydrated, snapshot, candidate_lookup)
            hydrated["x"] = x
            hydrated["y"] = y
            self._kernel.browsers.computer.move_mouse(id=self.session_id, x=x, y=y)
        elif action_name == "drag":
            start_x, start_y = self._resolve_coordinates(hydrated, snapshot, candidate_lookup)
            end_x = hydrated.get("end_x")
            end_y = hydrated.get("end_y")
            if end_x is None or end_y is None:
                raise ValueError("Drag action requires end_x and end_y.")
            end_px = self._normalize_to_pixels(float(end_x), float(end_y), snapshot)
            self._kernel.browsers.computer.drag_mouse(
                id=self.session_id,
                path=[[start_x, start_y], [end_px[0], end_px[1]]],
                button="left",
            )
        elif action_name == "navigate":
            url = str(hydrated.get("url") or "")
            if not url:
                raise ValueError("Navigate action requires a URL.")
            self._page.goto(url, wait_until="domcontentloaded")
        elif action_name == "wait":
            time.sleep(float(hydrated.get("seconds", 2.0)))
        elif action_name in {"terminate", "done", "answer"}:
            return hydrated
        else:
            raise ValueError(f"Unsupported action type: {action_name}")

        return hydrated

    def _resolve_coordinates(
        self,
        action: dict[str, Any],
        snapshot: Snapshot,
        candidate_lookup: dict[str, DOMCandidate],
    ) -> tuple[int, int]:
        x = action.get("x")
        y = action.get("y")
        if x is not None and y is not None:
            return self._normalize_to_pixels(float(x), float(y), snapshot)

        element_id = str(action.get("element_id") or "").strip()
        candidate = candidate_lookup.get(element_id)
        if candidate is None:
            raise ValueError("Action is missing coordinates and element_id did not match any live candidate.")

        center = candidate.center()
        if center is None:
            raise ValueError("Matched candidate does not have a bounding box.")
        return (round(center[0]), round(center[1]))

    def _normalize_to_pixels(self, x_999: float, y_999: float, snapshot: Snapshot) -> tuple[int, int]:
        x = max(0, min(snapshot.screenshot_width - 1, round((x_999 / 999.0) * snapshot.screenshot_width)))
        y = max(0, min(snapshot.screenshot_height - 1, round((y_999 / 999.0) * snapshot.screenshot_height)))
        return x, y

    def close(self) -> None:
        try:
            self._browser.close()
        except Exception:
            pass
        try:
            self._pw.stop()
        except Exception:
            pass
        try:
            self._kernel.browsers.delete_by_id(self.session_id)
        except Exception:
            pass


class BaseNorthstarPolicy:
    """Kernel browser + Lightcone Responses API using the official base Northstar model."""

    def __init__(
        self,
        task: str,
        *,
        model_name: str = "tzafon.northstar-cua-fast",
        viewport_width: int = 1440,
        viewport_height: int = 1280,
    ) -> None:
        Lightcone = _require_lightcone()
        self._client = Lightcone()
        self.task = task
        self.model_name = model_name
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self._previous_response_id = ""
        self._last_call_id = ""

    def decide(self, snapshot: Snapshot, history: list[str]) -> PolicyDecision:
        screenshot_b64 = base64.b64encode(snapshot.screenshot_bytes).decode("ascii")
        start = time.perf_counter()

        if not self._previous_response_id:
            response = self._client.responses.create(
                model=self.model_name,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": self.task},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{screenshot_b64}",
                                "detail": "auto",
                            },
                        ],
                    }
                ],
                tools=[
                    {
                        "type": "computer_use",
                        "display_width": self.viewport_width,
                        "display_height": self.viewport_height,
                        "environment": "browser",
                    }
                ],
            )
        else:
            response = self._client.responses.create(
                model=self.model_name,
                previous_response_id=self._previous_response_id,
                input=[
                    {
                        "type": "computer_call_output",
                        "call_id": self._last_call_id,
                        "output": {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{screenshot_b64}",
                            "detail": "auto",
                        },
                    }
                ],
                tools=[
                    {
                        "type": "computer_use",
                        "display_width": self.viewport_width,
                        "display_height": self.viewport_height,
                        "environment": "browser",
                    }
                ],
            )

        model_seconds = time.perf_counter() - start
        self._previous_response_id = str(_maybe_attr(response, "id", ""))

        output_items = list(_maybe_attr(response, "output", []) or [])
        for item in output_items:
            if _maybe_attr(item, "type", "") == "computer_call":
                self._last_call_id = str(_maybe_attr(item, "call_id", ""))
                action = _to_plain_action(_maybe_attr(item, "action", {}))
                action["action"] = action.get("action", action.get("type", ""))
                answer_text = str(action.get("result", ""))
                return PolicyDecision(action=action, model_seconds=model_seconds, answer_text=answer_text)

        answer_chunks: list[str] = []
        for item in output_items:
            if _maybe_attr(item, "type", "") != "message":
                continue
            for content_item in list(_maybe_attr(item, "content", []) or []):
                if _maybe_attr(content_item, "type", "") == "output_text":
                    text = str(_maybe_attr(content_item, "text", "")).strip()
                    if text:
                        answer_chunks.append(text)

        answer_text = "\n".join(answer_chunks).strip()
        return PolicyDecision(
            action={"action": "terminate", "result": answer_text},
            model_seconds=model_seconds,
            answer_text=answer_text,
        )


class DomstarPolicy:
    """Kernel browser + local/fine-tuned Northstar + live DOM extraction."""

    def __init__(
        self,
        task: str,
        *,
        base_model: str = "Tzafon/Northstar-CUA-Fast",
        adapter_path: str = "",
        ranker_model: str = "",
        top_k: int = 12,
        max_new_tokens: int = 128,
        bf16: bool = False,
        fp16: bool = False,
        load_in_4bit: bool = False,
        attn_implementation: str = "sdpa",
        trust_remote_code: bool = False,
        use_dom: bool = True,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        if bf16 and fp16:
            raise ValueError("Choose at most one of bf16 or fp16.")
        if load_in_4bit and not torch.cuda.is_available():
            raise ValueError("4-bit loading requires CUDA.")

        processor_source = adapter_path or base_model
        self.processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=trust_remote_code)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "attn_implementation": attn_implementation,
        }
        if bf16:
            model_kwargs["dtype"] = torch.bfloat16
        elif fp16:
            model_kwargs["dtype"] = torch.float16
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForImageTextToText.from_pretrained(base_model, **model_kwargs)
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self._torch = torch
        if torch.cuda.is_available() and not load_in_4bit:
            self.model.to("cuda")

        self.task = task
        self.ranker = DOMRanker(ranker_model) if ranker_model else None
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.use_dom = use_dom

    def decide(self, snapshot: Snapshot, history: list[str]) -> PolicyDecision:
        dom_seconds = 0.0
        ranked_items: list[Any] = []
        dom_summary = "(none)"

        if self.use_dom and self.ranker is not None and snapshot.candidates:
            dom_start = time.perf_counter()
            ranked_items = self.ranker.score(
                query=(
                    f"Task: {self.task}\n"
                    f"Previous actions:\n{chr(10).join(history) if history else '(none)'}\n"
                    "Predict the next interactable DOM element."
                ),
                candidates=snapshot.candidates,
            )
            top_candidates = [item.candidate for item in ranked_items[: self.top_k]]
            dom_summary = "\n".join(
                candidate.to_prompt_line(width=snapshot.screenshot_width, height=snapshot.screenshot_height)
                for candidate in top_candidates
            ) or "(none)"
            dom_seconds = time.perf_counter() - dom_start

        prompt_text = build_user_prompt(self.task, history=history, dom_summary=dom_summary)
        messages = build_chat_messages(image=snapshot.screenshot, prompt_text=prompt_text, target_text=None)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        if self._torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

        model_start = time.perf_counter()
        with self._torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        model_seconds = time.perf_counter() - model_start

        prompt_length = inputs["input_ids"].shape[1]
        decoded = self.processor.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        action = parse_action_response(decoded)
        if action:
            action["action"] = action.get("action", action.get("type", ""))

        if self.use_dom and ranked_items:
            if _normalize_action_name(action) in {"click", "select"} and action.get("element_id") and (
                action.get("x") is None or action.get("y") is None
            ):
                for ranked in ranked_items[: self.top_k]:
                    if ranked.candidate.element_id == str(action["element_id"]):
                        normalized = ranked.candidate.center_normalized(
                            width=snapshot.screenshot_width,
                            height=snapshot.screenshot_height,
                        )
                        if normalized is not None:
                            action["x"], action["y"] = normalized
                        break

        return PolicyDecision(
            action=action,
            dom_seconds=dom_seconds,
            model_seconds=model_seconds,
            raw_text=decoded,
            top_candidates=[
                {
                    "element_id": item.candidate.element_id,
                    "score": item.score,
                    "selector": item.candidate.selector,
                    "text": item.candidate.text,
                }
                for item in ranked_items[: self.top_k]
            ],
        )


class KernelTaskRunner:
    """Run one policy against one Kernel browser task and capture metrics."""

    def __init__(self, browser: KernelBrowserSession) -> None:
        self.browser = browser

    def run(
        self,
        *,
        runner_name: str,
        policy: BaseNorthstarPolicy | DomstarPolicy,
        spec: KernelTaskSpec,
        artifacts_dir: str = "",
    ) -> TaskRunResult:
        if spec.start_url:
            self.browser.goto(spec.start_url)

        history: list[str] = []
        step_records: list[dict[str, Any]] = []
        model_seconds_total = 0.0
        dom_seconds_total = 0.0
        answer_text = ""
        stop_reason = "max_steps_exceeded"

        started_at = time.perf_counter()
        for step_index in range(spec.max_steps):
            snapshot = self.browser.capture_snapshot()
            decision = policy.decide(snapshot, history)
            model_seconds_total += decision.model_seconds
            dom_seconds_total += decision.dom_seconds
            action_name = _normalize_action_name(decision.action)

            step_record = {
                "step": step_index + 1,
                "page_url": snapshot.url,
                "page_title": snapshot.title,
                "candidate_count": len(snapshot.candidates),
                "action": decision.action,
                "model_seconds": round(decision.model_seconds, 4),
                "dom_seconds": round(decision.dom_seconds, 4),
                "top_candidates": decision.top_candidates,
            }

            if action_name in {"terminate", "done", "answer"}:
                answer_text = str(decision.action.get("result") or decision.answer_text or "")
                stop_reason = action_name
                step_records.append(step_record)
                break

            executed = self.browser.execute_action(decision.action, snapshot)
            history.append(_action_to_history_line(executed))
            step_record["executed_action"] = executed
            step_records.append(step_record)
            time.sleep(spec.settle_seconds)

        total_seconds = time.perf_counter() - started_at
        success, success_details = self.browser.evaluate_success(spec)

        if artifacts_dir:
            Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
            artifact_path = Path(artifacts_dir, f"{runner_name}_{spec.name}.json")
            artifact_path.write_text(
                json.dumps(
                    {
                        "runner_name": runner_name,
                        "task": asdict(spec),
                        "success_details": success_details,
                        "history": history,
                        "step_records": step_records,
                        "live_view_url": self.browser.live_view_url,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        return TaskRunResult(
            runner_name=runner_name,
            task_name=spec.name,
            completed=stop_reason != "max_steps_exceeded",
            success=success,
            stop_reason=stop_reason,
            total_seconds=round(total_seconds, 4),
            steps=len(step_records),
            model_seconds_total=round(model_seconds_total, 4),
            dom_seconds_total=round(dom_seconds_total, 4),
            live_view_url=self.browser.live_view_url,
            start_url=spec.start_url,
            final_url=self.browser.page.url,
            final_title=self.browser.page.title(),
            answer_text=answer_text,
            history=history,
            step_records=step_records,
        )

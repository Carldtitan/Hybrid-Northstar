"""Compare base Northstar and a DOM-grounded local/fine-tuned model on the same Kernel tasks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from domstar.kernel.runtime import (
    BaseNorthstarPolicy,
    DomstarPolicy,
    KernelBrowserSession,
    KernelTaskRunner,
    KernelTaskSpec,
)
from domstar.utils.logging_utils import log_runtime_environment, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-file", required=True, help="JSON file with a list of task specs.")
    parser.add_argument("--output-path", default="artifacts/kernel_comparison.json")
    parser.add_argument("--artifacts-dir", default="artifacts/kernel_runs")

    parser.add_argument("--viewport-width", type=int, default=1440)
    parser.add_argument("--viewport-height", type=int, default=1280)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--stealth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--lightcone-model", default="tzafon.northstar-cua-fast")
    parser.add_argument("--base-model", default="Tzafon/Northstar-CUA-Fast")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--include-local-screenshot-baseline", action="store_true")
    parser.add_argument("--ranker-model", required=True)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default="")
    return parser.parse_args()


def load_tasks(path: str) -> list[KernelTaskSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks: list[KernelTaskSpec] = []
    for item in payload:
        tasks.append(
            KernelTaskSpec(
                name=str(item["name"]),
                task=str(item["task"]),
                start_url=str(item.get("start_url", "")),
                max_steps=int(item.get("max_steps", 40)),
                expected_url_contains=str(item.get("expected_url_contains", "")),
                expected_text_contains=list(item.get("expected_text_contains", [])),
                expected_selectors=list(item.get("expected_selectors", [])),
                settle_seconds=float(item.get("settle_seconds", 1.0)),
            )
        )
    return tasks


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted({key for row in results for key in row.keys() if key != "task"})
    summary: dict[str, Any] = {}
    for label in labels:
        runs = [row[label] for row in results]
        completed = sum(1 for run in runs if run["completed"])
        success = sum(1 for run in runs if run["success"])
        summary[label] = {
            "runs": len(runs),
            "completed": completed,
            "success": success,
            "avg_total_seconds": round(sum(run["total_seconds"] for run in runs) / max(1, len(runs)), 4),
            "avg_model_seconds_total": round(sum(run["model_seconds_total"] for run in runs) / max(1, len(runs)), 4),
            "avg_dom_seconds_total": round(sum(run["dom_seconds_total"] for run in runs) / max(1, len(runs)), 4),
            "avg_steps": round(sum(run["steps"] for run in runs) / max(1, len(runs)), 4),
        }
    return summary


def run_one(
    spec: KernelTaskSpec,
    *,
    runner_name: str,
    policy,
    args: argparse.Namespace,
):
    browser = KernelBrowserSession(
        viewport_width=args.viewport_width,
        viewport_height=args.viewport_height,
        timeout_seconds=args.timeout_seconds,
        stealth=args.stealth,
        headless=args.headless,
    )
    try:
        runner = KernelTaskRunner(browser)
        result = runner.run(
            runner_name=runner_name,
            policy=policy,
            spec=spec,
            artifacts_dir=args.artifacts_dir,
        )
        return result
    finally:
        browser.close()


def main() -> None:
    args = parse_args()
    logger = setup_logging(level=args.log_level, log_file=args.log_file)
    try:
        log_runtime_environment(logger)
        tasks = load_tasks(args.tasks_file)
        results: list[dict[str, Any]] = []

        for spec in tasks:
            logger.info("Running base model on task=%s", spec.name)
            base_policy = BaseNorthstarPolicy(
                task=spec.task,
                model_name=args.lightcone_model,
                viewport_width=args.viewport_width,
                viewport_height=args.viewport_height,
            )
            base_result = run_one(spec, runner_name="base-lightcone", policy=base_policy, args=args)

            local_screenshot_result = None
            if args.include_local_screenshot_baseline:
                logger.info("Running local screenshot-only baseline on task=%s", spec.name)
                screenshot_policy = DomstarPolicy(
                    task=spec.task,
                    base_model=args.base_model,
                    adapter_path="",
                    ranker_model="",
                    top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    bf16=args.bf16,
                    fp16=args.fp16,
                    load_in_4bit=args.load_in_4bit,
                    attn_implementation=args.attn_implementation,
                    trust_remote_code=args.trust_remote_code,
                    use_dom=False,
                )
                local_screenshot_result = run_one(
                    spec,
                    runner_name="local-screenshot",
                    policy=screenshot_policy,
                    args=args,
                )

            logger.info("Running DOM-grounded model on task=%s", spec.name)
            dom_policy = DomstarPolicy(
                task=spec.task,
                base_model=args.base_model,
                adapter_path=args.adapter_path,
                ranker_model=args.ranker_model,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                bf16=args.bf16,
                fp16=args.fp16,
                load_in_4bit=args.load_in_4bit,
                attn_implementation=args.attn_implementation,
                trust_remote_code=args.trust_remote_code,
                use_dom=True,
            )
            dom_result = run_one(spec, runner_name="domstar", policy=dom_policy, args=args)

            row = {
                "task": asdict(spec),
                "base": asdict(base_result),
                "domstar": asdict(dom_result),
            }
            if local_screenshot_result is not None:
                row["local_screenshot"] = asdict(local_screenshot_result)
            results.append(row)

        payload = {
            "results": results,
            "summary": summarize(results),
        }
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Comparison written to %s", output_path)
        print(json.dumps(payload, indent=2))
    except Exception:
        logger.exception("Kernel comparison failed")
        raise


if __name__ == "__main__":
    main()

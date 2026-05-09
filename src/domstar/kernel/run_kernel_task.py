"""Run one task on a Kernel browser using either base Northstar or a DOM-grounded local model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

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
    parser.add_argument("--policy", choices=["base-lightcone", "domstar", "domstar-screenshot"], required=True)
    parser.add_argument("--name", default="kernel-task")
    parser.add_argument("--task", required=True)
    parser.add_argument("--start-url", default="")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--expected-url-contains", default="")
    parser.add_argument("--expected-text-contains", nargs="*", default=[])
    parser.add_argument("--expected-selectors", nargs="*", default=[])
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--artifacts-dir", default="artifacts/kernel_runs")

    parser.add_argument("--viewport-width", type=int, default=1440)
    parser.add_argument("--viewport-height", type=int, default=1280)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--stealth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--profile-name", default="")
    parser.add_argument("--save-profile-changes", action="store_true")

    parser.add_argument("--lightcone-model", default="tzafon.northstar-cua-fast")
    parser.add_argument("--base-model", default="Tzafon/Northstar-CUA-Fast")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--ranker-model", default="")
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


def main() -> None:
    args = parse_args()
    logger = setup_logging(level=args.log_level, log_file=args.log_file)
    browser: KernelBrowserSession | None = None
    try:
        log_runtime_environment(logger)
        spec = KernelTaskSpec(
            name=args.name,
            task=args.task,
            start_url=args.start_url,
            max_steps=args.max_steps,
            expected_url_contains=args.expected_url_contains,
            expected_text_contains=args.expected_text_contains,
            expected_selectors=args.expected_selectors,
            settle_seconds=args.settle_seconds,
        )

        browser = KernelBrowserSession(
            viewport_width=args.viewport_width,
            viewport_height=args.viewport_height,
            timeout_seconds=args.timeout_seconds,
            stealth=args.stealth,
            headless=args.headless,
            profile_name=args.profile_name,
            save_profile_changes=args.save_profile_changes,
        )
        logger.info("Kernel browser ready | session_id=%s | live_view=%s", browser.session_id, browser.live_view_url)

        if args.policy == "base-lightcone":
            policy = BaseNorthstarPolicy(
                task=args.task,
                model_name=args.lightcone_model,
                viewport_width=args.viewport_width,
                viewport_height=args.viewport_height,
            )
        else:
            policy = DomstarPolicy(
                task=args.task,
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
                use_dom=(args.policy == "domstar"),
            )

        runner = KernelTaskRunner(browser)
        result = runner.run(
            runner_name=args.policy,
            policy=policy,
            spec=spec,
            artifacts_dir=args.artifacts_dir,
        )
        logger.info(
            "Kernel task finished | runner=%s | success=%s | stop_reason=%s | total_seconds=%.2f",
            result.runner_name,
            result.success,
            result.stop_reason,
            result.total_seconds,
        )
        print(json.dumps(asdict(result), indent=2))
    except Exception:
        logger.exception("Kernel task run failed")
        raise
    finally:
        if browser is not None:
            browser.close()


if __name__ == "__main__":
    main()

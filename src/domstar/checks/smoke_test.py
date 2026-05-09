"""Cheap smoke tests you can run on a laptop GPU before larger training jobs."""

from __future__ import annotations

import argparse
import json

from domstar.data.mind2web import build_ranker_query, build_target_action, row_to_action_example
from domstar.finetune.train import NorthstarDomDataset
from domstar.live.extractor import capture_live_page_sync
from domstar.ranker.runtime import DOMRanker
from domstar.utils.logging_utils import log_runtime_environment, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-rows", type=int, default=8)
    parser.add_argument("--ranker-model", default="")
    parser.add_argument("--url", default="")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(level=args.log_level, log_file=args.log_file)
    log_runtime_environment(logger)

    logger.info("Building a tiny NorthstarDomDataset smoke sample")
    dataset = NorthstarDomDataset(
        split=args.dataset_split,
        top_k=args.top_k,
        limit_rows=args.dataset_rows,
        max_negative_pool=16,
        ranker=DOMRanker(args.ranker_model) if args.ranker_model else None,
        seed=42,
        streaming=True,
        logger=logger,
    )
    logger.info("Prepared examples=%s", len(dataset))

    if len(dataset):
        sample = dataset[0]
        logger.info("First prompt length=%s chars", len(sample["prompt_text"]))
        logger.info("First target=%s", sample["target_text"])

    if args.ranker_model:
        logger.info("Validating the ranker on one raw example")
        from domstar.data.mind2web import iter_mind2web_rows

        first_row = next(iter_mind2web_rows(split=args.dataset_split, limit_rows=1, streaming=True))
        example = row_to_action_example(first_row, load_screenshot=False)
        ranked = DOMRanker(args.ranker_model).score(
            build_ranker_query(example),
            example.positive_candidates + example.negative_candidates,
        )
        logger.info("Top ranked candidate=%s", ranked[0].candidate.to_ranker_text() if ranked else "(none)")
        logger.info("Target action=%s", json.dumps(build_target_action(example)))

    if args.url:
        logger.info("Running live DOM extraction against %s", args.url)
        snapshot = capture_live_page_sync(args.url)
        logger.info(
            "Live page extracted | size=%sx%s | candidates=%s",
            snapshot.screenshot_width,
            snapshot.screenshot_height,
            len(snapshot.candidates),
        )
        if snapshot.candidates:
            logger.info("First live candidate=%s", snapshot.candidates[0].to_ranker_text())


if __name__ == "__main__":
    main()

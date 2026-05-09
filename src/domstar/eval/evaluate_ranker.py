"""Evaluate ranker recall and MRR on Multimodal-Mind2Web."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from domstar.data.mind2web import build_ranker_query, iter_mind2web_rows, row_to_action_example
from domstar.ranker.runtime import DOMRanker
from domstar.utils.logging_utils import log_runtime_environment, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranker-model", required=True)
    parser.add_argument("--split", default="test_task")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--output-path", default="artifacts/ranker_eval.json")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(level=args.log_level, log_file=args.log_file)
    try:
        log_runtime_environment(logger)
        logger.info("Loading ranker model %s", args.ranker_model)
        ranker = DOMRanker(args.ranker_model)
        dataset = iter_mind2web_rows(split=args.split, limit_rows=args.max_rows, streaming=args.streaming)

        total = 0
        hits = 0
        reciprocal_rank_sum = 0.0

        for index, raw_row in enumerate(dataset):
            example = row_to_action_example(raw_row, load_screenshot=False)
            if example.chosen_positive is None:
                continue

            pool = example.positive_candidates + example.negative_candidates
            ranked = ranker.score(build_ranker_query(example), pool)
            positive_ids = {candidate.element_id for candidate in example.positive_candidates}

            rank = None
            for item_index, item in enumerate(ranked, start=1):
                if item.candidate.element_id in positive_ids:
                    rank = item_index
                    break

            if rank is None:
                continue

            total += 1
            if rank <= args.top_k:
                hits += 1
            reciprocal_rank_sum += 1.0 / rank

        metrics = {
            "split": args.split,
            "evaluated_examples": total,
            f"recall@{args.top_k}": hits / total if total else 0.0,
            "mrr": reciprocal_rank_sum / total if total else 0.0,
        }

        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info("Ranker eval metrics: %s", json.dumps(metrics))
        print(json.dumps(metrics, indent=2))
    except Exception:
        logger.exception("Ranker evaluation failed")
        raise


if __name__ == "__main__":
    main()

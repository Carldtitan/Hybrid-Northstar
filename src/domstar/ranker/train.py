"""Train a lightweight DOM ranker on Multimodal-Mind2Web."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from domstar.data.mind2web import build_ranker_query, iter_mind2web_rows, row_to_action_example
from domstar.utils.logging_utils import log_runtime_environment, prune_checkpoints, setup_logging, validate_non_empty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default="artifacts/ranker")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test_task")
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=256)
    parser.add_argument("--max-negatives-per-example", type=int, default=24)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=32)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--cleanup-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default="")
    return parser.parse_args()


def _sample_negatives(items: list[Any], limit: int, rng: random.Random) -> list[Any]:
    if limit <= 0 or len(items) <= limit:
        return items
    return rng.sample(items, limit)


def _build_pairs(
    split_name: str,
    limit_rows: int,
    max_negatives: int,
    seed: int,
    streaming: bool = False,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """Convert each Mind2Web step into positive/negative ranker pairs."""

    rng = random.Random(seed)
    dataset = iter_mind2web_rows(split=split_name, limit_rows=limit_rows, streaming=streaming)
    rows: list[dict[str, Any]] = []
    kept_examples = 0

    for index, raw_row in enumerate(dataset):
        example = row_to_action_example(raw_row)
        if not example.positive_candidates:
            continue
        kept_examples += 1

        query = build_ranker_query(example)
        positives = example.positive_candidates
        negatives = _sample_negatives(example.negative_candidates, max_negatives, rng)

        for candidate in positives:
            rows.append(
                {
                    "text_a": query,
                    "text_b": candidate.to_ranker_text(),
                    "label": 1,
                }
            )

        for candidate in negatives:
            rows.append(
                {
                    "text_a": query,
                    "text_b": candidate.to_ranker_text(),
                    "label": 0,
                }
            )

    if logger is not None:
        logger.info(
            "Ranker pairs built | split=%s | examples=%s | pairs=%s",
            split_name,
            kept_examples,
            len(rows),
        )
    return rows


def _tokenize_batch(batch: dict[str, list[Any]], tokenizer):
    return tokenizer(
        batch["text_a"],
        batch["text_b"],
        truncation=True,
        max_length=512,
    )


def main() -> None:
    args = parse_args()
    logger = setup_logging(level=args.log_level, log_file=args.log_file)
    try:
        random.seed(args.seed)
        log_runtime_environment(logger)
        if args.bf16 and args.fp16:
            raise ValueError("Choose at most one of --bf16 or --fp16.")
        logger.info("Training DOM ranker from %s", args.model_name)

        train_rows = _build_pairs(
            split_name=args.train_split,
            limit_rows=args.max_train_rows,
            max_negatives=args.max_negatives_per_example,
            seed=args.seed,
            streaming=args.streaming,
            logger=logger,
        )
        eval_rows = _build_pairs(
            split_name=args.eval_split,
            limit_rows=args.max_eval_rows,
            max_negatives=args.max_negatives_per_example,
            seed=args.seed + 1,
            streaming=args.streaming,
            logger=logger,
        )
        validate_non_empty("ranker_train_rows", len(train_rows))
        validate_non_empty("ranker_eval_rows", len(eval_rows))

        dataset = DatasetDict(
            {
                "train": Dataset.from_list(train_rows),
                "eval": Dataset.from_list(eval_rows),
            }
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenized = dataset.map(
            lambda batch: _tokenize_batch(batch, tokenizer),
            batched=True,
            remove_columns=["text_a", "text_b"],
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
        )

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            bf16=args.bf16,
            fp16=args.fp16,
            weight_decay=0.01,
            max_grad_norm=args.max_grad_norm,
            logging_steps=25,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=args.save_total_limit,
            remove_unused_columns=False,
            report_to="none",
            seed=args.seed,
            data_seed=args.seed,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            optim="adamw_torch",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["eval"],
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            processing_class=tokenizer,
        )
        logger.info("Starting ranker training")
        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        if args.cleanup_checkpoints:
            prune_checkpoints(args.output_dir, logger)

        metrics = trainer.evaluate()
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with Path(args.output_dir, "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        logger.info("Saved ranker artifacts to %s", args.output_dir)
        logger.info("Final ranker metrics: %s", json.dumps(metrics))
    except Exception:
        logger.exception("DOM ranker training failed")
        raise


if __name__ == "__main__":
    main()

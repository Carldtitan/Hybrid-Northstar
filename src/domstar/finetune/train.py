"""LoRA fine-tuning for Northstar using DOM-grounded Multimodal-Mind2Web prompts."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from domstar.data.mind2web import (
    build_prompt_candidates,
    build_ranker_query,
    build_target_action,
    iter_mind2web_rows,
    row_to_action_example,
)
from domstar.dom.schema import DOMCandidate
from domstar.finetune.prompting import build_chat_messages, build_user_prompt, format_target_action
from domstar.ranker.runtime import DOMRanker
from domstar.utils.logging_utils import log_runtime_environment, prune_checkpoints, setup_logging, validate_non_empty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Tzafon/Northstar-CUA-Fast")
    parser.add_argument("--output-dir", default="artifacts/northstar-dom-lora")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test_task")
    parser.add_argument("--ranker-model", default="")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=256)
    parser.add_argument("--max-negative-pool", type=int, default=64)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--cleanup-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default="")
    return parser.parse_args()


def _is_linear_like(module: torch.nn.Module) -> bool:
    """Accept torch and bitsandbytes linear layers without importing internal classes."""

    return "Linear" in module.__class__.__name__


def discover_lora_target_modules(model: torch.nn.Module) -> list[str]:
    """Auto-pick the dense projection modules most worth adapting."""

    preferred_suffixes = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
        "w1",
        "w2",
        "w3",
    }
    selected: set[str] = set()

    for name, module in model.named_modules():
        if not _is_linear_like(module):
            continue
        leaf_name = name.split(".")[-1]
        if leaf_name in preferred_suffixes:
            selected.add(leaf_name)

    if selected:
        return sorted(selected)

    # Fallback for architectures with different projection names.
    fallback: set[str] = set()
    for name, module in model.named_modules():
        if _is_linear_like(module):
            leaf_name = name.split(".")[-1]
            if leaf_name != "lm_head":
                fallback.add(leaf_name)
    return sorted(fallback)


def pick_training_candidates(
    positive_candidates: list[DOMCandidate],
    negative_candidates: list[DOMCandidate],
    top_k: int,
    rng: random.Random,
    ranker: DOMRanker | None,
    ranker_query: str | None,
) -> list[DOMCandidate]:
    """Build the short DOM list shown to Northstar."""

    pool = positive_candidates + negative_candidates
    if not pool:
        return []

    if ranker is not None and ranker_query:
        ranked = ranker.score(ranker_query, pool)
        top_candidates = [item.candidate for item in ranked[:top_k]]
        positive_ids = {candidate.element_id for candidate in positive_candidates}
        if positive_ids and not any(candidate.element_id in positive_ids for candidate in top_candidates):
            top_candidates = top_candidates[:-1] + [positive_candidates[0]]
        return top_candidates

    sampled_negatives = negative_candidates[: max(0, top_k - len(positive_candidates))]
    candidates = positive_candidates + sampled_negatives
    rng.shuffle(candidates)
    return candidates[:top_k]


@dataclass(slots=True)
class PreparedExample:
    """One fully prepared training/eval item."""

    image: Any
    prompt_text: str
    target_text: str


class NorthstarDomDataset(Dataset):
    """Precompute prompt-ready examples to keep training deterministic."""

    def __init__(
        self,
        split: str,
        top_k: int,
        limit_rows: int,
        max_negative_pool: int,
        ranker: DOMRanker | None,
        seed: int,
        streaming: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self.rows: list[PreparedExample] = []
        self.stats = {
            "loaded_rows": 0,
            "skipped_no_positive": 0,
            "skipped_no_candidates": 0,
        }
        dataset = iter_mind2web_rows(split=split, limit_rows=limit_rows, streaming=streaming)
        rng = random.Random(seed)

        for index, raw_row in enumerate(dataset):
            self.stats["loaded_rows"] += 1
            example = row_to_action_example(raw_row)
            if example.chosen_positive is None:
                self.stats["skipped_no_positive"] += 1
                continue

            negatives = example.negative_candidates[:max_negative_pool]
            ranker_query = build_ranker_query(example) if ranker is not None else None
            candidates = pick_training_candidates(
                positive_candidates=example.positive_candidates,
                negative_candidates=negatives,
                top_k=top_k,
                rng=rng,
                ranker=ranker,
                ranker_query=ranker_query,
            )
            if not candidates:
                self.stats["skipped_no_candidates"] += 1
                continue

            dom_summary = build_prompt_candidates(example, candidates)
            prompt_text = build_user_prompt(
                task=example.task,
                history=example.history,
                dom_summary=dom_summary,
            )
            target_text = format_target_action(build_target_action(example))

            self.rows.append(
                PreparedExample(
                    image=example.screenshot.convert("RGB"),
                    prompt_text=prompt_text,
                    target_text=target_text,
                )
            )

        if logger is not None:
            logger.info(
                "Prepared split=%s | raw_rows=%s | kept=%s | skipped_no_positive=%s | skipped_no_candidates=%s",
                split,
                self.stats["loaded_rows"],
                len(self.rows),
                self.stats["skipped_no_positive"],
                self.stats["skipped_no_candidates"],
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.rows[index]
        return {
            "image": item.image,
            "prompt_text": item.prompt_text,
            "target_text": item.target_text,
        }


class NorthstarVisionCollator:
    """Tokenize multimodal chat examples and mask loss to assistant tokens only."""

    def __init__(self, processor) -> None:
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = processor.tokenizer.eos_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        full_encodings: list[dict[str, torch.Tensor]] = []
        prompt_lengths: list[int] = []

        for feature in features:
            prompt_messages = build_chat_messages(
                image=feature["image"],
                prompt_text=feature["prompt_text"],
                target_text=None,
            )
            full_messages = build_chat_messages(
                image=feature["image"],
                prompt_text=feature["prompt_text"],
                target_text=feature["target_text"],
            )

            prompt_encoding = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            full_encoding = self.processor.apply_chat_template(
                full_messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            prompt_lengths.append(int(prompt_encoding["input_ids"].shape[1]))
            full_encodings.append(full_encoding)

        batch = self._pad_batch(full_encodings, prompt_lengths)
        return batch

    def _pad_batch(
        self,
        encodings: list[dict[str, torch.Tensor]],
        prompt_lengths: list[int],
    ) -> dict[str, torch.Tensor]:
        max_length = max(encoding["input_ids"].shape[1] for encoding in encodings)
        batch_size = len(encodings)

        input_ids = torch.full((batch_size, max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels = torch.full((batch_size, max_length), -100, dtype=torch.long)

        batch: dict[str, torch.Tensor] = {}

        for row_index, (encoding, prompt_length) in enumerate(zip(encodings, prompt_lengths, strict=True)):
            row_input_ids = encoding["input_ids"][0]
            row_attention_mask = encoding["attention_mask"][0]
            sequence_length = row_input_ids.shape[0]

            input_ids[row_index, :sequence_length] = row_input_ids
            attention_mask[row_index, :sequence_length] = row_attention_mask
            labels[row_index, :sequence_length] = row_input_ids
            labels[row_index, :prompt_length] = -100
            labels[row_index, sequence_length:] = -100

        batch["input_ids"] = input_ids
        batch["attention_mask"] = attention_mask
        batch["labels"] = labels

        for key in encodings[0]:
            if key in {"input_ids", "attention_mask"}:
                continue
            batch[key] = torch.cat([encoding[key] for encoding in encodings], dim=0)

        return batch


def build_model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build `from_pretrained` kwargs while keeping GPU strategy configurable."""

    kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "attn_implementation": args.attn_implementation,
    }

    if args.bf16:
        kwargs["dtype"] = torch.bfloat16
    elif args.fp16:
        kwargs["dtype"] = torch.float16

    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    return kwargs


def validate_runtime_args(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Fail early on impossible or risky runtime combinations."""

    if args.bf16 and args.fp16:
        raise ValueError("Choose at most one of --bf16 or --fp16.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.max_negative_pool < 0:
        raise ValueError("--max-negative-pool cannot be negative.")
    if args.load_in_4bit and not torch.cuda.is_available():
        raise ValueError("--load-in-4bit requires CUDA and bitsandbytes support.")
    if args.bf16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not report BF16 support. Use --fp16 on the RTX 3060.")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info("Primary GPU=%s | vram_gb=%.2f", device_name, total_gb)
        if total_gb < 8 and not args.load_in_4bit:
            logger.warning("VRAM is tight for Northstar fine-tuning. Add --load-in-4bit on this machine.")
        if total_gb < 8 and not args.fp16 and not args.bf16:
            logger.warning("Mixed precision is off. On a 6 GB RTX 3060 you usually want --fp16.")
    if (args.max_train_rows or args.max_eval_rows) and not args.streaming:
        logger.warning(
            "Small-row run requested without --streaming. Hugging Face may still download/build the full split."
        )


def main() -> None:
    args = parse_args()
    logger = setup_logging(level=args.log_level, log_file=args.log_file)
    try:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        log_runtime_environment(logger)
        validate_runtime_args(args, logger)

        logger.info("Loading DOM ranker: %s", args.ranker_model or "(disabled)")
        ranker = DOMRanker(args.ranker_model) if args.ranker_model else None

        logger.info("Preparing training dataset")
        train_dataset = NorthstarDomDataset(
            split=args.train_split,
            top_k=args.top_k,
            limit_rows=args.max_train_rows,
            max_negative_pool=args.max_negative_pool,
            ranker=ranker,
            seed=args.seed,
            streaming=args.streaming,
            logger=logger,
        )
        logger.info("Preparing evaluation dataset")
        eval_dataset = NorthstarDomDataset(
            split=args.eval_split,
            top_k=args.top_k,
            limit_rows=args.max_eval_rows,
            max_negative_pool=args.max_negative_pool,
            ranker=ranker,
            seed=args.seed + 1,
            streaming=args.streaming,
            logger=logger,
        )
        validate_non_empty("train_dataset", len(train_dataset))

        logger.info("Loading processor and model from %s", args.model_name)
        processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name,
            **build_model_kwargs(args),
        )
        model.config.use_cache = False

        if args.load_in_4bit:
            logger.info("Preparing model for 4-bit LoRA training")
            model = prepare_model_for_kbit_training(model)

        target_modules = discover_lora_target_modules(model)
        logger.info("LoRA target modules: %s", ", ".join(target_modules))
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        if args.gradient_checkpointing:
            logger.info("Enabling gradient checkpointing")
            model.gradient_checkpointing_enable()

        collator = NorthstarVisionCollator(processor=processor)
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            bf16=args.bf16,
            fp16=args.fp16,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            eval_strategy="steps" if len(eval_dataset) else "no",
            save_strategy="steps",
            save_total_limit=args.save_total_limit,
            remove_unused_columns=False,
            report_to="none",
            seed=args.seed,
            data_seed=args.seed,
            dataloader_num_workers=0,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if len(eval_dataset) else None,
            data_collator=collator,
            processing_class=processor,
        )

        logger.info("Starting Northstar LoRA training")
        trainer.train()
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)
        if args.cleanup_checkpoints:
            prune_checkpoints(args.output_dir, logger)

        metrics = trainer.evaluate() if len(eval_dataset) else {}
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with Path(args.output_dir, "train_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        logger.info("Saved training artifacts to %s", args.output_dir)
        logger.info("Final eval metrics: %s", json.dumps(metrics))
    except Exception:
        logger.exception("Northstar LoRA training failed")
        raise


if __name__ == "__main__":
    main()

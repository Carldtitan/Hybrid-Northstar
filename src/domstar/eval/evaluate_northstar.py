"""Evaluate DOM-grounded Northstar on Multimodal-Mind2Web action prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from domstar.data.mind2web import (
    build_prompt_candidates,
    build_ranker_query,
    build_target_action,
    iter_mind2web_rows,
    row_to_action_example,
)
from domstar.finetune.prompting import build_chat_messages, build_user_prompt, parse_action_response
from domstar.finetune.train import pick_training_candidates
from domstar.ranker.runtime import DOMRanker
from domstar.utils.logging_utils import log_runtime_environment, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Tzafon/Northstar-CUA-Fast")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--ranker-model", default="")
    parser.add_argument("--disable-dom", action="store_true")
    parser.add_argument("--split", default="test_task")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-rows", type=int, default=128)
    parser.add_argument("--max-negative-pool", type=int, default=64)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-path", default="artifacts/northstar_eval.json")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default="")
    return parser.parse_args()


def load_model_and_processor(args: argparse.Namespace):
    processor_source = args.adapter_path or args.base_model
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=args.trust_remote_code)
    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "attn_implementation": args.attn_implementation,
    }
    if args.bf16:
        model_kwargs["dtype"] = torch.bfloat16
    elif args.fp16:
        model_kwargs["dtype"] = torch.float16
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    model = AutoModelForImageTextToText.from_pretrained(args.base_model, **model_kwargs)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    if torch.cuda.is_available() and not args.load_in_4bit:
        model.to("cuda")
    return model, processor


def generate_action(model, processor, image, prompt_text: str, max_new_tokens: int) -> dict[str, object]:
    """Run one forward pass and parse the JSON response."""

    messages = build_chat_messages(image=image, prompt_text=prompt_text, target_text=None)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    if torch.cuda.is_available():
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prompt_length = inputs["input_ids"].shape[1]
    decoded = processor.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    return parse_action_response(decoded)


def main() -> None:
    args = parse_args()
    logger = setup_logging(level=args.log_level, log_file=args.log_file)
    try:
        if args.bf16 and args.fp16:
            raise ValueError("Choose at most one of --bf16 or --fp16.")
        if args.load_in_4bit and not torch.cuda.is_available():
            raise ValueError("--load-in-4bit requires CUDA.")
        log_runtime_environment(logger)
        logger.info("Loading Northstar for evaluation from %s", args.base_model)
        model, processor = load_model_and_processor(args)
        ranker = DOMRanker(args.ranker_model) if args.ranker_model else None
        dataset = iter_mind2web_rows(split=args.split, limit_rows=args.max_rows, streaming=args.streaming)

        rows_evaluated = 0
        op_correct = 0
        element_correct = 0
        value_correct = 0
        valid_json = 0

        for index, raw_row in enumerate(dataset):
            example = row_to_action_example(raw_row)
            if example.chosen_positive is None:
                continue

            if args.disable_dom:
                dom_summary = "(none)"
            else:
                candidates = pick_training_candidates(
                    positive_candidates=example.positive_candidates,
                    negative_candidates=example.negative_candidates[: args.max_negative_pool],
                    top_k=args.top_k,
                    rng=random.Random(index),
                    ranker=ranker,
                    ranker_query=build_ranker_query(example) if ranker is not None else None,
                )
                dom_summary = build_prompt_candidates(example, candidates)
            prompt_text = build_user_prompt(example.task, example.history, dom_summary)
            prediction = generate_action(
                model=model,
                processor=processor,
                image=example.screenshot.convert("RGB"),
                prompt_text=prompt_text,
                max_new_tokens=args.max_new_tokens,
            )
            target = build_target_action(example)

            rows_evaluated += 1
            if prediction:
                valid_json += 1
            if str(prediction.get("action", "")).strip().lower() == str(target.get("action", "")).strip().lower():
                op_correct += 1
            if str(prediction.get("element_id", "")).strip() == str(target.get("element_id", "")).strip():
                element_correct += 1
            if str(prediction.get("value", "")).strip() == str(target.get("value", "")).strip():
                value_correct += 1

        metrics = {
            "split": args.split,
            "evaluated_examples": rows_evaluated,
            "json_valid_rate": valid_json / rows_evaluated if rows_evaluated else 0.0,
            "operation_accuracy": op_correct / rows_evaluated if rows_evaluated else 0.0,
            "element_accuracy": element_correct / rows_evaluated if rows_evaluated else 0.0,
            "value_accuracy": value_correct / rows_evaluated if rows_evaluated else 0.0,
        }

        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info("Northstar eval metrics: %s", json.dumps(metrics))
        print(json.dumps(metrics, indent=2))
    except Exception:
        logger.exception("Northstar evaluation failed")
        raise


if __name__ == "__main__":
    main()

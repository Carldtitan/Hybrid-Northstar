"""Run one DOM-grounded Northstar step against a live webpage."""

from __future__ import annotations

import argparse
import json
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from domstar.finetune.prompting import build_chat_messages, build_user_prompt, parse_action_response
from domstar.live.extractor import capture_live_page_sync
from domstar.ranker.runtime import DOMRanker
from domstar.utils.logging_utils import log_runtime_environment, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--base-model", default="Tzafon/Northstar-CUA-Fast")
    parser.add_argument("--adapter-path", default="")
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


def load_model(args: argparse.Namespace, logger: logging.Logger):
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
    logger.info("Loading base model %s", args.base_model)
    model = AutoModelForImageTextToText.from_pretrained(args.base_model, **model_kwargs)
    if args.adapter_path:
        logger.info("Loading LoRA adapter %s", args.adapter_path)
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    if torch.cuda.is_available() and not args.load_in_4bit:
        model.to("cuda")
    return model, processor


def main() -> None:
    args = parse_args()
    logger = setup_logging(level=args.log_level, log_file=args.log_file)
    try:
        if args.bf16 and args.fp16:
            raise ValueError("Choose at most one of --bf16 or --fp16.")
        if args.load_in_4bit and not torch.cuda.is_available():
            raise ValueError("--load-in-4bit requires CUDA.")
        log_runtime_environment(logger)
        model, processor = load_model(args, logger)
        ranker = DOMRanker(args.ranker_model)

        logger.info("Capturing live page %s", args.url)
        page = capture_live_page_sync(args.url)
        logger.info("Extracted %s DOM candidates", len(page.candidates))
        ranked = ranker.score(
            query=f"Task: {args.task}\nPrevious actions:\n(none)\nPredict the next interactable DOM element.",
            candidates=page.candidates,
        )
        top_candidates = [item.candidate for item in ranked[: args.top_k]]
        logger.info("Top ranked candidates retained=%s", len(top_candidates))
        dom_summary = "\n".join(
            candidate.to_prompt_line(width=page.screenshot_width, height=page.screenshot_height)
            for candidate in top_candidates
        )
        prompt_text = build_user_prompt(args.task, history=[], dom_summary=dom_summary)
        messages = build_chat_messages(image=page.screenshot, prompt_text=prompt_text, target_text=None)
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
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

        prompt_length = inputs["input_ids"].shape[1]
        decoded = processor.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        parsed = parse_action_response(decoded)
        logger.info("Predicted action=%s", json.dumps(parsed))

        print(
            json.dumps(
                {
                    "action": parsed,
                    "top_candidates": [
                        {
                            "element_id": item.candidate.element_id,
                            "score": item.score,
                            "selector": item.candidate.selector,
                            "text": item.candidate.text,
                        }
                        for item in ranked[: args.top_k]
                    ],
                },
                indent=2,
            )
        )
    except Exception:
        logger.exception("Live DOM-grounded step failed")
        raise


if __name__ == "__main__":
    main()

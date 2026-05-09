# DOM-Grounded Northstar

This repo adds **DOM reading abilities** to `Tzafon/Northstar-CUA-Fast` **without changing the model architecture**.

It does four things:

1. Extracts visible, interactable DOM elements from live webpages with Playwright.
2. Trains a small **DOM ranker** on `osunlp/Multimodal-Mind2Web` to decide which DOM elements matter most for the current task.
3. Fine-tunes Northstar with **LoRA** so it can use `screenshot + ranked DOM summary + action history` together.
4. Evaluates the system clearly in three modes:
   - screenshot-only base Northstar
   - DOM-grounded base Northstar
   - DOM-grounded + LoRA Northstar

## Project layout

`src/domstar/dom`
: DOM schema and candidate parsing.

`src/domstar/live`
: Playwright live extraction and a one-step live runner.

`src/domstar/ranker`
: Train and run the DOM relevance ranker.

`src/domstar/finetune`
: Prompting and LoRA fine-tuning for Northstar.

`src/domstar/eval`
: Ranker metrics and model action metrics.

`src/domstar/kernel`
: Kernel-hosted browser task runners for base Northstar and DOM-grounded/fine-tuned variants.

## Install

Install a CUDA-enabled PyTorch build that matches the NVIDIA machine first.

Example for CUDA 12.1:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install the repo and browser runtime:

```powershell
pip install -e .
playwright install chromium
```

## Recommended run order

### 1. Train the DOM ranker

```powershell
python -m domstar.ranker.train `
  --output-dir artifacts/ranker `
  --model-name distilbert-base-uncased `
  --max-negatives-per-example 24 `
  --num-train-epochs 2
```

### 2. Evaluate the ranker

```powershell
python -m domstar.eval.evaluate_ranker `
  --ranker-model artifacts/ranker `
  --split test_website `
  --top-k 10 `
  --output-path artifacts/ranker_test_website.json
```

Key metrics:

- `recall@10`: how often the true target element is in the top 10 candidates
- `mrr`: how early the true target appears in the ranked list

### 3. Fine-tune Northstar with DOM summaries

For H100/A100 class GPUs:

```powershell
python -m domstar.finetune.train `
  --model-name Tzafon/Northstar-CUA-Fast `
  --ranker-model artifacts/ranker `
  --output-dir artifacts/northstar-dom-lora `
  --bf16 `
  --gradient-checkpointing `
  --top-k 12 `
  --num-train-epochs 1 `
  --learning-rate 1e-4 `
  --gradient-accumulation-steps 8
```

For smaller GPUs, add:

```powershell
--load-in-4bit
```

What this trainer does:

- keeps Northstar architecture unchanged
- ranks DOM candidates first
- injects only the top DOM candidates into the text prompt
- trains a LoRA adapter to emit action JSON with `action`, `value`, `element_id`, `x`, `y`

### 4. Evaluate the three model variants

#### A. Screenshot-only base Northstar

```powershell
python -m domstar.eval.evaluate_northstar `
  --base-model Tzafon/Northstar-CUA-Fast `
  --split test_task `
  --disable-dom `
  --bf16 `
  --output-path artifacts/eval_base_screenshot_only.json
```

#### B. DOM-grounded base Northstar

```powershell
python -m domstar.eval.evaluate_northstar `
  --base-model Tzafon/Northstar-CUA-Fast `
  --ranker-model artifacts/ranker `
  --split test_task `
  --top-k 12 `
  --bf16 `
  --output-path artifacts/eval_base_dom.json
```

#### C. DOM-grounded + LoRA Northstar

```powershell
python -m domstar.eval.evaluate_northstar `
  --base-model Tzafon/Northstar-CUA-Fast `
  --adapter-path artifacts/northstar-dom-lora `
  --ranker-model artifacts/ranker `
  --split test_task `
  --top-k 12 `
  --bf16 `
  --output-path artifacts/eval_dom_lora.json
```

Model metrics:

- `json_valid_rate`
- `operation_accuracy`
- `element_accuracy`
- `value_accuracy`

`element_accuracy` is the main metric for whether the DOM grounding helped.

## Live DOM reading demo

Run a single live step on a real webpage:

```powershell
python -m domstar.live.run_live_step `
  --url "https://www.wikipedia.org" `
  --task "Click the English language link." `
  --base-model Tzafon/Northstar-CUA-Fast `
  --adapter-path artifacts/northstar-dom-lora `
  --ranker-model artifacts/ranker `
  --top-k 12 `
  --bf16
```

The script prints:

- the predicted action JSON
- the top ranked DOM candidates
- their selectors and scores

## Kernel-hosted task demos

This is the demo shape you want for the hackathon:

- **base Northstar on a Kernel browser**
- **your DOM-grounded / fine-tuned Northstar on the same kind of Kernel browser**
- same task spec, same viewport, same success checks

## TypeScript Kernel Harness

An additional TypeScript harness lives in [`kernel-harness/`](kernel-harness/). It provides:

- a Kernel + Lightcone Responses API live runner for base Northstar
- safe task definitions for Craigslist, Airbnb, DoorDash, and UberEats-style workflows
- sanitized page-state logging, screenshot dimension checks, coordinate tracing, and loop guards
- model adapter placeholders for comparing base Northstar against a future fine-tuned/hybrid model in the same Kernel browser environment

Keep real credentials in local environment variables or `kernel-harness/.env`; do not commit secrets.

Use safe tasks first. For commerce flows like UberEats, stop before final checkout unless you explicitly want to place a live order.

### Run base Northstar on Kernel

This uses **Kernel for the browser** and **Lightcone Responses API for the base model**.

```powershell
python -m domstar.kernel.run_kernel_task `
  --policy base-lightcone `
  --name wikipedia-ada `
  --task "Go to wikipedia.org, search for Ada Lovelace, and open the article." `
  --start-url "https://www.wikipedia.org" `
  --expected-url-contains "Ada_Lovelace" `
  --expected-text-contains "Ada Lovelace" `
  --artifacts-dir artifacts/kernel_runs
```

### Run your DOM-grounded local / fine-tuned model on Kernel

This uses **Kernel for the browser** and your **local/fine-tuned HF model** for decisions.

```powershell
python -m domstar.kernel.run_kernel_task `
  --policy domstar `
  --name wikipedia-ada `
  --task "Go to wikipedia.org, search for Ada Lovelace, and open the article." `
  --start-url "https://www.wikipedia.org" `
  --expected-url-contains "Ada_Lovelace" `
  --expected-text-contains "Ada Lovelace" `
  --base-model Tzafon/Northstar-CUA-Fast `
  --adapter-path artifacts/northstar-dom-lora `
  --ranker-model artifacts/ranker `
  --top-k 12 `
  --bf16 `
  --artifacts-dir artifacts/kernel_runs
```

### Run a direct comparison

```powershell
python -m domstar.kernel.compare_kernel_models `
  --tasks-file examples/kernel_tasks.example.json `
  --ranker-model artifacts/ranker `
  --adapter-path artifacts/northstar-dom-lora `
  --base-model Tzafon/Northstar-CUA-Fast `
  --bf16 `
  --output-path artifacts/kernel_comparison.json `
  --artifacts-dir artifacts/kernel_runs
```

The comparison file includes:

- `success`
- `steps`
- `total_seconds`
- `model_seconds_total`
- `dom_seconds_total`
- step-by-step action traces

That is the cleanest way to show whether the DOM-grounded model is actually better or just slower.

## Notes on compute

This code is designed to scale with NVIDIA hackathon compute:

- Use `--bf16` on H100/A100 class GPUs.
- Use `--load-in-4bit` only if memory is tight.
- The ranker is cheap.
- The LoRA stage is the expensive part.
- `top-k=12` is a good default because it keeps the DOM prompt small enough to stay stable while still giving the model useful structure.

## Reports and visuals

Remote runs on Brev do **not** update your local VS Code automatically.

- Code changes need `git push` from one side and `git pull` on the other.
- `artifacts/` and `logs/` are ignored by git here, so they stay on the machine that produced them unless you copy them back.

Both training scripts now save `log_history.json` into their output directories. You can turn those histories, eval JSON files, and log files into a compact report with charts:

```powershell
python -m domstar.reporting.generate_report `
  --output-dir artifacts/reports/latest `
  --ranker-dir artifacts/ranker_distilbert `
  --ranker-eval-json artifacts/ranker_eval.json `
  --ranker-log-file logs/ranker_distilbert_train.log `
  --base-eval-json artifacts/eval_base_screenshot_only.json `
  --dom-eval-json artifacts/eval_base_dom.json `
  --dom-lora-eval-json artifacts/eval_dom_lora.json `
  --kernel-comparison-json artifacts/kernel_comparison.json
```

The report writes:

- `report.md`: summary, sanity checks, and links to charts
- `summary.json`: machine-readable summary
- PNG charts for:
  - training curves
  - eval bars
  - Kernel success / latency comparison

## Expected demo story

1. Show screenshot-only base Northstar.
2. Show DOM-grounded base Northstar with the same page and task.
3. Show DOM-grounded + LoRA Northstar.
4. Point to `recall@10`, `element_accuracy`, and a live failure case that DOM grounding fixes.

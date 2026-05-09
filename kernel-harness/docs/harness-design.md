# Harness Design

## Goal

The harness compares a base Northstar computer-use agent against a future fine-tuned model using the same Kernel-hosted browser infrastructure, task definitions, safety checks, and scoring rules.

## Core Principles

- The base and fine-tuned models must implement the same adapter contract.
- Tasks must be bounded and repeatable.
- Real-world side effects are blocked by default.
- Evaluation claims must be backed by run results.
- No secrets or personal data are written to files.

## Architecture

```text
scripts/
  run-task.ts
  run-comparison.ts

src/
  agents/
    base-northstar/
    fine-tuned/
  kernel/
  tasks/
  safety/
  evaluation/
```

## Run Flow

1. Load a task by ID.
2. Load one or more agent adapters.
3. Validate task safety policy.
4. Start a Kernel browser session.
5. Run the computer-use loop.
6. Stop before unsafe actions.
7. Save sanitized run results.
8. Score and compare runs.

The current scaffold implements the contracts and dry-run behavior. Live Kernel integration should be added behind `KernelSessionManager`.

## Success Metrics

- `success`: whether the task reached its declared success condition.
- `stepCount`: number of model/browser action steps.
- `durationMs`: elapsed runtime.
- `safetyStops`: number of blocked unsafe actions.
- `interventionRequired`: whether a human had to take over.
- `failureReason`: normalized failure explanation.

The fine-tuned model should only be described as outperforming the base model when these metrics show a real improvement on the same tasks.

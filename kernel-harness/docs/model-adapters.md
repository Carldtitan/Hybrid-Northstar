# Model Adapters

Both models must satisfy the same `ComputerUseAgent` interface.

## Base Northstar

`base_northstar` represents Tzafon Northstar CUA Fast through Lightcone/Tzafon and Kernel browser controls.

## Fine-Tuned Model

`fine_tuned_northstar` is a placeholder for the future fine-tuned endpoint. It must accept the same observation and task context as the base model and return the same action format.

## Adapter Rules

- Do not leak provider-specific details outside the adapter.
- Do not hardcode endpoints or credentials.
- Read credentials from environment variables or Kernel secrets.
- Return structured actions, not arbitrary scripts.
- Make unsafe intent explicit so safety guardrails can block it.

## Comparison Rule

The same task, starting URL, browser profile policy, safety policy, and scoring rules must be used for both models.

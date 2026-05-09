# Northstar Kernel Harness

This project is a safe evaluation harness for comparing:

- `base_northstar`: Tzafon Northstar CUA Fast running in a Kernel-hosted browser.
- `fine_tuned_northstar`: a future fine-tuned model wired into the same Kernel/task interface.

The harness is designed to demonstrate whether the fine-tuned model outperforms the base model on the same browser tasks because it receives better task context.

## Current State

This scaffold contains:

- model adapter contracts
- safe task definitions
- safety guardrails
- deterministic scoring utilities
- dry-run scripts for task and comparison flows
- documentation for Kernel, model adapters, and privacy rules

It does not contain credentials, live Kernel calls, or real order/payment actions.

## Safe Defaults

The default mode is dry-run. Commerce tasks must stop before checkout, purchase, send, subscribe, delete, or any irreversible action.

## Commands

After installing dependencies:

```bash
npm install
npm run typecheck
npm test
npm run run:task -- --agent base_northstar --task ubereats_search_restaurant
npm run run:comparison -- --task ubereats_search_restaurant
```

Task 1 live Kernel/Northstar run:

```bash
npm run run:live:northstar -- \
  --task ubereats_search_restaurant \
  --query "Open UberEats, search for sushi, open a restaurant page, and stop before checkout." \
  --max-steps 20 \
  --confirm-live
```

Live runs require `KERNEL_API_KEY` and `TZAFON_API_KEY` or `LIGHTCONE_API_KEY` in `.env`.

DoorDash live baseline:

```bash
npm run run:live:northstar -- \
  --task doordash_search_restaurant \
  --query "Open DoorDash. If asked for an address, use the public landmark Ferry Building, San Francisco. If address suggestions appear, click the first Ferry Building suggestion. Search for sushi, open the first visible restaurant or store result card, and stop before checkout." \
  --max-steps 25 \
  --step-timeout-ms 45000 \
  --confirm-live
```

Airbnb visual-search baseline:

```bash
npm run run:live:northstar -- \
  --task airbnb_red_exterior_san_francisco \
  --query "Open Airbnb and search for stays in San Francisco. Inspect listing card photos or listing photo galleries to find homes with visibly red-painted exterior walls. Return listing names and URLs when possible. Do not log in, message a host, reserve, book, checkout, or pay." \
  --max-steps 35 \
  --step-timeout-ms 45000 \
  --max-text-only-nudges 2 \
  --confirm-live
```

## Documentation

- [Harness design](docs/harness-design.md)
- [Safety policy](docs/safety-policy.md)
- [Kernel deployment](docs/kernel-deployment.md)
- [Model adapters](docs/model-adapters.md)

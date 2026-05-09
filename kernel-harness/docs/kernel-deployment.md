# Kernel Deployment Notes

Kernel provides a Tzafon example template:

```bash
kernel create --name my-tzafon-app --template tzafon
```

The template supports TypeScript or Python. This harness is TypeScript-oriented and should integrate Kernel-specific browser/session behavior inside `src/kernel/`.

## Intended Integration Points

- `KernelSessionManager.startSession`
- `KernelSessionManager.performAction`
- `KernelSessionManager.captureObservation`
- `KernelSessionManager.stopSession`

The live Task 1 runner is implemented in `src/live/northstar-kernel-runner.ts`.
It follows the Kernel plus Lightcone loop:

1. Create a Kernel browser session.
2. Navigate to the task start URL.
3. Capture a screenshot from Kernel.
4. Send the screenshot and task instruction to Northstar through Lightcone Responses.
5. Execute the returned computer action through Kernel Computer Controls.
6. Repeat with `previous_response_id`.
7. Delete the Kernel browser session in cleanup.

## Secrets

Use Kernel secrets or environment variables for:

- `KERNEL_API_KEY`
- `LIGHTCONE_API_KEY`
- fine-tuned model endpoint credentials

Do not commit real values.

## Live View

Kernel live view should be used for debugging runs, but screenshots and traces must be checked for sensitive information before being saved.

## Session Cleanup

Every live run must terminate the browser session, even on failure.

## Local Live Run

Live runs require `.env` values but never print them:

```bash
KERNEL_API_KEY=<set locally>
TZAFON_API_KEY=<set locally>
```

`LIGHTCONE_API_KEY` is also accepted as a fallback for `TZAFON_API_KEY`.

Run only after approving live hosted browser usage:

```bash
npm run run:live:northstar -- \
  --task ubereats_search_restaurant \
  --query "Open UberEats, search for sushi, open a restaurant page, and stop before checkout." \
  --max-steps 20 \
  --confirm-live
```

# Safety Policy

## Non-Negotiable Rules

The harness must never perform irreversible actions without explicit approval. This includes:

- checkout
- purchase
- payment
- send message
- submit order
- account deletion
- subscription changes
- address changes
- password or recovery changes

## Sensitive Information

Never write the following to files, logs, reports, screenshots, or fixtures:

- API keys
- Kernel credentials
- Tzafon or Lightcone credentials
- model-provider tokens
- passwords
- OAuth tokens
- session cookies
- payment data
- personal addresses
- private messages
- private account pages
- personal email or phone data

Use placeholders instead:

```text
KERNEL_API_KEY=<set in environment>
LIGHTCONE_API_KEY=<set in Kernel secrets>
FINE_TUNED_MODEL_ENDPOINT=<set later>
```

## Commerce Demo Policy

Commerce demos may navigate, search, inspect menus, and compare options. They must stop before checkout or payment. Cart changes are only allowed when explicitly approved for a safe demo account.

## Logging Policy

Logs should contain:

- task ID
- agent ID
- sanitized action summaries
- score metrics
- non-sensitive failure reasons

Logs should not contain:

- raw headers
- cookies
- tokens
- passwords
- screenshots with personal account data
- full page text from logged-in private pages

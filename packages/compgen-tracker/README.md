# compgen-tracker

Standalone Gemini API usage + cost + budget tracker. Carved out of
[CompGen](../../) so any project that calls Gemini can pip-install
just the tracker without pulling in a compiler stack.

## What it gives you

- **Append-only JSONL ledger** at `<repo>/.compgen/gemini_usage/events.jsonl` —
  one row per API call: timestamp, model, prompt/completion tokens,
  cached tokens, USD cost, latency, source tag, free-form metadata.
- **Live dashboard:** `compgen-gemini-usage watch` (Rich-based, reflows
  on resize). Add `--group-by source` / `--group-by pipe` / `--group-by target`
  to split spend by any tag the caller put on its events.
- **Budget alerts:** `compgen-gemini-usage budget set --cumulative-usd 200`
  — the watch dashboard turns yellow at 80%, red at 100%.
- **Auto-instrumentation:** call `install_genai_instrumentation()` once
  and every subsequent `google.genai` call lands in the ledger automatically.
- **Per-pipeline tagging:** `with tracking_source("pipe=A", **metadata):`
  attributes every nested call to that label.

## Install

```bash
pip install compgen-tracker
# Optional extras:
pip install compgen-tracker[yaml]    # PyYAML for pricing overrides
pip install compgen-tracker[genai]   # google-genai auto-instrumentation
pip install compgen-tracker[openai]  # openai SDK auto-instrumentation
```

## Use from Python

```python
from compgen_tracker import (
    install_genai_instrumentation,
    tracking_source,
    load_summary,
    record_call,
)

# One-line setup: every google.genai call from this process logs cost.
install_genai_instrumentation()

# Tag a block of calls by pipeline / target / region.
with tracking_source("pipe=C", pipe="C", target="gemmini", region="matmul_0"):
    ...  # call Gemini

# Inspect totals at any time.
summary = load_summary()
print(f"${summary.total_cost_usd:.2f} across {summary.total_calls} calls")
```

## Storage location

Resolution order (highest priority first):

1. `$COMPGEN_GEMINI_USAGE_DIR` env var — explicit override.
2. `$COMPGEN_REPO_ROOT/.compgen/gemini_usage/` — repo override.
3. CWD walked up to nearest `pyproject.toml` → `<that>/.compgen/gemini_usage/`.
4. `~/.compgen/gemini_usage/` — last-resort fallback.

To share one ledger across multiple repos (e.g. CompGen + a sibling
project) export the env var in each shell:

```bash
export COMPGEN_GEMINI_USAGE_DIR=/path/to/shared/.compgen/gemini_usage
```

Both repos then count against the same cumulative cap.

# Phase D — Status

_Last updated: 2026-05-07_  ·  _Current head: `e3d7454` (pre-M-55 commit)_  ·  _Phase plan: `~/.claude/plans/stateful-jumping-lovelace.md`_

This is the canonical Phase D tracker. Every Phase D milestone's done
condition includes updating this document with the new commit hash,
evidence paths, and test count. If the doc is not updated, the
milestone is not done.

## Paper-facing claim being built toward

> CompGen treats kernel codegen as a multi-bidder auction over a
> canonical contract; ≥4 distinct providers (Claude-Code agent,
> Triton template, C reference, user-supplied) bid against the same
> KernelContractV3, all verified bids carry certificates, the selector
> picks by perf, and the canonical-shape-class hash makes one verified
> kernel reusable across many concrete regions.

Intermediate paper-claimable milestones: M-57 (multi-bidder auction
landed), M-58 (canonical-hash cross-model cache leverage), M-66 (the
four-bidder benchmark).

## Architecture (one-line summary)

```
Recipe IR → KernelContractV3 → canonical_hash + instance_hash
  → ProviderRegistry.applicable → top-K bid → top-K fulfill
  → contract-driven verifier (per fulfilled bid) → certificate per winner
  → selector picks by perf → ExecutionPlan binding → emitted glue
```

User-space:

```
--user-kernel-path /repo  → /compgen-discover-user-kernels skill
                          → walks path, classifies, persists provider manifests
                          → UserKernelProvider bids against future contracts
```

## Milestone table

Status legend: `planned` → `in_progress` → `complete` (tests green + commit landed + this doc updated).

| ID    | Name                                                | Status        | Commit  | Evidence                                                                                            | Test count |
| ----- | --------------------------------------------------- | ------------- | ------- | --------------------------------------------------------------------------------------------------- | ---------- |
| M-55  | Wire ProviderRegistry into Phase C kernel-codegen   | in_progress   | —       | `python/compgen/kernels/registry.py` (applicable() + default_registry), `python/compgen/graph_compilation/kernel_codegen.py` (registry_resolution emit), `tests/graph_compilation/test_registry_wired.py`, `docs/realness/m55_registry_wired.yaml` | 7 (M-55), 62/62 Phase C regression preserved |
| M-56  | Two-stage provider protocol (bid + fulfill)         | planned       | —       | —                                                                                                   | —          |
| M-57  | Multi-bidder auction with tool-mediated pruning     | planned       | —       | —                                                                                                   | —          |
| M-58  | Canonical shape-class hash                          | planned       | —       | —                                                                                                   | —          |
| M-59  | contract_feedback re-enters Recipe IR (two-tier)    | planned       | —       | —                                                                                                   | —          |
| M-60  | Contract field completion from target + dossier     | planned       | —       | —                                                                                                   | —          |
| M-61  | Pre/post-conditions as typed predicates             | planned       | —       | —                                                                                                   | —          |
| M-62  | User-space kernel-provider discovery + MCP + skill  | planned       | —       | —                                                                                                   | —          |
| M-63  | Coverage-first scheduling                           | planned       | —       | —                                                                                                   | —          |
| M-64  | Refinement contracts + version migration            | planned       | —       | —                                                                                                   | —          |
| M-65  | Vertical slices 2 + 3 under multi-bidder auction    | planned       | —       | —                                                                                                   | —          |
| M-66  | Four-bidder benchmark + paper-claim closure         | planned       | —       | —                                                                                                   | —          |
| M-67  | Phase D status doc + Section 7 closure              | planned       | —       | —                                                                                                   | —          |

## Vertical-slice status

| Slice | Description                                                                | Target milestone gate | Status |
| ----- | -------------------------------------------------------------------------- | --------------------- | ------ |
| 1     | merlin_mlp_wide on host_cpu via cffi-C, single-bidder (Phase C closure)    | M-49                  | complete (Phase C) |
| 2     | proxy_vla on host_cpu (fusion path), Claude-Code + CReference auction      | M-65                  | planned |
| 3     | merlin_mlp_wide on cuda_sm75 via Triton, TritonTemplate + Claude-Code auction (SYNC + ASYNC) | M-65                  | planned |
| 4     | merlin_mlp_wide host_cpu, four-bidder auction inc. user-supplied kernel    | M-66                  | planned |

## Open questions / blockers

- _(none — design questions resolved during planning. Slot reserved for issues that surface during implementation.)_

## Honest residuals (cross-reference caveat ledger)

- Phase D inherits `m15b_natural_failure_unreachable` (M-37.13) — Phase D may resolve when multi-bidder execution surfaces genuine numerical disagreement across providers.
- M-55 honest residuals (see contract): subagent not yet a registered provider (M-56); `applicable()` does not consider numerics or memory residency yet (M-61); user-path discovery deferred (M-62).

## Last 3 trust reports

(Append-only — every full Phase D audit run lands here with its commit + verdict.)

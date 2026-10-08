"""Reporting + Pareto plots for the four-way bench.

Consumes ``results/four_way/<run_id>/runs.jsonl`` and produces:

  * ``pareto_<target>.png`` — tokens-vs-cycles scatter per pipeline,
    one figure per hardware target. The expected shape if H1 holds is
    pipelines C and D appearing below and left of A and B
    respectively.
  * ``ablation_table.md`` — iso-token-budget and iso-perf tables that
    quote the actual numbers the paper claim rests on.
  * ``summary.json`` — a structured digest (sub-100 lines) the trust
    report can ingest.

Matplotlib is the only heavy dep — gracefully degrade to "tables only"
if it's not available so CI can still validate the data path.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------- #


def load_runs(runs_path: Path) -> list[dict[str, Any]]:
    """Load runs.jsonl into a flat list of dicts."""
    rows: list[dict[str, Any]] = []
    for line in runs_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# --------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------- #


def _bucket_key(row: dict[str, Any]) -> tuple[str, str]:
    return (row.get("pipe") or "?", row.get("target") or "?")


def aggregate_by_pipe_target(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Aggregate rows by ``(pipe, target)`` and compute summary stats.

    The Pareto plot consumes the raw (tokens, cycles) points. The
    iso-token / iso-perf tables consume the aggregates: mean tokens,
    median cycles, success rate.
    """
    buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "rows": [],
            "tokens_points": [],
            "cycles_points": [],
            "successes": 0,
            "total": 0,
        }
    )
    for row in rows:
        key = _bucket_key(row)
        b = buckets[key]
        b["rows"].append(row)
        b["total"] += 1
        if row.get("succeeded"):
            b["successes"] += 1
            b["tokens_points"].append(row.get("total_tokens", 0))
            b["cycles_points"].append(row.get("best_cycles") or math.inf)

    # Add derived summary fields per bucket.
    for key, b in buckets.items():
        tokens = b["tokens_points"]
        cycles = [c for c in b["cycles_points"] if c != math.inf]
        b["mean_tokens"] = statistics.mean(tokens) if tokens else 0
        b["median_tokens"] = statistics.median(tokens) if tokens else 0
        b["mean_cycles"] = statistics.mean(cycles) if cycles else None
        b["median_cycles"] = statistics.median(cycles) if cycles else None
        b["best_cycles"] = min(cycles) if cycles else None
        b["success_rate"] = (b["successes"] / b["total"]) if b["total"] else 0.0
    return dict(buckets)


# --------------------------------------------------------------------- #
# Iso-token / Iso-perf tables (H1, H2)
# --------------------------------------------------------------------- #


def iso_token_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each (target, budget): list every pipeline's perf with that token bound.

    Tests H2 ("perf at iso-token-budget") by holding budget constant.
    """
    by = defaultdict(list)
    for row in rows:
        by[(row.get("target"), row.get("budget"))].append(row)

    out: list[dict[str, Any]] = []
    for (target, budget), bucket in sorted(by.items()):
        row_out: dict[str, Any] = {"target": target, "budget": budget}
        for pipe in ("A", "B", "C", "D"):
            cells = [r for r in bucket if r.get("pipe") == pipe and r.get("succeeded")]
            if not cells:
                row_out[f"pipe_{pipe}_best_cycles"] = None
                row_out[f"pipe_{pipe}_mean_tokens"] = None
                continue
            row_out[f"pipe_{pipe}_best_cycles"] = min(c["best_cycles"] for c in cells)
            row_out[f"pipe_{pipe}_mean_tokens"] = int(
                statistics.mean(c["total_tokens"] for c in cells)
            )
        out.append(row_out)
    return out


def iso_perf_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each target: tokens-to-reach-best-perf across pipelines.

    Tests H1 ("tokens at iso-perf") by holding perf (the best achieved
    by any pipeline on that target) constant and reading off each
    pipeline's token spend to reach within 10% of it.
    """
    by_target = defaultdict(list)
    for row in rows:
        if row.get("succeeded") and row.get("best_cycles") is not None:
            by_target[row.get("target")].append(row)

    out: list[dict[str, Any]] = []
    for target, bucket in sorted(by_target.items()):
        if not bucket:
            continue
        best_overall = min(r["best_cycles"] for r in bucket)
        threshold = best_overall * 1.10  # within 10%

        row_out: dict[str, Any] = {
            "target": target,
            "best_cycles_any_pipeline": best_overall,
            "threshold_cycles_within_10pct": int(threshold),
        }
        for pipe in ("A", "B", "C", "D"):
            cells = sorted(
                (r for r in bucket if r.get("pipe") == pipe),
                key=lambda r: r["total_tokens"],
            )
            reaching = [r for r in cells if r["best_cycles"] <= threshold]
            if reaching:
                # Tokens to reach iso-perf = the cheapest run that hit threshold.
                row_out[f"pipe_{pipe}_tokens_to_iso_perf"] = reaching[0]["total_tokens"]
            else:
                row_out[f"pipe_{pipe}_tokens_to_iso_perf"] = None
        out.append(row_out)
    return out


# --------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------- #


def render_ablation_md(
    rows: list[dict[str, Any]],
    iso_token: list[dict[str, Any]],
    iso_perf: list[dict[str, Any]],
) -> str:
    """Render the markdown table the paper claim cites."""
    parts: list[str] = []
    parts.append("# Four-way bench — constrained-vs-unconstrained\n")
    parts.append(
        "Pipelines: **A** = KernelBlaster vanilla, **B** = Autocomp vanilla, "
        "**C** = KernelBlaster + CompGen, **D** = Autocomp + CompGen.\n"
    )

    parts.append("\n## H1 — Tokens to reach iso-perf (best ± 10%)\n")
    if not iso_perf:
        parts.append("_No successful cells. H1 can't be evaluated._\n")
    else:
        parts.append(
            "| target | best_cycles | A tokens | B tokens | C tokens | D tokens |\n"
            "|--------|------------:|---------:|---------:|---------:|---------:|"
        )
        for r in iso_perf:
            parts.append(
                "| {target} | {best} | {a} | {b} | {c} | {d} |".format(
                    target=r["target"],
                    best=r["best_cycles_any_pipeline"],
                    a=r.get("pipe_A_tokens_to_iso_perf") or "—",
                    b=r.get("pipe_B_tokens_to_iso_perf") or "—",
                    c=r.get("pipe_C_tokens_to_iso_perf") or "—",
                    d=r.get("pipe_D_tokens_to_iso_perf") or "—",
                )
            )

    parts.append("\n\n## H2 — Best cycles at iso-token-budget\n")
    if not iso_token:
        parts.append("_No successful cells._\n")
    else:
        parts.append(
            "| target | budget | A cycles | B cycles | C cycles | D cycles |\n"
            "|--------|--------|---------:|---------:|---------:|---------:|"
        )
        for r in iso_token:
            parts.append(
                "| {target} | {budget} | {a} | {b} | {c} | {d} |".format(
                    target=r["target"],
                    budget=r["budget"],
                    a=r.get("pipe_A_best_cycles") or "—",
                    b=r.get("pipe_B_best_cycles") or "—",
                    c=r.get("pipe_C_best_cycles") or "—",
                    d=r.get("pipe_D_best_cycles") or "—",
                )
            )

    parts.append("\n\n## Coverage summary\n")
    n_total = len(rows)
    n_ok = sum(1 for r in rows if r.get("succeeded"))
    n_budget_guard = sum(
        1 for r in rows
        if (r.get("failure_reason") or "").startswith("budget_guard")
    )
    parts.append(
        f"- Total cells: **{n_total}**\n"
        f"- Succeeded: **{n_ok}** ({n_ok / n_total * 100:.1f}%)\n"
        if n_total else "- No cells in this run.\n"
    )
    if n_budget_guard:
        parts.append(
            f"- Cells short-circuited by the $200 budget guard: "
            f"**{n_budget_guard}** — raise the cap with "
            "``compgen-gemini-usage budget set --cumulative-usd N`` to "
            "complete the sweep.\n"
        )
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------- #
# Pareto plot
# --------------------------------------------------------------------- #


_PIPELINE_COLOR = {"A": "#d62728", "B": "#ff7f0e", "C": "#2ca02c", "D": "#1f77b4"}
_PIPELINE_LABEL = {
    "A": "A: KB vanilla",
    "B": "B: Autocomp vanilla",
    "C": "C: KB + CompGen",
    "D": "D: Autocomp + CompGen",
}


def plot_pareto(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    """Tokens-vs-cycles scatter per target. Returns the paths written.

    Matplotlib is imported lazily so the reporter still works
    (tables-only) when matplotlib isn't installed.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:
        return []

    by_target = defaultdict(list)
    for row in rows:
        if row.get("succeeded") and row.get("best_cycles") is not None:
            by_target[row["target"]].append(row)

    paths: list[Path] = []
    for target, bucket in sorted(by_target.items()):
        fig, ax = plt.subplots(figsize=(7, 5))
        for pipe in ("A", "B", "C", "D"):
            cells = [r for r in bucket if r.get("pipe") == pipe]
            if not cells:
                continue
            xs = [c["total_tokens"] for c in cells]
            ys = [c["best_cycles"] for c in cells]
            ax.scatter(
                xs, ys,
                color=_PIPELINE_COLOR[pipe],
                label=_PIPELINE_LABEL[pipe],
                s=80, alpha=0.8, edgecolor="black", linewidth=0.5,
            )
        ax.set_xlabel("Total Gemini tokens (prompt + completion)")
        ax.set_ylabel("Best mcycle count (lower is better)")
        ax.set_title(f"Four-way Pareto — {target}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", frameon=True)
        out_path = out_dir / f"pareto_{target}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        paths.append(out_path)
    return paths


# --------------------------------------------------------------------- #
# Top-level entry
# --------------------------------------------------------------------- #


def build_report(out_dir: Path) -> dict[str, Any]:
    """Read runs.jsonl in ``out_dir`` and emit every report artifact.

    Returns a small summary dict (also written to ``summary.json``)
    suitable for the trust report's ingest path.
    """
    runs_path = out_dir / "runs.jsonl"
    if not runs_path.exists():
        raise FileNotFoundError(f"no runs.jsonl at {runs_path}")

    rows = load_runs(runs_path)
    buckets = aggregate_by_pipe_target(rows)
    iso_token = iso_token_table(rows)
    iso_perf = iso_perf_table(rows)
    pareto_paths = plot_pareto(rows, out_dir)
    md = render_ablation_md(rows, iso_token, iso_perf)

    (out_dir / "ablation_table.md").write_text(md)

    summary = {
        "total_cells": len(rows),
        "succeeded_cells": sum(1 for r in rows if r.get("succeeded")),
        "iso_token_table": iso_token,
        "iso_perf_table": iso_perf,
        "pareto_plots": [str(p) for p in pareto_paths],
        "buckets": {
            f"{p}_{t}": {
                "mean_tokens": b["mean_tokens"],
                "median_cycles": b["median_cycles"],
                "best_cycles": b["best_cycles"],
                "success_rate": b["success_rate"],
                "n_total": b["total"],
            }
            for (p, t), b in buckets.items()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Pareto plots + ablation tables from a four-way runs.jsonl"
    )
    parser.add_argument("--out", type=Path, required=True,
                        help="The directory the driver wrote to.")
    args = parser.parse_args()

    summary = build_report(args.out)
    print(f"Built {summary['succeeded_cells']}/{summary['total_cells']} cells.")
    print(f"Tables: {args.out / 'ablation_table.md'}")
    print(f"Summary: {args.out / 'summary.json'}")
    if summary["pareto_plots"]:
        print(f"Pareto plots: {summary['pareto_plots']}")
    else:
        print("Pareto plots: skipped (matplotlib not installed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

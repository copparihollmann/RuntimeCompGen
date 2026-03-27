"""Cross-run comparison and summary table generation."""

from __future__ import annotations

from pathlib import Path

from benchmarks.record import RunRecord


def load_all_results(results_dir: str | Path) -> list[RunRecord]:
    """Load all RunRecord JSON files from a directory."""
    results_dir = Path(results_dir)
    records = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            records.append(RunRecord.load(path))
        except Exception:
            continue
    return records


def summary_table(records: list[RunRecord]) -> str:
    """Generate a markdown summary table from RunRecords."""
    if not records:
        return "No records found."

    lines = [
        "| Model | Target | Objective | Compile (ms) | EqSat Δ% | Recipe Ops | Verification | Latency (μs) | Speedup |",
        "|-------|--------|-----------|-------------|---------|-----------|-------------|-------------|---------|",
    ]

    for r in records:
        eqsat_pct = f"{r.eqsat.ops_reduction_pct:+.1f}" if r.eqsat.changed else "—"
        ver = r.verification.overall_status
        latency = f"{r.performance.latency_median_us:.1f}" if r.performance.latency_median_us else "—"
        speedup = f"{r.baselines.speedup_vs_compiled:.2f}x" if r.baselines.speedup_vs_compiled else "—"
        lines.append(
            f"| {r.model_name} | {r.target_name} | {r.objective} | "
            f"{r.total_compile_time_ms:.0f} | {eqsat_pct} | "
            f"{r.recipe.total_recipe_ops} | {ver} | {latency} | {speedup} |"
        )

    return "\n".join(lines)


def ablation_table(records: list[RunRecord]) -> str:
    """Generate ablation comparison table."""
    if not records:
        return "No ablation records found."

    lines = [
        "| Ablation | Model | Compile (ms) | EqSat Ops | Recipe Ops | Verification |",
        "|----------|-------|-------------|----------|-----------|-------------|",
    ]

    for r in records:
        ablation = r.config.get("ablation", "full")
        ver = r.verification.overall_status
        lines.append(
            f"| {ablation} | {r.model_name} | "
            f"{r.total_compile_time_ms:.0f} | {r.eqsat.ops_after} | "
            f"{r.recipe.total_recipe_ops} | {ver} |"
        )

    return "\n".join(lines)


def export_csv(records: list[RunRecord], output_path: str | Path) -> Path:
    """Export key metrics to CSV for external analysis."""
    path = Path(output_path)
    import csv

    fieldnames = [
        "run_id", "model_name", "target_name", "objective",
        "total_compile_time_ms",
        "export_success", "decomposition_coverage",
        "eqsat_ops_before", "eqsat_ops_after", "eqsat_reduction_pct", "eqsat_time_ms",
        "recipe_total_ops", "recipe_candidate_ops", "recipe_verify_ops",
        "solver_placement_time_ms", "solver_schedule_time_ms", "solver_gap",
        "verification_status",
        "latency_median_us", "latency_p99_us",
        "speedup_vs_compiled",
        "llm_total_tokens", "llm_cost_usd",
        "promotion_status",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "run_id": r.run_id,
                "model_name": r.model_name,
                "target_name": r.target_name,
                "objective": r.objective,
                "total_compile_time_ms": r.total_compile_time_ms,
                "export_success": r.capture.export_success,
                "decomposition_coverage": r.capture.decomposition_coverage,
                "eqsat_ops_before": r.eqsat.ops_before,
                "eqsat_ops_after": r.eqsat.ops_after,
                "eqsat_reduction_pct": r.eqsat.ops_reduction_pct,
                "eqsat_time_ms": r.eqsat.eqsat_time_ms,
                "recipe_total_ops": r.recipe.total_recipe_ops,
                "recipe_candidate_ops": r.recipe.candidate_ops,
                "recipe_verify_ops": r.recipe.verify_ops,
                "solver_placement_time_ms": r.solver.placement_time_ms,
                "solver_schedule_time_ms": r.solver.schedule_time_ms,
                "solver_gap": r.solver.placement_gap,
                "verification_status": r.verification.overall_status,
                "latency_median_us": r.performance.latency_median_us,
                "latency_p99_us": r.performance.latency_p99_us,
                "speedup_vs_compiled": r.baselines.speedup_vs_compiled,
                "llm_total_tokens": r.llm.total_tokens,
                "llm_cost_usd": r.llm.total_cost_usd,
                "promotion_status": r.promotion_status,
            })

    return path


__all__ = ["ablation_table", "export_csv", "load_all_results", "summary_table"]

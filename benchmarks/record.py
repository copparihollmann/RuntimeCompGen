"""Benchmark run record — captures every metric from a CompGen compilation."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class CaptureMetrics:
    """Frontend capture metrics."""
    export_success: bool = False
    graph_break_count: int = 0
    op_coverage: float = 0.0
    unsupported_ops: list[str] = field(default_factory=list)
    export_time_ms: float = 0.0
    decomposition_coverage: float = 0.0
    total_fx_nodes: int = 0
    decomposed_ops: int = 0
    opaque_ops: int = 0


@dataclass
class IRMetrics:
    """Payload IR quality metrics."""
    total_ops: int = 0
    region_count: int = 0
    total_flops: int = 0
    total_bytes: int = 0
    compute_ops: int = 0
    memory_ops: int = 0
    op_type_histogram: dict[str, int] = field(default_factory=dict)


@dataclass
class RecipeMetrics:
    """Recipe IR generation and lowering metrics."""
    total_recipe_ops: int = 0
    scope_ops: int = 0
    fact_ops: int = 0
    candidate_ops: int = 0
    choice_ops: int = 0
    verify_ops: int = 0
    provenance_ops: int = 0
    seed_generation_time_ms: float = 0.0
    validation_passed: bool = False
    validation_errors: int = 0
    # Lowering outputs
    transform_scripts_count: int = 0
    kernel_jobs_count: int = 0
    plan_fragments_count: int = 0
    verification_obligations_count: int = 0
    eqsat_jobs_count: int = 0
    lowering_diagnostics: int = 0


@dataclass
class EqSatMetrics:
    """Equality saturation metrics."""
    ops_before: int = 0
    ops_after: int = 0
    ops_reduction_pct: float = 0.0
    eclasses_initial: int = 0
    eclasses_after_rewrite: int = 0
    enodes_after_rewrite: int = 0
    rules_applied: dict[str, int] = field(default_factory=dict)
    total_rule_applications: int = 0
    changed: bool = False
    eqsat_time_ms: float = 0.0


@dataclass
class SolverMetrics:
    """Solver-backed planning metrics."""
    # Placement
    placement_feasible: bool = False
    placement_objective: float = 0.0
    placement_gap: float = 0.0
    placement_time_ms: float = 0.0
    placement_transfer_cost: float = 0.0
    # Scheduling
    schedule_feasible: bool = False
    schedule_makespan_us: float = 0.0
    schedule_time_ms: float = 0.0
    schedule_deadline_met: bool = True
    # Memory
    memory_feasible: bool = False
    memory_peak_bytes: int = 0
    memory_reuse_count: int = 0
    memory_time_ms: float = 0.0


@dataclass
class KernelMetrics:
    """Kernel generation and validation metrics."""
    total_kernel_specs: int = 0
    strategy_histogram: dict[str, int] = field(default_factory=dict)
    # Per-kernel results (list of dicts)
    kernel_results: list[dict[str, Any]] = field(default_factory=list)
    # Aggregates
    kernels_searched: int = 0
    kernels_correct: int = 0
    kernels_pass_rate: float = 0.0
    best_speedup: float = 0.0
    total_search_tokens: int = 0
    total_search_time_ms: float = 0.0


@dataclass
class VerificationMetrics:
    """Verification ladder metrics."""
    structural_pass: bool = False
    structural_violations: int = 0
    check_assertions_pass: bool = False
    check_assertions_run: int = 0
    differential_pass: bool = False
    differential_max_error: float = 0.0
    translation_validation_pass: bool | None = None
    translation_validation_time_ms: float = 0.0
    overall_status: str = "pending"  # "pass", "fail", "skip", "pending"


@dataclass
class PerformanceMetrics:
    """Runtime performance measurements."""
    # Latency
    latency_median_us: float = 0.0
    latency_p99_us: float = 0.0
    latency_mean_us: float = 0.0
    latency_std_us: float = 0.0
    per_run_us: list[float] = field(default_factory=list)
    # Throughput
    throughput_samples_per_sec: float = 0.0
    # Memory
    peak_memory_bytes: int = 0
    # Device
    device: str = ""
    mode: str = ""
    num_iterations: int = 0
    warmup_iterations: int = 0


@dataclass
class BaselineMetrics:
    """Baseline comparison metrics."""
    eager_cpu_latency_us: float = 0.0
    eager_gpu_latency_us: float = 0.0
    compiled_gpu_latency_us: float = 0.0
    compgen_latency_us: float = 0.0
    speedup_vs_eager_cpu: float = 0.0
    speedup_vs_eager_gpu: float = 0.0
    speedup_vs_compiled: float = 0.0


@dataclass
class LLMMetrics:
    """LLM interaction metrics."""
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    calls_per_stage: dict[str, int] = field(default_factory=dict)
    model_id: str = ""


@dataclass
class AgenticMetrics:
    """Agentic compilation loop metrics."""
    iterations_run: int = 0
    iterations_improved: int = 0
    initial_cost_us: float = 0.0
    final_cost_us: float = 0.0
    total_improvement_pct: float = 0.0
    convergence_iteration: int = 0  # iteration where improvement stopped
    # Per-iteration history
    iteration_costs: list[float] = field(default_factory=list)
    iteration_improvements: list[float] = field(default_factory=list)
    iteration_actions: list[str] = field(default_factory=list)


@dataclass
class ProfilingMetrics:
    """Hardware profiling metrics."""
    compute_utilization: float = 0.0
    memory_utilization: float = 0.0
    dma_compute_overlap: float = 0.0
    idle_fraction: float = 0.0
    bottleneck_regions: list[dict[str, Any]] = field(default_factory=list)
    roofline_points: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunRecord:
    """Complete benchmark run record capturing every metric."""
    # Identity
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    model_name: str = ""
    target_name: str = ""
    objective: str = "latency"
    # Config
    config: dict[str, Any] = field(default_factory=dict)
    # All metric categories
    capture: CaptureMetrics = field(default_factory=CaptureMetrics)
    ir: IRMetrics = field(default_factory=IRMetrics)
    recipe: RecipeMetrics = field(default_factory=RecipeMetrics)
    eqsat: EqSatMetrics = field(default_factory=EqSatMetrics)
    solver: SolverMetrics = field(default_factory=SolverMetrics)
    kernels: KernelMetrics = field(default_factory=KernelMetrics)
    verification: VerificationMetrics = field(default_factory=VerificationMetrics)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    baselines: BaselineMetrics = field(default_factory=BaselineMetrics)
    llm: LLMMetrics = field(default_factory=LLMMetrics)
    agentic: AgenticMetrics = field(default_factory=AgenticMetrics)
    profiling: ProfilingMetrics = field(default_factory=ProfilingMetrics)
    # Timing
    total_compile_time_ms: float = 0.0
    # Status
    promotion_status: str = "pending"  # "promoted", "rejected", "pending"
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    def save(self, output_dir: str | Path) -> Path:
        """Save to JSON file in output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.run_id}_{self.model_name}_{self.target_name}.json"
        path = output_dir / filename
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str | Path) -> RunRecord:
        """Load from JSON file."""
        data = json.loads(Path(path).read_text())
        # Reconstruct nested dataclasses
        record = cls()
        for key, val in data.items():
            if key == "capture":
                record.capture = CaptureMetrics(**val)
            elif key == "ir":
                record.ir = IRMetrics(**val)
            elif key == "recipe":
                record.recipe = RecipeMetrics(**val)
            elif key == "eqsat":
                record.eqsat = EqSatMetrics(**val)
            elif key == "solver":
                record.solver = SolverMetrics(**val)
            elif key == "kernels":
                record.kernels = KernelMetrics(**val)
            elif key == "verification":
                record.verification = VerificationMetrics(**val)
            elif key == "performance":
                record.performance = PerformanceMetrics(**val)
            elif key == "baselines":
                record.baselines = BaselineMetrics(**val)
            elif key == "llm":
                record.llm = LLMMetrics(**val)
            elif key == "agentic":
                record.agentic = AgenticMetrics(**val)
            elif key == "profiling":
                record.profiling = ProfilingMetrics(**val)
            else:
                setattr(record, key, val)
        return record


__all__ = [
    "AgenticMetrics",
    "BaselineMetrics",
    "CaptureMetrics",
    "EqSatMetrics",
    "IRMetrics",
    "KernelMetrics",
    "LLMMetrics",
    "PerformanceMetrics",
    "ProfilingMetrics",
    "RecipeMetrics",
    "RunRecord",
    "SolverMetrics",
    "VerificationMetrics",
]

"""Artifact bundling and manifest generation.

Packages all pipeline outputs into a deployable bundle directory matching
the Artifact Contract from CLAUDE.md.

Invariants:
    - manifest.json is the single source of truth for bundle contents.
    - All artifact paths in the manifest are relative to the bundle root.
    - Bundle is self-contained (no external references).
    - Bundle format is versioned.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xdsl.dialects.builtin import ModuleOp
from xdsl.printer import Printer

from compgen.runtime.planner import ExecutionPlan


@dataclass(frozen=True)
class BundleManifest:
    """Bundle manifest -- index of all artifacts.

    Attributes:
        version: Bundle format version.
        target_profile: Name of the target profile.
        model_hash: Hash of the original model IR.
        objective: Optimization objective.
        artifacts: Dict mapping artifact name to relative path.
        creation_timestamp: ISO 8601 timestamp.
    """

    version: str = "1.0"
    target_profile: str = ""
    model_hash: str = ""
    objective: str = "latency"
    artifacts: dict[str, str] = field(default_factory=dict)
    creation_timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "version": self.version,
            "target_profile": self.target_profile,
            "model_hash": self.model_hash,
            "objective": self.objective,
            "artifacts": self.artifacts,
            "creation_timestamp": self.creation_timestamp,
        }


@dataclass
class BundleBuilder:
    """Builds artifact bundles from pipeline outputs.

    Attributes:
        output_dir: Root directory for the bundle.
    """

    output_dir: Path

    def build(
        self,
        module: ModuleOp,
        execution_plan: ExecutionPlan | None = None,
        target_name: str = "",
        objective: str = "latency",
        golden_inputs: Any = None,
        golden_outputs: Any = None,
        kernel_files: dict[str, str] | None = None,
        transform_scripts: list[str] | None = None,
    ) -> BundleManifest:
        """Build a complete artifact bundle.

        Args:
            module: The optimized xDSL module.
            execution_plan: Execution plan (placement, scheduling).
            target_name: Target profile name.
            objective: Optimization objective.
            golden_inputs: Reference input tensors.
            golden_outputs: Reference output tensors.
            kernel_files: Dict of filename → kernel code.
            transform_scripts: List of transform script contents.

        Returns:
            BundleManifest describing the bundle.
        """
        root = Path(self.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, str] = {}

        # 1. Write payload.mlir
        buf = io.StringIO()
        Printer(stream=buf).print_op(module)
        payload_path = root / "payload.mlir"
        payload_path.write_text(buf.getvalue())
        artifacts["payload"] = "payload.mlir"

        # 2. Write execution_plan.yaml
        if execution_plan is not None:
            import yaml
            plan_path = root / "execution_plan.yaml"
            plan_path.write_text(yaml.dump(execution_plan.to_dict(), default_flow_style=False))
            artifacts["execution_plan"] = "execution_plan.yaml"

        # 3. Write golden I/O
        if golden_inputs is not None:
            import torch
            inputs_path = root / "golden_inputs.pt"
            torch.save(golden_inputs, inputs_path)
            artifacts["golden_inputs"] = "golden_inputs.pt"

        if golden_outputs is not None:
            import torch
            outputs_path = root / "golden_outputs.pt"
            torch.save(golden_outputs, outputs_path)
            artifacts["golden_outputs"] = "golden_outputs.pt"

        # 4. Write kernel files
        if kernel_files:
            kernels_dir = root / "generated_kernels"
            kernels_dir.mkdir(exist_ok=True)
            for name, code in kernel_files.items():
                kernel_path = kernels_dir / name
                kernel_path.write_text(code)
            artifacts["generated_kernels"] = "generated_kernels/"

        # 5. Write transform scripts
        if transform_scripts:
            transforms_dir = root / "transforms"
            transforms_dir.mkdir(exist_ok=True)
            for i, script in enumerate(transform_scripts):
                script_path = transforms_dir / f"transform_{i}.py"
                script_path.write_text(script)
            artifacts["transforms"] = "transforms/"

        # 6. Compute model hash
        model_hash = hashlib.sha256(buf.getvalue().encode()).hexdigest()[:16]

        # 7. Write manifest.json
        manifest = BundleManifest(
            version="1.0",
            target_profile=target_name,
            model_hash=model_hash,
            objective=objective,
            artifacts=artifacts,
            creation_timestamp=datetime.now(UTC).isoformat(),
        )
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))

        return manifest


def create_bundle(
    output_dir: str | Path,
    module: ModuleOp,
    execution_plan: ExecutionPlan | None = None,
    **kwargs: Any,
) -> BundleManifest:
    """Convenience function: build a bundle."""
    builder = BundleBuilder(output_dir=Path(output_dir))
    return builder.build(module, execution_plan, **kwargs)


# Alias for backwards compat (promotion/ imports Bundle)
Bundle = BundleManifest

__all__ = ["Bundle", "BundleBuilder", "BundleManifest", "create_bundle"]

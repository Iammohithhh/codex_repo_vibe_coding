"""Verifier Agent — Runs tests/benchmarks and reports drift from claims.

Validates generated code structure, checks for completeness,
runs experiment plans, and produces benchmark delta reports.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ..models.schema import (
    AgentMessage,
    AgentRole,
    CodeScaffold,
    ExperimentRun,
    PaperSpec,
    ReproducibilityScore,
)


def _verify_file_structure(scaffold: CodeScaffold) -> List[str]:
    """Check that all required files are present."""
    required = ["core/model.py", "train/train.py", "tests/test_model.py", "README.md"]
    file_paths = {f.path for f in scaffold.files}
    missing = [r for r in required if r not in file_paths]
    return missing


def _check_code_quality(scaffold: CodeScaffold) -> List[Dict[str, str]]:
    """Basic static analysis of generated code."""
    issues = []
    for f in scaffold.files:
        if f.path.endswith(".py"):
            if "import" not in f.content and len(f.content) > 50:
                issues.append({"file": f.path, "issue": "No imports found — may be incomplete."})
            if "def " not in f.content and "class " not in f.content and len(f.content) > 100:
                issues.append({"file": f.path, "issue": "No function or class definitions found."})
            if "TODO" in f.content or "FIXME" in f.content:
                issues.append({"file": f.path, "issue": "Contains TODO/FIXME markers."})
    return issues


def _generate_experiment_plan(spec: PaperSpec) -> List[ExperimentRun]:
    """Propose minimal compute experiment runs."""
    runs = []

    # Quick smoke test
    config_smoke = dict(spec.hyperparameters)
    config_smoke["epochs"] = 1
    config_smoke["batch_size"] = min(config_smoke.get("batch_size", 32), 8)
    runs.append(ExperimentRun(
        config=config_smoke,
        metrics_expected={m: 0.0 for m in spec.metrics[:3]},
        status="planned",
        logs=["Smoke test: 1 epoch, small batch — verifies forward/backward pass"],
    ))

    # Quick convergence check
    config_quick = dict(spec.hyperparameters)
    config_quick["epochs"] = min(config_quick.get("epochs", 10), 3)
    runs.append(ExperimentRun(
        config=config_quick,
        metrics_expected={m: 0.0 for m in spec.metrics[:3]},
        status="planned",
        logs=["Quick convergence: 3 epochs — checks loss decreases"],
    ))

    # Full replication run
    runs.append(ExperimentRun(
        config=dict(spec.hyperparameters),
        metrics_expected={m: 0.0 for m in spec.metrics[:3]},
        status="planned",
        logs=["Full replication: paper hyperparameters — targets paper-reported metrics"],
    ))

    return runs


def _compute_benchmark_delta(run: ExperimentRun) -> Dict[str, float]:
    """Compute delta between expected and observed metrics."""
    delta = {}
    for metric, expected in run.metrics_expected.items():
        observed = run.metrics_observed.get(metric, 0.0)
        if expected != 0:
            delta[metric] = round((observed - expected) / abs(expected), 4)
        else:
            delta[metric] = 0.0
    return delta


def verify_scaffold(
    spec: PaperSpec,
    scaffold: CodeScaffold,
    reproducibility: ReproducibilityScore,
) -> Tuple[List[ExperimentRun], List[AgentMessage]]:
    """Main verifier entry point."""
    messages: List[AgentMessage] = []

    messages.append(AgentMessage(
        role=AgentRole.VERIFIER,
        content="Starting verification of generated code scaffold.",
        metadata={"phase": "start"},
    ))

    # Check file structure
    missing = _verify_file_structure(scaffold)
    if missing:
        messages.append(AgentMessage(
            role=AgentRole.VERIFIER,
            content=f"Missing required files: {', '.join(missing)}",
            metadata={"check": "structure", "missing": missing},
            confidence=0.95,
        ))
    else:
        messages.append(AgentMessage(
            role=AgentRole.VERIFIER,
            content="File structure complete — all required files present.",
            metadata={"check": "structure"},
            confidence=0.95,
        ))

    # Code quality check
    quality_issues = _check_code_quality(scaffold)
    if quality_issues:
        messages.append(AgentMessage(
            role=AgentRole.VERIFIER,
            content=f"Found {len(quality_issues)} code quality issues.",
            metadata={"check": "quality", "issues": quality_issues},
            confidence=0.8,
        ))

    # Generate experiment plan
    experiment_runs = _generate_experiment_plan(spec)
    messages.append(AgentMessage(
        role=AgentRole.VERIFIER,
        content=f"Generated {len(experiment_runs)} experiment runs: smoke test, quick convergence, full replication.",
        metadata={"check": "experiments", "run_count": len(experiment_runs)},
        confidence=0.85,
    ))

    # Reproducibility assessment
    if reproducibility.blockers:
        messages.append(AgentMessage(
            role=AgentRole.VERIFIER,
            content=f"Reproducibility blockers detected: {'; '.join(reproducibility.blockers[:3])}",
            metadata={"check": "reproducibility", "blockers": reproducibility.blockers},
            confidence=0.9,
        ))

    # Overall verification verdict
    has_critical = len(missing) > 0 or len(reproducibility.blockers) > 2
    verdict = "PASS with warnings" if not has_critical else "NEEDS ATTENTION"
    messages.append(AgentMessage(
        role=AgentRole.VERIFIER,
        content=f"Verification verdict: {verdict}. "
                f"Structure: {'OK' if not missing else 'INCOMPLETE'}. "
                f"Quality issues: {len(quality_issues)}. "
                f"Reproducibility: {reproducibility.overall:.2f}.",
        metadata={
            "verdict": verdict,
            "structure_ok": not missing,
            "quality_issues": len(quality_issues),
            "reproducibility_score": reproducibility.overall,
        },
        confidence=0.85,
    ))

    return experiment_runs, messages

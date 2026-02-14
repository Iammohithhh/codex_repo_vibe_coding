"""Skeptic Agent — Challenges missing details and low-confidence steps.

Reviews the PaperSpec for completeness, flags missing hyperparameters,
questionable claims, insufficient baselines, and reproducibility risks.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ..models.schema import (
    AgentMessage,
    AgentRole,
    PaperSpec,
    ReproducibilityScore,
)


def _check_hyperparams(spec: PaperSpec) -> Tuple[List[str], List[str]]:
    """Check for missing critical hyperparameters."""
    required = ["learning_rate", "batch_size", "epochs", "optimizer"]
    missing = []
    found = []
    for param in required:
        if param in spec.hyperparameters:
            found.append(param)
        else:
            missing.append(param)
    return found, missing


def _check_baselines(spec: PaperSpec) -> List[str]:
    """Flag baseline comparison issues."""
    issues = []
    if len(spec.baselines) < 2:
        issues.append("Insufficient baselines: fewer than 2 comparison methods found.")
    if not any("state-of-the-art" in b.lower() or "sota" in b.lower() for b in spec.baselines):
        issues.append("No explicit state-of-the-art baseline comparison detected.")
    return issues


def _check_evaluation(spec: PaperSpec) -> List[str]:
    """Flag evaluation methodology issues."""
    issues = []
    if len(spec.metrics) < 2:
        issues.append("Single metric evaluation: consider multiple metrics for robustness.")
    if len(spec.datasets) < 2:
        issues.append("Single dataset evaluation: results may not generalize.")
    strong_claims = [c for c in spec.claims if any(w in c.lower() for w in ["state-of-the-art", "outperform", "surpass"])]
    if strong_claims and len(spec.datasets) < 3:
        issues.append(f"Strong claims ({len(strong_claims)}) with limited evaluation scope ({len(spec.datasets)} datasets).")
    return issues


def _check_reproducibility_gaps(spec: PaperSpec) -> List[str]:
    """Identify reproducibility blockers."""
    gaps = []
    if not spec.hyperparameters:
        gaps.append("CRITICAL: No hyperparameters specified — reproduction impossible without guessing.")
    if not spec.training_details:
        gaps.append("Training details missing: cannot determine compute requirements.")
    if not spec.key_equations:
        gaps.append("No equations extracted — core method may be ambiguous.")
    if not spec.architecture_components:
        gaps.append("Architecture components unclear — implementation requires inference.")
    return gaps


def _compute_reproducibility_score(spec: PaperSpec, issues: List[str]) -> ReproducibilityScore:
    """Compute a reproducibility scorecard."""
    found_params, missing_params = _check_hyperparams(spec)

    code_available = 0.3 if spec.hyperparameters else 0.0  # No actual code repo check
    data_available = min(len(spec.datasets) * 0.25, 1.0)
    hyperparams_complete = len(found_params) / max(len(found_params) + len(missing_params), 1)
    compute_specified = 0.5 if spec.training_details else 0.0
    results_variance = 0.5  # Default moderate confidence

    # Penalize for issues
    penalty = min(len(issues) * 0.05, 0.3)
    overall = max(
        0.0,
        (code_available * 0.2 + data_available * 0.2 + hyperparams_complete * 0.3 +
         compute_specified * 0.15 + results_variance * 0.15) - penalty
    )

    blockers = []
    fixes = []

    if missing_params:
        blockers.append(f"Missing hyperparameters: {', '.join(missing_params)}")
        fixes.append(f"Add defaults for: {', '.join(missing_params)} (use common values from similar papers)")

    if len(spec.datasets) < 2:
        blockers.append("Limited dataset coverage")
        fixes.append("Add evaluation on at least 2 standard benchmarks")

    for issue in issues:
        if "CRITICAL" in issue:
            blockers.append(issue)

    return ReproducibilityScore(
        overall=round(overall, 3),
        code_available=round(code_available, 3),
        data_available=round(data_available, 3),
        hyperparams_complete=round(hyperparams_complete, 3),
        compute_specified=round(compute_specified, 3),
        results_variance=round(results_variance, 3),
        blockers=blockers,
        fixes=fixes,
    )


def review_paper(spec: PaperSpec) -> Tuple[ReproducibilityScore, List[AgentMessage]]:
    """Main skeptic entry point. Returns (ReproducibilityScore, agent_messages)."""
    messages: List[AgentMessage] = []

    messages.append(AgentMessage(
        role=AgentRole.SKEPTIC,
        content="Starting critical review of extracted paper specification.",
        metadata={"phase": "start"},
    ))

    all_issues: List[str] = []

    # Check hyperparameters
    found, missing = _check_hyperparams(spec)
    if missing:
        msg = f"Missing hyperparameters: {', '.join(missing)}. Found: {', '.join(found) if found else 'none'}."
        all_issues.append(msg)
        messages.append(AgentMessage(
            role=AgentRole.SKEPTIC,
            content=msg,
            metadata={"check": "hyperparameters", "missing": missing, "found": found},
            confidence=0.9,
        ))

    # Check baselines
    baseline_issues = _check_baselines(spec)
    all_issues.extend(baseline_issues)
    for issue in baseline_issues:
        messages.append(AgentMessage(
            role=AgentRole.SKEPTIC,
            content=issue,
            metadata={"check": "baselines"},
            confidence=0.8,
        ))

    # Check evaluation
    eval_issues = _check_evaluation(spec)
    all_issues.extend(eval_issues)
    for issue in eval_issues:
        messages.append(AgentMessage(
            role=AgentRole.SKEPTIC,
            content=issue,
            metadata={"check": "evaluation"},
            confidence=0.85,
        ))

    # Check reproducibility gaps
    repro_gaps = _check_reproducibility_gaps(spec)
    all_issues.extend(repro_gaps)
    for gap in repro_gaps:
        messages.append(AgentMessage(
            role=AgentRole.SKEPTIC,
            content=gap,
            metadata={"check": "reproducibility"},
            confidence=0.9,
        ))

    # Compute score
    score = _compute_reproducibility_score(spec, all_issues)

    messages.append(AgentMessage(
        role=AgentRole.SKEPTIC,
        content=(
            f"Review complete. Overall reproducibility score: {score.overall:.2f}. "
            f"Found {len(all_issues)} issues, {len(score.blockers)} blockers."
        ),
        metadata={
            "overall_score": score.overall,
            "issues_count": len(all_issues),
            "blockers_count": len(score.blockers),
        },
        confidence=0.85,
    ))

    return score, messages

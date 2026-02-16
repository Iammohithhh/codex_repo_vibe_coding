"""Skeptic Agent — AI-powered critical review of research papers.

Uses Groq LLM for nuanced reproducibility assessment and gap detection.
Falls back to rule-based checks when GROQ_API_KEY is not set.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ..models.schema import (
    AgentMessage,
    AgentRole,
    PaperSpec,
    ReproducibilityScore,
)
from ..llm.groq_client import groq_json, is_groq_available


# ---------------------------------------------------------------------------
# AI-powered review via Groq
# ---------------------------------------------------------------------------

def _ai_review(spec: PaperSpec) -> Tuple[ReproducibilityScore, List[str]] | None:
    """Use Groq LLM to perform deep critical review."""
    if not is_groq_available():
        return None

    spec_summary = (
        f"Problem: {spec.problem}\n"
        f"Method: {spec.method}\n"
        f"Equations: {[eq.raw for eq in spec.key_equations]}\n"
        f"Datasets: {spec.datasets}\n"
        f"Metrics: {spec.metrics}\n"
        f"Hyperparameters: {spec.hyperparameters}\n"
        f"Architecture: {spec.architecture_components}\n"
        f"Claims: {spec.claims}\n"
        f"Limitations: {spec.limitations}\n"
        f"Baselines: {spec.baselines}\n"
        f"Training details: {spec.training_details}\n"
        f"Assumptions: {spec.assumptions}"
    )

    prompt = f"""You are a rigorous ML paper reviewer. Critically analyze this extracted paper specification for reproducibility and scientific quality.

{spec_summary}

Return a JSON object:
{{
  "overall_score": 0.0-1.0,
  "code_available": 0.0-1.0,
  "data_available": 0.0-1.0,
  "hyperparams_complete": 0.0-1.0,
  "compute_specified": 0.0-1.0,
  "results_variance": 0.0-1.0,
  "issues": ["list of specific issues found"],
  "blockers": ["critical blockers preventing reproduction"],
  "fixes": ["specific actionable fixes for each blocker"],
  "strengths": ["what the paper does well"],
  "missing_details": ["details that are missing but needed for reproduction"]
}}

Be specific and actionable. Score harshly but fairly. A perfect paper gets 0.9, not 1.0."""

    result = groq_json([
        {"role": "system", "content": "You are a critical ML paper reviewer focused on reproducibility. Return valid JSON."},
        {"role": "user", "content": prompt},
    ])

    if result is None:
        return None

    try:
        score = ReproducibilityScore(
            overall=min(max(float(result.get("overall_score", 0.5)), 0.0), 1.0),
            code_available=min(max(float(result.get("code_available", 0.0)), 0.0), 1.0),
            data_available=min(max(float(result.get("data_available", 0.0)), 0.0), 1.0),
            hyperparams_complete=min(max(float(result.get("hyperparams_complete", 0.0)), 0.0), 1.0),
            compute_specified=min(max(float(result.get("compute_specified", 0.0)), 0.0), 1.0),
            results_variance=min(max(float(result.get("results_variance", 0.5)), 0.0), 1.0),
            blockers=result.get("blockers", []),
            fixes=result.get("fixes", []),
        )
        issues = result.get("issues", []) + result.get("missing_details", [])
        return score, issues
    except (KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Heuristic fallback (original logic)
# ---------------------------------------------------------------------------

def _check_hyperparams(spec: PaperSpec) -> Tuple[List[str], List[str]]:
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
    issues = []
    if len(spec.baselines) < 2:
        issues.append("Insufficient baselines: fewer than 2 comparison methods found.")
    if not any("state-of-the-art" in b.lower() or "sota" in b.lower() for b in spec.baselines):
        issues.append("No explicit state-of-the-art baseline comparison detected.")
    return issues


def _check_evaluation(spec: PaperSpec) -> List[str]:
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
    found_params, missing_params = _check_hyperparams(spec)
    code_available = 0.3 if spec.hyperparameters else 0.0
    data_available = min(len(spec.datasets) * 0.25, 1.0)
    hyperparams_complete = len(found_params) / max(len(found_params) + len(missing_params), 1)
    compute_specified = 0.5 if spec.training_details else 0.0
    results_variance = 0.5

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


def _heuristic_review(spec: PaperSpec) -> Tuple[ReproducibilityScore, List[str]]:
    """Original rule-based review."""
    all_issues: List[str] = []
    _, missing = _check_hyperparams(spec)
    if missing:
        all_issues.append(f"Missing hyperparameters: {', '.join(missing)}.")
    all_issues.extend(_check_baselines(spec))
    all_issues.extend(_check_evaluation(spec))
    all_issues.extend(_check_reproducibility_gaps(spec))
    score = _compute_reproducibility_score(spec, all_issues)
    return score, all_issues


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def review_paper(spec: PaperSpec) -> Tuple[ReproducibilityScore, List[AgentMessage]]:
    """Main skeptic entry point. Uses AI review with heuristic fallback."""
    messages: List[AgentMessage] = []

    ai_mode = is_groq_available()
    messages.append(AgentMessage(
        role=AgentRole.SKEPTIC,
        content=f"Starting critical review [{'AI-powered via Groq' if ai_mode else 'rule-based mode'}].",
        metadata={"phase": "start", "mode": "ai" if ai_mode else "heuristic"},
    ))

    # Try AI review
    ai_result = None
    if ai_mode:
        ai_result = _ai_review(spec)

    if ai_result is not None:
        score, all_issues = ai_result
        messages.append(AgentMessage(
            role=AgentRole.SKEPTIC,
            content="AI-powered deep review complete — used Groq LLM for nuanced assessment.",
            metadata={"mode": "ai"},
            confidence=0.90,
        ))
    else:
        score, all_issues = _heuristic_review(spec)
        if ai_mode:
            messages.append(AgentMessage(
                role=AgentRole.SKEPTIC,
                content="AI review failed — falling back to rule-based checks.",
                metadata={"mode": "heuristic_fallback"},
                confidence=0.75,
            ))

    for issue in all_issues:
        messages.append(AgentMessage(
            role=AgentRole.SKEPTIC,
            content=issue,
            metadata={"check": "ai_review" if ai_result else "heuristic"},
            confidence=0.85,
        ))

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

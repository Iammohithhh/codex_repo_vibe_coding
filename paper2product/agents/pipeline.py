"""Multi-Agent Pipeline Orchestrator.

Coordinates the five agents in sequence:
  1. Reader Agent → structured extraction
  2. Skeptic Agent → critical review + reproducibility score
  3. Implementer Agent → code scaffold generation
  4. Verifier Agent → validation + experiment planning
  5. Explainer Agent → visuals + intuition + distillation
"""
from __future__ import annotations

from typing import List

from ..models.schema import (
    AgentMessage,
    AgentPipelineResult,
    IngestRequest,
    Project,
    ProjectStatus,
)
from .reader import read_paper
from .skeptic import review_paper
from .implementer import generate_scaffold
from .verifier import verify_scaffold
from .explainer import explain_paper


def run_pipeline(request: IngestRequest, framework: str = "pytorch") -> Project:
    """Execute the full multi-agent pipeline and return a complete Project."""
    all_messages: List[AgentMessage] = []

    # 1. Reader Agent
    paper_spec, reader_msgs = read_paper(request)
    all_messages.extend(reader_msgs)

    # 2. Skeptic Agent
    reproducibility, skeptic_msgs = review_paper(paper_spec)
    all_messages.extend(skeptic_msgs)

    # 3. Implementer Agent
    code_scaffold, impl_msgs = generate_scaffold(paper_spec, framework)
    all_messages.extend(impl_msgs)

    # 4. Verifier Agent
    experiment_runs, verifier_msgs = verify_scaffold(paper_spec, code_scaffold, reproducibility)
    all_messages.extend(verifier_msgs)

    # 5. Explainer Agent
    visual_pack, distillation, explainer_msgs = explain_paper(paper_spec)
    all_messages.extend(explainer_msgs)

    # Compute overall confidence from agent outputs
    confidence = (
        reproducibility.overall * 0.4 +
        code_scaffold.confidence * 0.3 +
        (1.0 - min(len(reproducibility.blockers) * 0.1, 0.5)) * 0.3
    )

    # Assemble project
    project = Project(
        ingest=request,
        status=ProjectStatus.VALIDATED,
        paper_spec=paper_spec,
        code_scaffold=code_scaffold,
        reproducibility=reproducibility,
        visual_pack=visual_pack,
        distillation=distillation,
        experiment_runs=experiment_runs,
        agent_messages=all_messages,
        confidence_score=round(confidence, 3),
    )

    return project

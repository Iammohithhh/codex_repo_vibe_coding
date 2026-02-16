"""Explainer Agent — AI-powered visual and textual intuition layers.

Uses Groq LLM to generate rich, context-aware explanations, failure modes,
counterfactual analyses, and quiz cards. Falls back to template-based
generation when GROQ_API_KEY is not set.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..models.schema import (
    AgentMessage,
    AgentRole,
    GraphEdge,
    GraphNode,
    PaperSpec,
    VisualPack,
)
from ..llm.groq_client import groq_json, is_groq_available


# ---------------------------------------------------------------------------
# AI-powered explanation generation via Groq
# ---------------------------------------------------------------------------

def _ai_build_summaries(spec: PaperSpec) -> Dict[str, str] | None:
    """Use Groq LLM to generate multi-depth summaries."""
    if not is_groq_available():
        return None

    prompt = f"""Generate three summaries of this research paper at different depths.

Problem: {spec.problem}
Method: {spec.method}
Key equations: {[eq.raw + ' — ' + eq.intuition for eq in spec.key_equations]}
Datasets: {spec.datasets}
Metrics: {spec.metrics}
Architecture: {spec.architecture_components}
Claims: {spec.claims}
Limitations: {spec.limitations}

Return JSON:
{{
  "eli5": "Explain like I'm 5 — simple analogy, no jargon, 2-3 sentences",
  "practitioner": "For an ML engineer — key technical details, what to know to implement, 1 paragraph",
  "researcher": "For a researcher — full technical depth with equations, assumptions, limitations, 2-3 paragraphs"
}}"""

    return groq_json([
        {"role": "system", "content": "You are a science communicator who adapts explanations to different audiences. Return valid JSON."},
        {"role": "user", "content": prompt},
    ])


def _ai_build_failure_modes(spec: PaperSpec) -> List[Dict[str, Any]] | None:
    """Use Groq LLM to identify paper-specific failure modes."""
    if not is_groq_available():
        return None

    prompt = f"""Analyze potential failure modes for this ML paper's approach.

Method: {spec.method}
Architecture: {spec.architecture_components}
Datasets: {spec.datasets}
Limitations: {spec.limitations}
Assumptions: {spec.assumptions}

Return JSON:
{{
  "failure_modes": [
    {{
      "scenario": "specific failure scenario",
      "description": "what goes wrong and why",
      "impact": "effect on model performance",
      "mitigation": "how to prevent or detect",
      "severity": "high/medium/low"
    }}
  ]
}}

Include at least 4 failure modes. Be specific to THIS paper, not generic ML failures."""

    result = groq_json([
        {"role": "system", "content": "You are an ML safety and reliability expert. Return valid JSON."},
        {"role": "user", "content": prompt},
    ])

    if result and "failure_modes" in result:
        return result["failure_modes"]
    return None


def _ai_build_counterfactuals(spec: PaperSpec) -> List[Dict[str, str]] | None:
    """Use Groq LLM to generate insightful counterfactual analyses."""
    if not is_groq_available():
        return None

    prompt = f"""Generate counterfactual "what if" analyses for this paper.

Method: {spec.method}
Architecture: {spec.architecture_components}
Key equations: {[eq.raw for eq in spec.key_equations]}
Hyperparameters: {spec.hyperparameters}

Return JSON:
{{
  "counterfactuals": [
    {{
      "question": "What if we changed X?",
      "impact": "Expected effect on performance",
      "alternative": "Concrete alternative to try"
    }}
  ]
}}

Generate 4-5 insightful counterfactuals specific to this paper's design choices."""

    result = groq_json([
        {"role": "system", "content": "You are a thoughtful ML researcher exploring design alternatives. Return valid JSON."},
        {"role": "user", "content": prompt},
    ])

    if result and "counterfactuals" in result:
        return result["counterfactuals"]
    return None


def _ai_build_quiz_cards(spec: PaperSpec) -> List[Dict[str, str]] | None:
    """Use Groq LLM to generate educational quiz cards."""
    if not is_groq_available():
        return None

    prompt = f"""Generate quiz/interview cards about this research paper at different difficulty levels.

Problem: {spec.problem}
Method: {spec.method}
Key equations: {[eq.raw + ' — ' + eq.intuition for eq in spec.key_equations]}
Datasets: {spec.datasets}
Metrics: {spec.metrics}
Architecture: {spec.architecture_components}
Assumptions: {spec.assumptions}
Limitations: {spec.limitations}

Return JSON:
{{
  "quiz_cards": [
    {{"question": "...", "answer": "detailed answer", "difficulty": "beginner/intermediate/advanced/expert"}}
  ]
}}

Generate 8-10 cards spanning all difficulty levels. Include conceptual, technical, and critical thinking questions."""

    result = groq_json([
        {"role": "system", "content": "You are an ML educator creating assessment materials. Return valid JSON."},
        {"role": "user", "content": prompt},
    ])

    if result and "quiz_cards" in result:
        return result["quiz_cards"]
    return None


# ---------------------------------------------------------------------------
# Template-based fallback (original logic)
# ---------------------------------------------------------------------------

def _build_multi_depth_summary(spec: PaperSpec) -> Dict[str, str]:
    return {
        "eli5": (
            f"This paper solves the problem: {spec.problem[:100]}. "
            f"Think of it like teaching a small student model to copy a big teacher model's answers, "
            f"but making the student much faster and smaller."
        ),
        "practitioner": (
            f"Problem: {spec.problem}\n\n"
            f"Method: {spec.method}\n\n"
            f"The approach uses {', '.join(spec.architecture_components[:3]) if spec.architecture_components else 'neural network components'} "
            f"evaluated on {', '.join(spec.datasets[:3]) if spec.datasets else 'standard benchmarks'} "
            f"measuring {', '.join(spec.metrics[:3]) if spec.metrics else 'standard metrics'}.\n\n"
            f"Key hyperparameters: {spec.hyperparameters if spec.hyperparameters else 'not fully specified'}."
        ),
        "researcher": (
            f"Problem Statement: {spec.problem}\n\n"
            f"Methodology: {spec.method}\n\n"
            f"Key Equations:\n"
            + "\n".join(f"  - {eq.raw}: {eq.intuition}" for eq in spec.key_equations)
            + f"\n\nDatasets: {', '.join(spec.datasets)}"
            f"\nMetrics: {', '.join(spec.metrics)}"
            f"\nAssumptions: {'; '.join(spec.assumptions)}"
            f"\nLimitations: {'; '.join(spec.limitations)}"
            f"\nClaims: {'; '.join(spec.claims[:3])}"
        ),
    }


def _build_knowledge_graph(spec: PaperSpec) -> Tuple[List[GraphNode], List[GraphEdge]]:
    nodes = [
        GraphNode(id="problem", label=spec.problem[:60], node_type="problem"),
        GraphNode(id="method", label=spec.method[:60], node_type="method"),
    ]
    edges = [
        GraphEdge(source="problem", target="method", label="solved by"),
    ]

    for i, eq in enumerate(spec.key_equations):
        eq_id = f"eq_{i}"
        nodes.append(GraphNode(id=eq_id, label=eq.raw[:50], node_type="equation"))
        edges.append(GraphEdge(source="method", target=eq_id, label="uses"))

    for i, ds in enumerate(spec.datasets):
        ds_id = f"dataset_{i}"
        nodes.append(GraphNode(id=ds_id, label=ds, node_type="dataset"))
        edges.append(GraphEdge(source="method", target=ds_id, label="evaluated on"))

    for i, m in enumerate(spec.metrics):
        m_id = f"metric_{i}"
        nodes.append(GraphNode(id=m_id, label=m, node_type="metric"))
        edges.append(GraphEdge(source="method", target=m_id, label="measures"))

    for i, comp in enumerate(spec.architecture_components[:5]):
        comp_id = f"arch_{i}"
        nodes.append(GraphNode(id=comp_id, label=comp, node_type="architecture"))
        edges.append(GraphEdge(source="method", target=comp_id, label="component"))

    for i, claim in enumerate(spec.claims[:3]):
        claim_id = f"claim_{i}"
        nodes.append(GraphNode(id=claim_id, label=claim[:60], node_type="claim"))
        edges.append(GraphEdge(source="method", target=claim_id, label="claims"))

    return nodes, edges


def _build_mermaid_diagrams(spec: PaperSpec) -> Dict[str, str]:
    diagrams = {}

    arch_lines = ["flowchart TD"]
    arch_lines.append("    INPUT[Input Data] --> PREPROCESS[Preprocessing]")
    arch_lines.append("    PREPROCESS --> ENCODER[Encoder]")
    prev = "ENCODER"
    for i, comp in enumerate(spec.architecture_components[:5]):
        node_id = f"COMP_{i}"
        arch_lines.append(f"    {prev} --> {node_id}[{comp}]")
        prev = node_id
    arch_lines.append(f"    {prev} --> OUTPUT[Output]")
    arch_lines.append(f"    OUTPUT --> EVAL[Evaluation]")
    for m in spec.metrics[:3]:
        safe_m = m.replace(" ", "_")
        arch_lines.append(f"    EVAL --> M_{safe_m}[{m}]")
    diagrams["architecture"] = "\n".join(arch_lines)

    flow_lines = ["flowchart LR"]
    flow_lines.append("    RAW[Raw Paper] --> EXTRACT[Feature Extraction]")
    flow_lines.append("    EXTRACT --> TRAIN[Training Pipeline]")
    flow_lines.append("    TRAIN --> VALIDATE[Validation]")
    flow_lines.append("    VALIDATE --> DEPLOY[Deployment]")
    for ds in spec.datasets[:3]:
        flow_lines.append(f"    DS_{ds.replace('-', '_').replace(' ', '_')}[{ds}] --> TRAIN")
    diagrams["data_flow"] = "\n".join(flow_lines)

    loss_lines = ["flowchart TD"]
    for eq in spec.key_equations[:3]:
        safe_raw = eq.raw[:30].replace('"', "'").replace("[", "(").replace("]", ")")
        loss_lines.append(f'    EQ["{safe_raw}"]')
        loss_lines.append(f"    EQ --> OPTIM[Optimizer]")
        loss_lines.append(f"    OPTIM --> UPDATE[Parameter Update]")
        break
    diagrams["objective"] = "\n".join(loss_lines)

    return diagrams


def _build_failure_mode_map(spec: PaperSpec) -> List[Dict[str, Any]]:
    modes = [
        {
            "scenario": "Distribution shift",
            "description": "Training data distribution differs from deployment distribution.",
            "impact": "Degraded metrics, unreliable predictions.",
            "mitigation": "Domain adaptation, data augmentation, monitoring.",
            "severity": "high",
        },
        {
            "scenario": "Insufficient training data",
            "description": "Dataset too small for model capacity.",
            "impact": "Overfitting, poor generalization.",
            "mitigation": "Data augmentation, regularization, transfer learning.",
            "severity": "medium",
        },
        {
            "scenario": "Noisy labels",
            "description": "Training labels contain errors.",
            "impact": "Unstable convergence, biased model.",
            "mitigation": "Label smoothing, noise-robust losses, data cleaning.",
            "severity": "medium",
        },
        {
            "scenario": "Compute constraints",
            "description": "Insufficient GPU/TPU resources for full training.",
            "impact": "Undertrained model, missed hyperparameter tuning.",
            "mitigation": "Mixed precision, gradient accumulation, smaller model variants.",
            "severity": "low",
        },
    ]
    for lim in spec.limitations[:2]:
        modes.append({
            "scenario": "Paper limitation",
            "description": lim,
            "impact": "May affect reproducibility or generalization.",
            "mitigation": "Address in future work or experimental design.",
            "severity": "medium",
        })
    return modes


def _build_counterfactual_analysis(spec: PaperSpec) -> List[Dict[str, str]]:
    analyses = []
    for comp in spec.architecture_components[:3]:
        analyses.append({
            "question": f"What if we remove {comp}?",
            "impact": f"Removing {comp} would likely degrade performance on {', '.join(spec.metrics[:2]) if spec.metrics else 'key metrics'}.",
            "alternative": f"Consider lighter variants of {comp} for efficiency trade-offs.",
        })
    for eq in spec.key_equations[:2]:
        analyses.append({
            "question": f"What if we change the loss function ({eq.raw[:30]}...)?",
            "impact": "Different loss functions alter convergence dynamics and final performance.",
            "alternative": "Try variants like focal loss, label smoothing, or contrastive objectives.",
        })
    if not analyses:
        analyses.append({
            "question": "What if we use a simpler architecture?",
            "impact": "May reduce accuracy but improve inference latency.",
            "alternative": "Try knowledge distillation or model pruning.",
        })
    return analyses


def _build_quiz_cards(spec: PaperSpec) -> List[Dict[str, str]]:
    cards = [
        {"question": "What problem does this paper solve?", "answer": spec.problem, "difficulty": "beginner"},
        {"question": "What is the core method proposed?", "answer": spec.method, "difficulty": "beginner"},
        {"question": "What datasets are used for evaluation?",
         "answer": ", ".join(spec.datasets) if spec.datasets else "Not specified", "difficulty": "intermediate"},
        {"question": "What metrics are reported?",
         "answer": ", ".join(spec.metrics) if spec.metrics else "Not specified", "difficulty": "intermediate"},
    ]
    for eq in spec.key_equations[:2]:
        cards.append({"question": f"Explain the equation: {eq.raw}", "answer": eq.intuition, "difficulty": "advanced"})
    cards.append({"question": "What are the key assumptions of this work?",
                  "answer": "; ".join(spec.assumptions), "difficulty": "advanced"})
    cards.append({"question": "What are the limitations and how would you address them?",
                  "answer": "; ".join(spec.limitations), "difficulty": "expert"})
    return cards


def _build_distillation(spec: PaperSpec, summaries: Dict[str, str],
                        failure_modes: List[Dict[str, Any]],
                        counterfactuals: List[Dict[str, str]],
                        quiz_cards: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "summaries": summaries,
        "poster": {
            "problem": spec.problem,
            "method": spec.method,
            "key_equations": [{"raw": eq.raw, "intuition": eq.intuition} for eq in spec.key_equations],
            "architecture_components": spec.architecture_components,
            "datasets": spec.datasets,
            "metrics": spec.metrics,
            "results": f"Track {', '.join(spec.metrics)}" if spec.metrics else "Pending evaluation",
            "limitations": spec.limitations,
            "formats": ["A0", "A1", "social_media", "slide_deck", "web_embed"],
        },
        "knowledge_cards": {
            "concept_cards": [
                {"title": comp, "note": f"Core architectural component used in the method."}
                for comp in spec.architecture_components[:5]
            ] or [
                {"title": "Model Architecture", "note": "The neural network structure proposed."},
                {"title": "Loss Function", "note": "Objective function guiding optimization."},
            ],
            "equation_intuition_panels": [
                {
                    "equation": eq.raw,
                    "intuition": eq.intuition,
                    "variables": eq.variables,
                    "diagram": f"Balance diagram for {eq.raw[:30]}",
                }
                for eq in spec.key_equations
            ],
            "failure_cards": [
                {"scenario": fm.get("scenario", "Unknown"), "impact": fm.get("impact", "")}
                for fm in failure_modes[:4]
            ],
        },
        "executive_takeaways": {
            "one_page_brief": {
                "why_it_matters": spec.problem,
                "what_it_does": spec.method,
                "key_results": f"Evaluated on {', '.join(spec.datasets[:3])} using {', '.join(spec.metrics[:3])}",
                "risks": spec.limitations[:2] if spec.limitations else ["Limited evaluation scope"],
                "recommendation": "Proceed to prototype if metrics align with product requirements.",
            },
            "five_slide_deck": [
                {"slide": 1, "title": "Problem", "content": spec.problem},
                {"slide": 2, "title": "Solution", "content": spec.method},
                {"slide": 3, "title": "Key Results", "content": f"Metrics: {', '.join(spec.metrics)}"},
                {"slide": 4, "title": "Demo / Architecture", "content": f"Components: {', '.join(spec.architecture_components[:3])}"},
                {"slide": 5, "title": "Roadmap", "content": "Phase 1: Replicate → Phase 2: Optimize → Phase 3: Deploy"},
            ],
        },
        "extension_engine": {
            "what_next": [
                f"Extend evaluation to additional datasets beyond {', '.join(spec.datasets[:2])}.",
                "Explore architecture variants for latency-accuracy trade-offs.",
                "Run scaling-law experiments across model sizes.",
                "Domain transfer to adjacent problem areas.",
            ],
            "missing_piece_detector": [
                f"Assumption test: validate '{a}' empirically." for a in spec.assumptions[:2]
            ] + [
                "Ablation study: measure contribution of each component.",
                "Baseline gap: compare with latest published baselines.",
            ],
            "research_roadmap": [
                "Phase 1: Faithful replication of paper results",
                "Phase 2: Ablation and sensitivity analysis",
                "Phase 3: Improvement and optimization",
                "Phase 4: Domain extension and new applications",
                "Phase 5: Publication / open-source release",
            ],
        },
        "learning_path": {
            "beginner": [
                "Understand the problem domain and motivation",
                "Review mathematical prerequisites",
                "Walk through intuition notes and ELI5 summary",
                "Run the baseline code with default config",
                "Try one small experiment (change a hyperparameter)",
            ],
            "intermediate": [
                "Study the key equations and their derivations",
                "Implement a component from scratch",
                "Run full training and compare with paper metrics",
                "Analyze failure modes and edge cases",
            ],
            "advanced": [
                "Conduct ablation studies",
                "Explore architecture modifications",
                "Run scaling experiments",
                "Write up findings for a technical blog or paper",
            ],
        },
        "quiz_cards": quiz_cards,
        "counterfactual_analysis": counterfactuals,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def explain_paper(spec: PaperSpec) -> Tuple[VisualPack, Dict[str, Any], List[AgentMessage]]:
    """Main explainer entry point. Uses AI explanations with template fallback."""
    messages: List[AgentMessage] = []

    ai_mode = is_groq_available()
    messages.append(AgentMessage(
        role=AgentRole.EXPLAINER,
        content=f"Generating explanations [{'AI-powered via Groq' if ai_mode else 'template mode'}].",
        metadata={"phase": "start", "mode": "ai" if ai_mode else "template"},
    ))

    # Build knowledge graph (always structural — no LLM needed)
    nodes, edges = _build_knowledge_graph(spec)
    mermaid_diagrams = _build_mermaid_diagrams(spec)

    # AI-powered or template-based content
    ai_summaries = _ai_build_summaries(spec) if ai_mode else None
    ai_failure_modes = _ai_build_failure_modes(spec) if ai_mode else None
    ai_counterfactuals = _ai_build_counterfactuals(spec) if ai_mode else None
    ai_quiz_cards = _ai_build_quiz_cards(spec) if ai_mode else None

    summaries = ai_summaries if ai_summaries else _build_multi_depth_summary(spec)
    failure_modes = ai_failure_modes if ai_failure_modes else _build_failure_mode_map(spec)
    counterfactuals = ai_counterfactuals if ai_counterfactuals else _build_counterfactual_analysis(spec)
    quiz_cards = ai_quiz_cards if ai_quiz_cards else _build_quiz_cards(spec)

    ai_components_used = sum(1 for x in [ai_summaries, ai_failure_modes, ai_counterfactuals, ai_quiz_cards] if x)
    if ai_components_used > 0:
        messages.append(AgentMessage(
            role=AgentRole.EXPLAINER,
            content=f"AI generated {ai_components_used}/4 explanation components via Groq LLM.",
            metadata={"mode": "ai", "ai_components": ai_components_used},
            confidence=0.90,
        ))

    dependency_view = {
        "components": spec.architecture_components,
        "dependencies": [
            {"from": comp, "to": "training pipeline", "type": "required"}
            for comp in spec.architecture_components[:3]
        ],
        "counterfactuals": counterfactuals,
    }

    data_flow = [
        {"step": 1, "name": "Data Loading", "description": f"Load {', '.join(spec.datasets[:2]) if spec.datasets else 'dataset'}"},
        {"step": 2, "name": "Preprocessing", "description": "Transform and augment data"},
        {"step": 3, "name": "Forward Pass", "description": f"Pass through {', '.join(spec.architecture_components[:2]) if spec.architecture_components else 'model'}"},
        {"step": 4, "name": "Loss Computation", "description": f"Compute {spec.key_equations[0].raw[:40] if spec.key_equations else 'loss'}"},
        {"step": 5, "name": "Backpropagation", "description": "Update parameters"},
        {"step": 6, "name": "Evaluation", "description": f"Measure {', '.join(spec.metrics[:3]) if spec.metrics else 'metrics'}"},
    ]

    visual_pack = VisualPack(
        architecture_graph={"nodes": [{"id": n.id, "label": n.label, "type": n.node_type} for n in nodes],
                            "edges": [{"source": e.source, "target": e.target, "label": e.label} for e in edges]},
        data_flow_timeline=data_flow,
        failure_mode_map=failure_modes,
        dependency_view=dependency_view,
        mermaid_diagrams=mermaid_diagrams,
        nodes=nodes,
        edges=edges,
    )

    distillation = _build_distillation(spec, summaries, failure_modes, counterfactuals, quiz_cards)

    messages.append(AgentMessage(
        role=AgentRole.EXPLAINER,
        content=f"Generated {len(nodes)} graph nodes, {len(mermaid_diagrams)} diagrams, "
                f"{len(failure_modes)} failure modes, {len(counterfactuals)} counterfactual analyses.",
        metadata={
            "nodes": len(nodes),
            "edges": len(edges),
            "diagrams": len(mermaid_diagrams),
            "failure_modes": len(failure_modes),
        },
        confidence=0.90 if ai_components_used > 0 else 0.85,
    ))

    return visual_pack, distillation, messages

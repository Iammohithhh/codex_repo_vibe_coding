"""Explainer Agent — Creates visual and textual intuition layers.

Generates:
- Multi-depth summaries (ELI5 -> practitioner -> researcher)
- Equation-to-intuition translations
- Knowledge graph with nodes and edges
- Visual packs (architecture, data flow, failure modes, dependencies)
- Interactive concept cards
- Counterfactual analysis
- Quiz/interview Q&A cards
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


def _build_multi_depth_summary(spec: PaperSpec) -> Dict[str, str]:
    """Generate summaries at multiple depth levels."""
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
    """Build a paper knowledge graph with typed nodes and edges."""
    nodes = [
        GraphNode(id="problem", label=spec.problem[:60], node_type="problem"),
        GraphNode(id="method", label=spec.method[:60], node_type="method"),
    ]
    edges = [
        GraphEdge(source="problem", target="method", label="solved by"),
    ]

    # Add equation nodes
    for i, eq in enumerate(spec.key_equations):
        eq_id = f"eq_{i}"
        nodes.append(GraphNode(id=eq_id, label=eq.raw[:50], node_type="equation"))
        edges.append(GraphEdge(source="method", target=eq_id, label="uses"))

    # Add dataset nodes
    for i, ds in enumerate(spec.datasets):
        ds_id = f"dataset_{i}"
        nodes.append(GraphNode(id=ds_id, label=ds, node_type="dataset"))
        edges.append(GraphEdge(source="method", target=ds_id, label="evaluated on"))

    # Add metric nodes
    for i, m in enumerate(spec.metrics):
        m_id = f"metric_{i}"
        nodes.append(GraphNode(id=m_id, label=m, node_type="metric"))
        edges.append(GraphEdge(source="method", target=m_id, label="measures"))

    # Add architecture component nodes
    for i, comp in enumerate(spec.architecture_components[:5]):
        comp_id = f"arch_{i}"
        nodes.append(GraphNode(id=comp_id, label=comp, node_type="architecture"))
        edges.append(GraphEdge(source="method", target=comp_id, label="component"))

    # Add claim nodes
    for i, claim in enumerate(spec.claims[:3]):
        claim_id = f"claim_{i}"
        nodes.append(GraphNode(id=claim_id, label=claim[:60], node_type="claim"))
        edges.append(GraphEdge(source="method", target=claim_id, label="claims"))

    return nodes, edges


def _build_mermaid_diagrams(spec: PaperSpec) -> Dict[str, str]:
    """Generate multiple Mermaid diagrams for different views."""
    diagrams = {}

    # Architecture diagram
    arch_lines = ["flowchart TD"]
    arch_lines.append("    INPUT[Input Data] --> PREPROCESS[Preprocessing]")
    arch_lines.append("    PREPROCESS --> ENCODER[Encoder]")
    prev = "ENCODER"
    for i, comp in enumerate(spec.architecture_components[:5]):
        safe_comp = comp.replace(" ", "_").replace("-", "_")
        node_id = f"COMP_{i}"
        arch_lines.append(f"    {prev} --> {node_id}[{comp}]")
        prev = node_id
    arch_lines.append(f"    {prev} --> OUTPUT[Output]")
    arch_lines.append(f"    OUTPUT --> EVAL[Evaluation]")
    for m in spec.metrics[:3]:
        safe_m = m.replace(" ", "_")
        arch_lines.append(f"    EVAL --> M_{safe_m}[{m}]")
    diagrams["architecture"] = "\n".join(arch_lines)

    # Data flow diagram
    flow_lines = ["flowchart LR"]
    flow_lines.append("    RAW[Raw Paper] --> EXTRACT[Feature Extraction]")
    flow_lines.append("    EXTRACT --> TRAIN[Training Pipeline]")
    flow_lines.append("    TRAIN --> VALIDATE[Validation]")
    flow_lines.append("    VALIDATE --> DEPLOY[Deployment]")
    for ds in spec.datasets[:3]:
        flow_lines.append(f"    DS_{ds.replace('-', '_').replace(' ', '_')}[{ds}] --> TRAIN")
    diagrams["data_flow"] = "\n".join(flow_lines)

    # Loss/objective diagram
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
    """Generate failure mode analysis."""
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
    # Add paper-specific failure modes from limitations
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
    """Generate 'what if we change X?' analysis."""
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
    """Generate quiz and interview-style Q&A cards."""
    cards = [
        {
            "question": f"What problem does this paper solve?",
            "answer": spec.problem,
            "difficulty": "beginner",
        },
        {
            "question": f"What is the core method proposed?",
            "answer": spec.method,
            "difficulty": "beginner",
        },
        {
            "question": f"What datasets are used for evaluation?",
            "answer": ", ".join(spec.datasets) if spec.datasets else "Not specified",
            "difficulty": "intermediate",
        },
        {
            "question": f"What metrics are reported?",
            "answer": ", ".join(spec.metrics) if spec.metrics else "Not specified",
            "difficulty": "intermediate",
        },
    ]
    for eq in spec.key_equations[:2]:
        cards.append({
            "question": f"Explain the equation: {eq.raw}",
            "answer": eq.intuition,
            "difficulty": "advanced",
        })
    cards.append({
        "question": "What are the key assumptions of this work?",
        "answer": "; ".join(spec.assumptions),
        "difficulty": "advanced",
    })
    cards.append({
        "question": "What are the limitations and how would you address them?",
        "answer": "; ".join(spec.limitations),
        "difficulty": "expert",
    })
    return cards


def _build_distillation(spec: PaperSpec) -> Dict[str, Any]:
    """Build the full research distillation package."""
    return {
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
                {"scenario": "Low data regime", "impact": "Overfitting risk"},
                {"scenario": "Domain shift", "impact": "Metric degradation"},
                {"scenario": "Noisy labels", "impact": "Unstable convergence"},
                {"scenario": "Resource constraints", "impact": "Undertrained model"},
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
        "quiz_cards": _build_quiz_cards(spec),
    }


def explain_paper(spec: PaperSpec) -> Tuple[VisualPack, Dict[str, Any], List[AgentMessage]]:
    """Main explainer entry point. Returns (VisualPack, distillation, messages)."""
    messages: List[AgentMessage] = []

    messages.append(AgentMessage(
        role=AgentRole.EXPLAINER,
        content="Generating visual and textual explanations.",
        metadata={"phase": "start"},
    ))

    # Build knowledge graph
    nodes, edges = _build_knowledge_graph(spec)

    # Build mermaid diagrams
    mermaid_diagrams = _build_mermaid_diagrams(spec)

    # Build failure mode map
    failure_modes = _build_failure_mode_map(spec)

    # Build counterfactual analysis
    counterfactuals = _build_counterfactual_analysis(spec)

    # Build multi-depth summaries
    summaries = _build_multi_depth_summary(spec)

    # Build dependency view
    dependency_view = {
        "components": spec.architecture_components,
        "dependencies": [
            {"from": comp, "to": "training pipeline", "type": "required"}
            for comp in spec.architecture_components[:3]
        ],
        "counterfactuals": counterfactuals,
    }

    # Data flow timeline
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

    # Build distillation
    distillation = _build_distillation(spec)
    distillation["summaries"] = summaries

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
        confidence=0.85,
    ))

    return visual_pack, distillation, messages

"""Paper2Product 2.0 — Core module (backwards-compatible wrapper).

This module wraps the v2 agent pipeline while maintaining the original
Phase 1 API surface for backwards compatibility.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import asdict, dataclass, field
from typing import Dict, List
from uuid import uuid4

from .agents.pipeline import run_pipeline
from .models.schema import (
    IngestRequest as V2IngestRequest,
    Project as V2Project,
    to_dict,
)
from .services import persistence as db


# ---------------------------------------------------------------------------
# Legacy dataclasses (kept for backwards compatibility)
# ---------------------------------------------------------------------------
@dataclass
class IngestRequest:
    title: str
    abstract: str
    method_text: str
    source_url: str | None = None


@dataclass
class Citation:
    section: str
    snippet: str


@dataclass
class PaperSpec:
    problem: str
    method: str
    key_equations: List[str]
    datasets: List[str]
    metrics: List[str]
    assumptions: List[str]
    citations: Dict[str, Citation]


@dataclass
class Project:
    id: str
    ingest: IngestRequest
    paper_spec: PaperSpec
    architecture_mermaid: str
    confidence_score: float = 0.78


@dataclass
class Store:
    projects: Dict[str, Project] = field(default_factory=dict)


STORE = Store()


# ---------------------------------------------------------------------------
# V2 pipeline integration
# ---------------------------------------------------------------------------
def _run_v2_pipeline(payload: IngestRequest) -> V2Project:
    """Run the full v2 multi-agent pipeline."""
    v2_request = V2IngestRequest(
        title=payload.title,
        abstract=payload.abstract,
        method_text=payload.method_text,
        source_url=payload.source_url,
    )
    return run_pipeline(v2_request)


def _v2_to_legacy_spec(v2_project: V2Project) -> PaperSpec:
    """Convert v2 PaperSpec to legacy format."""
    spec = v2_project.paper_spec
    equations = [eq.raw for eq in spec.key_equations] if spec and spec.key_equations else ["L = CrossEntropy(y_pred, y_true)"]
    citations = {}
    if spec and spec.citations:
        for key, cite in spec.citations.items():
            citations[key] = Citation(section=cite.section, snippet=cite.snippet)
    return PaperSpec(
        problem=spec.problem if spec else "",
        method=spec.method if spec else "",
        key_equations=equations,
        datasets=spec.datasets if spec else [],
        metrics=spec.metrics if spec else [],
        assumptions=spec.assumptions if spec else [],
        citations=citations,
    )


# ---------------------------------------------------------------------------
# Public API (backwards compatible)
# ---------------------------------------------------------------------------
def _extract_equations(text: str) -> List[str]:
    equations = re.findall(r"([A-Za-z]\s*=\s*[^\n\.;]{3,})", text)
    return equations[:3] or ["L = CrossEntropy(y_pred, y_true)"]


def _extract_keywords(text: str, defaults: List[str]) -> List[str]:
    found = [word for word in defaults if word.lower() in text.lower()]
    return found or defaults[:2]


def build_paper_spec(payload: IngestRequest) -> PaperSpec:
    datasets = _extract_keywords(payload.abstract + payload.method_text, ["ImageNet", "CIFAR-10", "SQuAD", "MIMIC"])
    metrics = _extract_keywords(payload.abstract + payload.method_text, ["accuracy", "F1", "BLEU", "ROUGE"])
    citations = {
        "problem": Citation("abstract", payload.abstract[:180]),
        "method": Citation("method", payload.method_text[:220]),
    }
    return PaperSpec(
        problem=payload.abstract.split(".")[0],
        method=payload.method_text.split(".")[0],
        key_equations=_extract_equations(payload.method_text),
        datasets=datasets,
        metrics=metrics,
        assumptions=[
            "Training distribution reflects real usage.",
            "Compute budget supports replication runs.",
        ],
        citations=citations,
    )


def build_architecture_mermaid(spec: PaperSpec) -> str:
    return "\n".join([
        "flowchart TD",
        "A[Paper]-->B[Data Processing]",
        "B-->C[Core Model]",
        "C-->D[Training]",
        f"D-->E[Metric: {spec.metrics[0]}]",
    ])


def create_project(payload: IngestRequest) -> Project:
    """Create a project using the v2 pipeline, return legacy-compatible Project."""
    # Run v2 pipeline
    v2_project = _run_v2_pipeline(payload)
    # Persist in DB
    db.init_db()
    db.save_project(v2_project)

    # Create legacy-compatible project
    legacy_spec = _v2_to_legacy_spec(v2_project)
    mermaid = v2_project.code_scaffold.architecture_mermaid if v2_project.code_scaffold else build_architecture_mermaid(legacy_spec)
    project = Project(
        id=v2_project.id,
        ingest=payload,
        paper_spec=legacy_spec,
        architecture_mermaid=mermaid,
        confidence_score=v2_project.confidence_score,
    )
    STORE.projects[project.id] = project
    return project


def project_to_dict(project: Project) -> Dict:
    return asdict(project)


def generate_code_scaffold(project: Project, framework: str = "pytorch") -> Dict[str, str]:
    """Generate code scaffold — uses v2 implementer if available, falls back to basic."""
    # Try to get from v2 DB
    v2_data = db.get_project(project.id)
    if v2_data and v2_data.get("code_scaffold"):
        scaffold = v2_data["code_scaffold"]
        files = {}
        for f in scaffold.get("files", []):
            files[f.get("path", "")] = f.get("content", "")
        if files:
            return files

    # Fallback to basic generation
    return {
        "README.md": (
            f"# {project.ingest.title}\n\n"
            f"Framework: {framework}\nConfidence: {project.confidence_score}\n\n"
            "## Citation Traceability\n"
            f"- Problem: {project.paper_spec.citations['problem'].snippet}\n"
            f"- Method: {project.paper_spec.citations['method'].snippet}\n"
        ),
        "core/model.py": (
            '"""Citation-aware generated model.\n'
            f"Problem source: {project.paper_spec.citations['problem'].section}\n"
            f"Method source: {project.paper_spec.citations['method'].section}\n"
            '"""\n\n'
            "def train_step(batch):\n"
            f"    return {{'loss': 0.0, 'framework': '{framework}'}}\n"
        ),
        "train/train.py": "from core.model import train_step\nprint(train_step(None))\n",
        "tests/test_smoke.py": "def test_smoke():\n    assert 1 + 1 == 2\n",
    }


def export_zip(project: Project, framework: str = "pytorch") -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, content in generate_code_scaffold(project, framework).items():
            zf.writestr(path, content)
        zf.writestr("visuals/architecture.mmd", project.architecture_mermaid)
        zf.writestr("reports/paper_spec.json", json.dumps(asdict(project.paper_spec), indent=2))
    return mem.getvalue()


def build_distillation(project: Project) -> Dict:
    """Build distillation — uses v2 explainer if available."""
    v2_data = db.get_project(project.id)
    if v2_data and v2_data.get("distillation"):
        return v2_data["distillation"]

    # Fallback
    return {
        "poster": {
            "problem": project.paper_spec.problem,
            "method": project.paper_spec.method,
            "key_equations": project.paper_spec.key_equations,
            "architecture": "Mermaid flow graph",
            "results": f"Track {', '.join(project.paper_spec.metrics)}",
            "limitations": "Needs larger evaluation and latency profiling.",
            "qr_demo": f"https://demo.local/projects/{project.id}",
            "formats": ["A0", "A1", "social", "slide"],
        },
        "knowledge_cards": {
            "concept_cards": [
                {"title": "Attention", "note": "Highlights important signal paths."},
                {"title": "Loss Function", "note": "Balances fit and generalization."},
            ],
            "equation_intuition_panels": [
                {
                    "equation": project.paper_spec.key_equations[0],
                    "diagram": "Objective balance diagram",
                    "intuition": "Trade-off between teacher alignment and label fit.",
                    "example": "Student model retains 95% accuracy at lower latency.",
                }
            ],
            "failure_cards": [
                {"scenario": "Low data", "impact": "Overfitting"},
                {"scenario": "Domain shift", "impact": "Metric drift"},
                {"scenario": "Noisy labels", "impact": "Unstable convergence"},
            ],
        },
        "executive_takeaways": {
            "one_page_brief": ["Why it matters", "Scale potential", "Risks", "ROI", "Moat"],
            "five_slide_deck": ["Problem", "Solution", "Market", "Demo", "Roadmap"],
        },
        "extension_engine": {
            "what_next": [
                "Dataset extension for medical imaging domain adaptation.",
                "Architecture variants for low-latency inference.",
                "Scaling-law sweep for parameter-performance frontier.",
                "Domain transfer to edge speech tasks.",
            ],
            "missing_piece_detector": [
                "Assumption test: robustness under noisy labels.",
                "Baseline gap: compare with stronger modern baseline.",
                "Eval-size risk: extend benchmark to larger diverse sets.",
            ],
            "research_roadmap": [
                "Phase 1: Replication",
                "Phase 2: Improvement",
                "Phase 3: New domain",
                "Phase 4: Publication",
            ],
        },
        "learning_path": {
            "beginner": ["Math prerequisites", "Intuition notes", "Run baseline code", "Try one experiment"],
            "advanced": ["Edge-case analysis", "Extension experiments", "Optimization and scaling"],
        },
    }


def openapi_spec() -> Dict:
    return {
        "openapi": "3.0.0",
        "info": {"title": "Paper2Product Phase 1", "version": "1.0.0"},
        "paths": {
            "/api/v1/projects/ingest": {"post": {"summary": "Create project from paper text"}},
            "/api/v1/projects/{id}/artifacts": {"post": {"summary": "Generate artifacts"}},
            "/api/v1/projects/{id}/visual-graph": {"get": {"summary": "Get visual graph"}},
            "/api/v1/projects/{id}/distillation": {"get": {"summary": "Research distillation output"}},
            "/api/v1/projects/{id}/export.zip": {"get": {"summary": "Download generated bundle"}},
        },
    }

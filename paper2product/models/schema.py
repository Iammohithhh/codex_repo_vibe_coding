"""Paper2Product 2.0 — Unified data models.

Every generated block references source passage/equation/figure via Citation.
Models support the full pipeline: ingest -> understand -> implement -> evaluate -> release.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ProjectStatus(str, Enum):
    DRAFT = "draft"
    INGESTED = "ingested"
    ARTIFACTS_GENERATED = "artifacts_generated"
    VALIDATED = "validated"
    READY_FOR_REVIEW = "ready_for_review"
    APPROVED = "approved"
    DEPLOYED = "deployed"


class ArtifactType(str, Enum):
    CODE = "code"
    VISUAL = "visual"
    REPORT = "report"
    API_SPEC = "api_spec"
    SDK = "sdk"
    MOBILE_APP = "mobile_app"
    BENCHMARK = "benchmark"
    TEST_SUITE = "test_suite"


class AgentRole(str, Enum):
    READER = "reader"
    SKEPTIC = "skeptic"
    IMPLEMENTER = "implementer"
    VERIFIER = "verifier"
    EXPLAINER = "explainer"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"


# ---------------------------------------------------------------------------
# Citation & Traceability
# ---------------------------------------------------------------------------
@dataclass
class Citation:
    """Sentence-level traceability link from generated artifact back to paper."""
    section: str
    snippet: str
    page: Optional[int] = None
    equation_ref: Optional[str] = None
    figure_ref: Optional[str] = None
    confidence: float = 1.0


@dataclass
class TraceLink:
    """Maps a generated artifact block to its source citation(s)."""
    artifact_path: str
    line_range: tuple = (0, 0)
    citations: List[Citation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Paper Spec (structured extraction from paper)
# ---------------------------------------------------------------------------
@dataclass
class Equation:
    raw: str
    intuition: str = ""
    variables: Dict[str, str] = field(default_factory=dict)
    source_section: str = ""


@dataclass
class PaperSpec:
    problem: str
    method: str
    key_equations: List[Equation] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    architecture_components: List[str] = field(default_factory=list)
    training_details: Dict[str, Any] = field(default_factory=dict)
    citations: Dict[str, Citation] = field(default_factory=dict)
    claims: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    baselines: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
@dataclass
class ReproducibilityScore:
    overall: float = 0.0
    code_available: float = 0.0
    data_available: float = 0.0
    hyperparams_complete: float = 0.0
    compute_specified: float = 0.0
    results_variance: float = 0.0
    blockers: List[str] = field(default_factory=list)
    fixes: List[str] = field(default_factory=list)


@dataclass
class ExperimentRun:
    id: str = field(default_factory=lambda: str(uuid4()))
    config: Dict[str, Any] = field(default_factory=dict)
    metrics_expected: Dict[str, float] = field(default_factory=dict)
    metrics_observed: Dict[str, float] = field(default_factory=dict)
    status: str = "pending"
    logs: List[str] = field(default_factory=list)
    delta_report: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
@dataclass
class GeneratedFile:
    path: str
    content: str
    artifact_type: ArtifactType = ArtifactType.CODE
    trace_links: List[TraceLink] = field(default_factory=list)
    language: str = "python"


@dataclass
class CodeScaffold:
    framework: str = "pytorch"
    files: List[GeneratedFile] = field(default_factory=list)
    architecture_mermaid: str = ""
    ci_workflow: str = ""
    docker_file: str = ""
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Agent Messages
# ---------------------------------------------------------------------------
@dataclass
class AgentMessage:
    role: AgentRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentPipelineResult:
    messages: List[AgentMessage] = field(default_factory=list)
    paper_spec: Optional[PaperSpec] = None
    code_scaffold: Optional[CodeScaffold] = None
    reproducibility: Optional[ReproducibilityScore] = None
    visual_pack: Optional[Dict[str, Any]] = None
    distillation: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Visual Pack
# ---------------------------------------------------------------------------
@dataclass
class GraphNode:
    id: str
    label: str
    node_type: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    label: str = ""
    edge_type: str = "default"


@dataclass
class VisualPack:
    architecture_graph: Dict[str, Any] = field(default_factory=dict)
    data_flow_timeline: List[Dict[str, Any]] = field(default_factory=list)
    failure_mode_map: List[Dict[str, Any]] = field(default_factory=list)
    dependency_view: Dict[str, Any] = field(default_factory=dict)
    mermaid_diagrams: Dict[str, str] = field(default_factory=dict)
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Productization
# ---------------------------------------------------------------------------
@dataclass
class APIContract:
    openapi_spec: Dict[str, Any] = field(default_factory=dict)
    endpoints: List[Dict[str, Any]] = field(default_factory=list)
    sdk_stubs: Dict[str, str] = field(default_factory=dict)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    auth_config: Dict[str, str] = field(default_factory=dict)


@dataclass
class DeploymentPlan:
    architecture_type: str = "server"  # edge, mobile, server, hybrid
    estimated_cost_monthly: float = 0.0
    cloud_provider: str = "aws"
    container_config: Dict[str, Any] = field(default_factory=dict)
    scaling_config: Dict[str, Any] = field(default_factory=dict)
    security_checklist: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LaunchPackage:
    web_deploy: Dict[str, Any] = field(default_factory=dict)
    mobile_release: Dict[str, Any] = field(default_factory=dict)
    staging_config: Dict[str, Any] = field(default_factory=dict)
    release_checklist: List[Dict[str, Any]] = field(default_factory=list)
    telemetry_config: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Collaboration
# ---------------------------------------------------------------------------
@dataclass
class ReviewComment:
    id: str = field(default_factory=lambda: str(uuid4()))
    author: str = "anonymous"
    content: str = ""
    artifact_path: Optional[str] = None
    line_number: Optional[int] = None
    status: ReviewStatus = ReviewStatus.PENDING
    created_at: float = field(default_factory=time.time)


@dataclass
class Workspace:
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    members: List[str] = field(default_factory=list)
    project_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Ingest Request
# ---------------------------------------------------------------------------
@dataclass
class IngestRequest:
    title: str
    abstract: str
    method_text: str
    source_url: Optional[str] = None
    arxiv_id: Optional[str] = None
    github_url: Optional[str] = None
    persona: str = "ml_engineer"  # founder_pm, ml_engineer, designer_educator, mobile_dev


# ---------------------------------------------------------------------------
# Project (top-level aggregate)
# ---------------------------------------------------------------------------
@dataclass
class Project:
    id: str = field(default_factory=lambda: str(uuid4()))
    ingest: Optional[IngestRequest] = None
    status: ProjectStatus = ProjectStatus.DRAFT
    paper_spec: Optional[PaperSpec] = None
    code_scaffold: Optional[CodeScaffold] = None
    reproducibility: Optional[ReproducibilityScore] = None
    visual_pack: Optional[VisualPack] = None
    api_contract: Optional[APIContract] = None
    deployment_plan: Optional[DeploymentPlan] = None
    launch_package: Optional[LaunchPackage] = None
    distillation: Optional[Dict[str, Any]] = None
    experiment_runs: List[ExperimentRun] = field(default_factory=list)
    agent_messages: List[AgentMessage] = field(default_factory=list)
    reviews: List[ReviewComment] = field(default_factory=list)
    workspace_id: Optional[str] = None
    version: int = 1
    confidence_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


def to_dict(obj) -> Any:
    """Recursively convert dataclass to dict, handling enums."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, list):
        return [to_dict(i) for i in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj

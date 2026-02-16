"""Paper2Product 2.0 — Full-featured HTTP server.

Exposes all services through a comprehensive REST API:
- Project lifecycle (ingest, artifacts, visual-graph, distillation, export)
- Multi-agent pipeline execution
- Reproducibility scoring
- Productization (API contracts, deployment plans, launch packages)
- Experiment lab (create, run, benchmark dashboard)
- Collaboration (workspaces, reviews, approvals)
- SDK generation and OpenAPI
"""
from __future__ import annotations

import io
import json
import os
import time
import zipfile
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .agents.pipeline import run_pipeline
from .models.schema import (
    IngestRequest,
    Project,
    ProjectStatus,
    to_dict,
)
from .services import persistence as db
from .services.collaboration import (
    add_review,
    approve_project,
    create_workspace,
    get_reviews,
    list_workspace_projects,
)
from .services.experiment_lab import (
    complete_experiment,
    create_experiment,
    get_benchmark_dashboard,
    list_experiments,
    plan_minimal_experiments,
    start_experiment,
)
from .services.productization import (
    generate_api_contract,
    generate_deployment_plan,
    generate_launch_package,
)
from .services.auth import (
    authenticate,
    generate_api_key,
    login,
)


def _load_frontend():
    """Load the frontend HTML file."""
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_path):
        with open(frontend_path, "r") as f:
            return f.read()
    return FALLBACK_HTML


FALLBACK_HTML = """<!doctype html>
<html><body><h1>Paper2Product 2.0</h1><p>Frontend not found. API is running.</p></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default request logging

    def _send_json(self, payload, status: int = 200):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, content_type: str, filename: str = ""):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        if filename:
            self.send_header("Content-Disposition", f"attachment; filename={filename}")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _get_project(self, project_id: str):
        return db.get_project(project_id)

    def _check_auth(self) -> dict | None:
        """Check authentication from request headers. Returns user info or None."""
        return authenticate(dict(self.headers))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)

        # Frontend
        if path == "/":
            return self._send_html(_load_frontend())

        # OpenAPI spec
        if path == "/openapi.json":
            return self._send_json(self._full_openapi())

        # List projects
        if path == "/api/v2/projects":
            workspace_id = query.get("workspace_id", [None])[0]
            projects = db.list_projects(workspace_id=workspace_id)
            return self._send_json({"projects": projects, "count": len(projects)})

        # Project detail routes
        parts = path.strip("/").split("/")
        if len(parts) >= 4 and parts[:2] == ["api", "v2"] and parts[2] == "projects":
            project_id = parts[3]
            project_data = self._get_project(project_id)
            if not project_data:
                return self._send_json({"detail": "Project not found"}, status=404)

            if len(parts) == 4:
                return self._send_json(project_data)

            sub = parts[4]

            if sub == "visual-graph":
                vp = project_data.get("visual_pack", {})
                return self._send_json({
                    "architecture_graph": vp.get("architecture_graph", {}),
                    "data_flow_timeline": vp.get("data_flow_timeline", []),
                    "failure_mode_map": vp.get("failure_mode_map", []),
                    "dependency_view": vp.get("dependency_view", {}),
                    "mermaid_diagrams": vp.get("mermaid_diagrams", {}),
                })

            if sub == "distillation":
                return self._send_json(project_data.get("distillation", {}))

            if sub == "reproducibility":
                return self._send_json(project_data.get("reproducibility", {}))

            if sub == "code-scaffold":
                scaffold = project_data.get("code_scaffold", {})
                return self._send_json({
                    "framework": scaffold.get("framework", ""),
                    "files": [{"path": f.get("path", ""), "language": f.get("language", "")} for f in scaffold.get("files", [])],
                    "architecture_mermaid": scaffold.get("architecture_mermaid", ""),
                    "confidence": scaffold.get("confidence", 0),
                })

            if sub == "code-scaffold" and len(parts) == 6 and parts[5] == "file":
                file_path = query.get("path", [None])[0]
                scaffold = project_data.get("code_scaffold", {})
                for f in scaffold.get("files", []):
                    if f.get("path") == file_path:
                        return self._send_json({"path": f["path"], "content": f.get("content", ""), "language": f.get("language", "")})
                return self._send_json({"detail": "File not found"}, status=404)

            if sub == "agent-messages":
                return self._send_json({
                    "messages": project_data.get("agent_messages", []),
                    "count": len(project_data.get("agent_messages", [])),
                })

            if sub == "api-contract":
                return self._send_json(project_data.get("api_contract", {}))

            if sub == "deployment-plan":
                return self._send_json(project_data.get("deployment_plan", {}))

            if sub == "launch-package":
                return self._send_json(project_data.get("launch_package", {}))

            if sub == "experiments":
                if len(parts) == 5:
                    return self._send_json({"experiments": list_experiments(project_id)})
                if len(parts) == 6 and parts[5] == "dashboard":
                    return self._send_json(get_benchmark_dashboard(project_id))
                if len(parts) == 6 and parts[5] == "plan":
                    spec = project_data.get("paper_spec", {})
                    from .models.schema import PaperSpec, Equation, Citation
                    ps = PaperSpec(
                        problem=spec.get("problem", ""),
                        method=spec.get("method", ""),
                        hyperparameters=spec.get("hyperparameters", {}),
                        metrics=spec.get("metrics", []),
                        datasets=spec.get("datasets", []),
                    )
                    return self._send_json({"plans": plan_minimal_experiments(ps)})

            if sub == "reviews":
                return self._send_json({"reviews": get_reviews(project_id)})

            if sub == "export.zip":
                framework = query.get("framework", ["pytorch"])[0]
                blob = self._build_export_zip(project_data, framework)
                return self._send_bytes(blob, "application/zip", f"{project_id}.zip")

        # Workspace routes
        if len(parts) >= 3 and parts[:2] == ["api", "v2"] and parts[2] == "workspaces":
            if len(parts) == 3:
                workspaces = db.list_workspaces()
                return self._send_json({"workspaces": workspaces})
            if len(parts) == 4:
                ws = db.get_workspace(parts[3])
                if ws:
                    return self._send_json(ws)
                return self._send_json({"detail": "Workspace not found"}, status=404)
            if len(parts) == 5 and parts[4] == "projects":
                return self._send_json({"projects": list_workspace_projects(parts[3])})

        # Legacy v1 support
        if len(parts) >= 4 and parts[:2] == ["api", "v1"] and parts[2] == "projects":
            return self._handle_legacy_get(parts, query)

        self._send_json({"detail": "Not found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        payload = self._read_body()

        parts = path.strip("/").split("/")

        # Project ingestion (v2)
        if path == "/api/v2/projects/ingest":
            return self._handle_ingest(payload)

        # Project-level actions
        if len(parts) >= 5 and parts[:2] == ["api", "v2"] and parts[2] == "projects":
            project_id = parts[3]
            sub = parts[4]

            if sub == "artifacts":
                return self._handle_artifacts(project_id, payload)

            if sub == "productize":
                return self._handle_productize(project_id, payload)

            if sub == "experiments":
                if len(parts) == 5:
                    return self._handle_create_experiment(project_id, payload)
                if len(parts) == 7 and parts[6] == "start":
                    return self._send_json(start_experiment(project_id, parts[5]))
                if len(parts) == 7 and parts[6] == "complete":
                    return self._send_json(complete_experiment(
                        project_id, parts[5],
                        payload.get("metrics_observed", {}),
                        payload.get("logs"),
                    ))

            if sub == "reviews":
                return self._handle_add_review(project_id, payload)

            if sub == "approve":
                return self._send_json(approve_project(project_id, payload.get("approver", "unknown")))

        # Auth endpoints
        if path == "/api/v2/auth/login":
            result = login(payload.get("username", ""), payload.get("password", ""))
            if result:
                return self._send_json(result)
            return self._send_json({"detail": "Invalid credentials"}, status=401)

        if path == "/api/v2/auth/api-key":
            user = self._check_auth()
            if not user:
                return self._send_json({"detail": "Authentication required"}, status=401)
            key_data = generate_api_key(
                owner=user.get("user", "unknown"),
                scopes=payload.get("scopes"),
            )
            return self._send_json(key_data, status=201)

        # Workspace creation
        if path == "/api/v2/workspaces":
            ws = create_workspace(payload.get("name", "Default"), payload.get("members"))
            return self._send_json(to_dict(ws), status=201)

        # Legacy v1 support
        if path == "/api/v1/projects/ingest":
            return self._handle_ingest(payload)
        if len(parts) >= 5 and parts[:2] == ["api", "v1"] and parts[2] == "projects" and parts[4] == "artifacts":
            return self._handle_artifacts(parts[3], payload)

        self._send_json({"detail": "Not found"}, status=404)

    # -----------------------------------------------------------------------
    # Handlers
    # -----------------------------------------------------------------------
    def _handle_ingest(self, payload: dict):
        title = payload.get("title", "").strip()
        abstract = payload.get("abstract", "").strip()
        method_text = payload.get("method_text", "").strip()

        if len(title) < 4 or len(abstract) < 40 or len(method_text) < 80:
            return self._send_json({"detail": "Invalid input lengths. title>=4, abstract>=40, method_text>=80."}, status=400)

        request = IngestRequest(
            title=title,
            abstract=abstract,
            method_text=method_text,
            source_url=payload.get("source_url"),
            arxiv_id=payload.get("arxiv_id"),
            github_url=payload.get("github_url"),
            persona=payload.get("persona", "ml_engineer"),
        )

        framework = payload.get("framework", "pytorch")
        project = run_pipeline(request, framework)
        db.save_project(project)

        result = to_dict(project)
        return self._send_json(result)

    def _handle_artifacts(self, project_id: str, payload: dict):
        project_data = self._get_project(project_id)
        if not project_data:
            return self._send_json({"detail": "Project not found"}, status=404)

        scaffold = project_data.get("code_scaffold", {})
        files = scaffold.get("files", [])
        return self._send_json({
            "project_id": project_id,
            "framework": scaffold.get("framework", "pytorch"),
            "files": [f.get("path", "") for f in files],
            "confidence_score": project_data.get("confidence_score", 0),
            "architecture_mermaid": scaffold.get("architecture_mermaid", ""),
        })

    def _handle_productize(self, project_id: str, payload: dict):
        project_data = self._get_project(project_id)
        if not project_data:
            return self._send_json({"detail": "Project not found"}, status=404)

        arch_type = payload.get("architecture_type", "server")

        from .models.schema import PaperSpec, CodeScaffold, Equation, Citation
        spec_data = project_data.get("paper_spec", {})
        spec = PaperSpec(
            problem=spec_data.get("problem", ""),
            method=spec_data.get("method", ""),
            metrics=spec_data.get("metrics", []),
            datasets=spec_data.get("datasets", []),
            hyperparameters=spec_data.get("hyperparameters", {}),
        )
        scaffold = CodeScaffold(framework=project_data.get("code_scaffold", {}).get("framework", "pytorch"))

        api_contract = generate_api_contract(spec, scaffold)
        deployment = generate_deployment_plan(spec, arch_type)
        launch = generate_launch_package(spec, api_contract, deployment)

        result = {
            "api_contract": to_dict(api_contract),
            "deployment_plan": to_dict(deployment),
            "launch_package": to_dict(launch),
        }

        # Update project in DB
        project_data["api_contract"] = to_dict(api_contract)
        project_data["deployment_plan"] = to_dict(deployment)
        project_data["launch_package"] = to_dict(launch)
        project_data["status"] = "ready_for_review"
        # Re-save
        project = Project(**{k: v for k, v in project_data.items() if k in Project.__dataclass_fields__})
        project.api_contract = api_contract
        project.deployment_plan = deployment
        project.launch_package = launch
        project.status = ProjectStatus.READY_FOR_REVIEW
        db.save_project(project)

        return self._send_json(result)

    def _handle_create_experiment(self, project_id: str, payload: dict):
        run = create_experiment(
            project_id,
            payload.get("config", {}),
            payload.get("expected_metrics"),
        )
        return self._send_json(to_dict(run), status=201)

    def _handle_add_review(self, project_id: str, payload: dict):
        review = add_review(
            project_id,
            payload.get("author", "anonymous"),
            payload.get("content", ""),
            payload.get("artifact_path"),
            payload.get("line_number"),
        )
        return self._send_json(to_dict(review), status=201)

    def _build_export_zip(self, project_data: dict, framework: str) -> bytes:
        """Build comprehensive ZIP export."""
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
            # Code scaffold files
            scaffold = project_data.get("code_scaffold", {})
            for f in scaffold.get("files", []):
                zf.writestr(f.get("path", "unknown"), f.get("content", ""))

            # Mermaid diagrams (architecture + visual pack)
            written_visuals = set()
            mermaid = scaffold.get("architecture_mermaid", "")
            if mermaid:
                zf.writestr("visuals/architecture.mmd", mermaid)
                written_visuals.add("architecture")

            vp = project_data.get("visual_pack", {})
            for name, diagram in vp.get("mermaid_diagrams", {}).items():
                if name not in written_visuals:
                    zf.writestr(f"visuals/{name}.mmd", diagram)

            # Reports
            zf.writestr("reports/paper_spec.json", json.dumps(project_data.get("paper_spec", {}), indent=2, default=str))
            zf.writestr("reports/reproducibility.json", json.dumps(project_data.get("reproducibility", {}), indent=2, default=str))
            zf.writestr("reports/distillation.json", json.dumps(project_data.get("distillation", {}), indent=2, default=str))

            # API contract
            api = project_data.get("api_contract", {})
            if api:
                zf.writestr("api/openapi.json", json.dumps(api.get("openapi_spec", {}), indent=2, default=str))
                for lang, code in api.get("sdk_stubs", {}).items():
                    ext = {"python": "py", "javascript": "js", "curl": "sh"}.get(lang, "txt")
                    zf.writestr(f"sdk/client.{ext}", code)

            # Deployment
            deployment = project_data.get("deployment_plan", {})
            if deployment:
                zf.writestr("deploy/plan.json", json.dumps(deployment, indent=2, default=str))

            # Launch checklist
            launch = project_data.get("launch_package", {})
            if launch:
                zf.writestr("launch/package.json", json.dumps(launch, indent=2, default=str))

            # Agent trace log
            messages = project_data.get("agent_messages", [])
            if messages:
                log_lines = []
                for m in messages:
                    role = m.get("role", "unknown")
                    content = m.get("content", "")
                    conf = m.get("confidence", 1.0)
                    log_lines.append(f"[{role}] (confidence: {conf}) {content}")
                zf.writestr("reports/agent_trace.log", "\n".join(log_lines))

        return mem.getvalue()

    # Legacy v1 GET handler
    def _handle_legacy_get(self, parts, query):
        if len(parts) < 4:
            return self._send_json({"detail": "Not found"}, status=404)
        project_id = parts[3]
        project_data = self._get_project(project_id)
        if not project_data:
            return self._send_json({"detail": "Project not found"}, status=404)

        if len(parts) == 4:
            return self._send_json(project_data)
        sub = parts[4]
        if sub == "visual-graph":
            vp = project_data.get("visual_pack", {})
            scaffold = project_data.get("code_scaffold", {})
            return self._send_json({
                "nodes": [{"id": "paper"}, {"id": "method"}, {"id": "eval"}],
                "edges": [{"source": "paper", "target": "method"}, {"source": "method", "target": "eval"}],
                "mermaid": scaffold.get("architecture_mermaid", ""),
            })
        if sub == "distillation":
            return self._send_json(project_data.get("distillation", {}))
        if sub == "export.zip":
            framework = query.get("framework", ["pytorch"])[0]
            blob = self._build_export_zip(project_data, framework)
            self._send_bytes(blob, "application/zip", f"{project_id}.zip")
            return

        return self._send_json({"detail": "Not found"}, status=404)

    def _full_openapi(self) -> dict:
        return {
            "openapi": "3.0.3",
            "info": {
                "title": "Paper2Product 2.0 — AI Research OS",
                "version": "2.0.0",
                "description": "Full-stack research-to-product platform API.",
            },
            "paths": {
                "/api/v2/projects/ingest": {
                    "post": {
                        "summary": "Ingest paper and run multi-agent pipeline",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["title", "abstract", "method_text"],
                                        "properties": {
                                            "title": {"type": "string", "minLength": 4},
                                            "abstract": {"type": "string", "minLength": 40},
                                            "method_text": {"type": "string", "minLength": 80},
                                            "source_url": {"type": "string"},
                                            "arxiv_id": {"type": "string"},
                                            "github_url": {"type": "string"},
                                            "persona": {"type": "string", "enum": ["founder_pm", "ml_engineer", "designer_educator", "mobile_dev"]},
                                            "framework": {"type": "string", "enum": ["pytorch", "tensorflow"]},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
                "/api/v2/projects": {"get": {"summary": "List all projects"}},
                "/api/v2/projects/{id}": {"get": {"summary": "Get project details"}},
                "/api/v2/projects/{id}/artifacts": {"post": {"summary": "Get generated artifacts"}},
                "/api/v2/projects/{id}/visual-graph": {"get": {"summary": "Get visual graph pack"}},
                "/api/v2/projects/{id}/distillation": {"get": {"summary": "Get research distillation"}},
                "/api/v2/projects/{id}/reproducibility": {"get": {"summary": "Get reproducibility scorecard"}},
                "/api/v2/projects/{id}/code-scaffold": {"get": {"summary": "Get code scaffold details"}},
                "/api/v2/projects/{id}/agent-messages": {"get": {"summary": "Get agent pipeline trace"}},
                "/api/v2/projects/{id}/api-contract": {"get": {"summary": "Get generated API contract"}},
                "/api/v2/projects/{id}/deployment-plan": {"get": {"summary": "Get deployment plan"}},
                "/api/v2/projects/{id}/launch-package": {"get": {"summary": "Get launch package"}},
                "/api/v2/projects/{id}/productize": {"post": {"summary": "Generate productization assets"}},
                "/api/v2/projects/{id}/experiments": {
                    "get": {"summary": "List experiments"},
                    "post": {"summary": "Create experiment"},
                },
                "/api/v2/projects/{id}/experiments/dashboard": {"get": {"summary": "Get benchmark dashboard"}},
                "/api/v2/projects/{id}/experiments/plan": {"get": {"summary": "Get experiment plans"}},
                "/api/v2/projects/{id}/experiments/{run_id}/start": {"post": {"summary": "Start experiment"}},
                "/api/v2/projects/{id}/experiments/{run_id}/complete": {"post": {"summary": "Complete experiment with results"}},
                "/api/v2/projects/{id}/reviews": {
                    "get": {"summary": "Get reviews"},
                    "post": {"summary": "Add review comment"},
                },
                "/api/v2/projects/{id}/approve": {"post": {"summary": "Approve project"}},
                "/api/v2/projects/{id}/export.zip": {"get": {"summary": "Download full export bundle"}},
                "/api/v2/workspaces": {
                    "get": {"summary": "List workspaces"},
                    "post": {"summary": "Create workspace"},
                },
                "/api/v2/workspaces/{id}": {"get": {"summary": "Get workspace"}},
                "/api/v2/workspaces/{id}/projects": {"get": {"summary": "List workspace projects"}},
            },
        }


def run(host: str = "0.0.0.0", port: int = 8000):
    db.init_db()
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Paper2Product 2.0 running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()

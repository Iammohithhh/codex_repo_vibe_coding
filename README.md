# Paper2Product 2.0 — AI Research OS

Transform research papers into production-ready assets: runnable implementations, interactive visualizations, productized APIs/SDKs, and deployable demos — with full traceability from paper to product.

## What's Implemented

### Multi-Agent Pipeline (5 agents)
- **Reader Agent** — Extracts structured facts: equations, datasets, metrics, claims, hyperparameters, architecture components, with sentence-level citations
- **Skeptic Agent** — Challenges missing details, flags low-confidence steps, computes reproducibility scorecard with blockers and fixes
- **Implementer Agent** — Generates modular, framework-specific code (PyTorch/TensorFlow) with `core/`, `train/`, `eval/`, `inference/` pipelines, configs, tests, CI, Dockerfile
- **Verifier Agent** — Validates code structure, checks quality, generates experiment plans (smoke test, quick convergence, full replication)
- **Explainer Agent** — Creates multi-depth summaries (ELI5/practitioner/researcher), knowledge graphs, Mermaid diagrams, failure mode maps, counterfactual analysis, quiz cards

### Platform Features
- **Reproducibility Intelligence** — Hyperparameter completeness scoring, experiment planner, benchmark delta reports, reproducibility scorecard
- **Productization Layer** — API contract generation (OpenAPI 3.0), SDK stubs (Python/JS/cURL), deployment plans (server/edge/mobile/hybrid), cost estimation, security checklists
- **Experiment Lab** — Create/track experiments, benchmark dashboards, minimal compute planning
- **Team Collaboration** — Shared workspaces, review comments on artifacts, approval workflows
- **Launch Center** — Web deploy steps, mobile release checklist, telemetry config, Play Store readiness
- **SQLite Persistence** — All projects, experiments, reviews, and workspaces persisted

### Web Frontend
Rich single-page application with 6 modules:
1. **Project Intake** — Paper upload with persona selection and framework choice
2. **Artifact Studio** — Code viewer, visual graphs, distillation, multi-depth summaries
3. **Experiment Lab** — Run configs, experiment plans, benchmark dashboard
4. **API Builder** — Generated endpoints, SDK stubs, OpenAPI spec viewer
5. **Launch Center** — Deployment plans, security checklist, release readiness
6. **Team Review** — Comments, approvals, governance

### Flutter Mobile Companion App (Android-first)
- Browse and monitor projects
- View visual knowledge graphs and data flow timelines
- Trigger artifact builds and productization
- Review confidence and risk reports
- Approve releases
- Push notification support (Firebase)

### API (v2 + legacy v1 support)
Full REST API with 25+ endpoints covering the entire pipeline.

## Quick Start

```bash
# Run v2 server (default)
python -m paper2product

# Run on custom port
python -m paper2product --port=9000

# Run legacy v1 server
python -m paper2product --legacy

# Direct module run
python -m paper2product.server_v2
```

Open in browser: `http://127.0.0.1:8000`

## API Endpoints (v2)

### Core Pipeline
```
POST /api/v2/projects/ingest          # Ingest paper and run multi-agent pipeline
GET  /api/v2/projects                 # List all projects
GET  /api/v2/projects/{id}            # Get project details
GET  /api/v2/projects/{id}/visual-graph      # Knowledge graph + Mermaid diagrams
GET  /api/v2/projects/{id}/distillation      # Research distillation package
GET  /api/v2/projects/{id}/reproducibility   # Reproducibility scorecard
GET  /api/v2/projects/{id}/code-scaffold     # Code scaffold details
GET  /api/v2/projects/{id}/agent-messages    # Agent pipeline trace
GET  /api/v2/projects/{id}/export.zip        # Full export bundle
```

### Productization
```
POST /api/v2/projects/{id}/productize        # Generate API contract + deploy plan + launch package
GET  /api/v2/projects/{id}/api-contract      # OpenAPI spec + SDK stubs
GET  /api/v2/projects/{id}/deployment-plan   # Deployment architecture + cost + security
GET  /api/v2/projects/{id}/launch-package    # Web + mobile release config
```

### Experiments
```
GET  /api/v2/projects/{id}/experiments           # List experiments
POST /api/v2/projects/{id}/experiments           # Create experiment
GET  /api/v2/projects/{id}/experiments/dashboard # Benchmark dashboard
GET  /api/v2/projects/{id}/experiments/plan      # Minimal compute experiment plans
POST /api/v2/projects/{id}/experiments/{run}/start    # Start experiment
POST /api/v2/projects/{id}/experiments/{run}/complete  # Complete with results
```

### Collaboration
```
GET  /api/v2/projects/{id}/reviews     # List reviews
POST /api/v2/projects/{id}/reviews     # Add review comment
POST /api/v2/projects/{id}/approve     # Approve project
GET  /api/v2/workspaces               # List workspaces
POST /api/v2/workspaces               # Create workspace
```

## Run Tests

```bash
# All tests
python -m unittest discover tests -v

# Agent unit tests (15 tests)
python -m unittest tests.test_agents -v

# V2 E2E tests (8 tests)
python -m unittest tests.test_v2_e2e -v

# Legacy V1 E2E test
python -m unittest tests.test_phase1_e2e -v
```

## Project Structure

```
paper2product/
├── __init__.py
├── __main__.py                  # Entry point
├── core.py                      # Legacy v1 wrapper (backwards-compatible)
├── server.py                    # Legacy v1 HTTP server
├── server_v2.py                 # V2 full-featured HTTP server
├── models/
│   └── schema.py                # All data models (30+ dataclasses)
├── agents/
│   ├── pipeline.py              # Multi-agent orchestrator
│   ├── reader.py                # Paper extraction agent
│   ├── skeptic.py               # Critical review agent
│   ├── implementer.py           # Code generation agent
│   ├── verifier.py              # Validation agent
│   └── explainer.py             # Visual/intuition agent
├── services/
│   ├── persistence.py           # SQLite storage layer
│   ├── productization.py        # API contracts, deployment, launch
│   ├── collaboration.py         # Workspaces, reviews, approvals
│   └── experiment_lab.py        # Experiment tracking and benchmarks
└── templates/

frontend/
└── index.html                   # Rich SPA with 6 modules

mobile/
├── pubspec.yaml                 # Flutter project config
└── lib/
    ├── main.dart                # App entry + navigation
    ├── services/
    │   └── api_service.dart     # Backend API client
    └── screens/
        ├── home_screen.dart     # Dashboard
        ├── project_detail_screen.dart  # 5-tab project view
        ├── visual_map_screen.dart      # Knowledge graphs
        └── settings_screen.dart        # Server config

tests/
├── test_agents.py               # 15 agent unit tests
├── test_v2_e2e.py              # 8 v2 API e2e tests
└── test_phase1_e2e.py          # Legacy v1 e2e test
```

## Architecture

```
Paper Upload -> Reader Agent -> Skeptic Agent -> Implementer Agent -> Verifier Agent -> Explainer Agent
                    |               |                |                  |                |
              PaperSpec    Reproducibility     CodeScaffold      ExperimentRuns      VisualPack
                                Score          (14 files)          (3 plans)        + Distillation
                                                                                   + Quiz Cards
                                                   |
                                            Productization
                                         (API + Deploy + Launch)
```

## No External Dependencies

The entire platform runs on Python 3.7+ standard library only. Generated code output targets PyTorch or TensorFlow, but the platform itself requires zero pip installs.

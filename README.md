# Paper2Product Phase 2 Backend

This repo now includes the requested **next-step implementation** on top of Phase 1:
- persistent storage,
- authentication + ownership,
- rate limiting,
- async run queue,
- dashboard/run history/artifact versioning,
- trust-layer outputs.

## Implemented now

### 1) Hardened backend foundations
- Persistent relational storage via SQLite (`paper2product.db`) for users, projects, runs, artifacts, and rate limits.
- Object storage via filesystem-backed `object_store/` for ZIP artifacts.
- Per-user project ownership enforced via API token auth (`X-API-Key`).
- Rate limiting per token.

### 2) Better extraction structure
- Added section-aware extraction (`problem`, `method`, `training`, `evaluation`) in `paper_spec.sections`.
- Continues equation/dataset/metric extraction while returning explicit uncertainty flags.

### 3) Scalable execution model
- Added async queue worker for long-running jobs:
  - `run_type=artifacts`
  - `run_type=distillation`
- Run status lifecycle: `queued -> running -> completed|failed`.
- Retry-ready architecture via queued run model.

### 4) Dashboard + run history + artifact versioning
- Dashboard endpoint lists projects with run counts and latest artifact versions.
- Artifact versions increment per project (`v1`, `v2`, ...).
- Latest artifact metadata and ZIP download endpoints are exposed.

### 5) Trust layer
- Confidence breakdown (extraction, equation coverage, reproducibility, citation traceability).
- Overall confidence score.
- Explicit uncertainty flags.
- Citation mapping by generated file.

## API surface
- `POST /api/v1/auth/register`
- `POST /api/v1/projects/ingest`
- `GET /api/v1/projects/{id}`
- `POST /api/v1/projects/{id}/runs`
- `GET /api/v1/projects/{id}/runs`
- `GET /api/v1/runs/{run_id}`
- `GET /api/v1/projects/{id}/artifacts/latest`
- `GET /api/v1/projects/{id}/visual-graph`
- `GET /api/v1/projects/{id}/distillation`
- `GET /api/v1/projects/{id}/export.zip`
- `GET /api/v1/dashboard`
- `GET /openapi.json`

## Run locally
```bash
python -m paper2product.server
```

Then open:
- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/openapi.json`

## Test
```bash
python -m unittest discover -s tests -v
```

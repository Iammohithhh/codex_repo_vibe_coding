# Paper2Product Phase 1 MVP (End-to-End)

This repo now contains a working end-to-end Phase 1 implementation with a browser workspace, APIs, and ZIP export.

## What is implemented

### Phase 1 (Weeks 1-4) fully covered
- Ingestion + structured extraction to `PaperSpec`.
- Citation-aware code scaffold generation.
- Basic validation/confidence output in artifact response.
- Architecture diagram generation (Mermaid + graph JSON endpoint).
- Web workspace to run the full flow end-to-end.
- ZIP export containing code, visuals, and report.
- OpenAPI spec endpoint at `/openapi.json`.

### Immediate steps from prior plan implemented
1. `PaperSpec` ingestion schema.
2. Citation-aware code generation templates.
3. Verification signal (confidence + smoke test scaffold).
4. Visual graph renderer payload.
5. OpenAPI exposure.
6. Mobile support planning handled through API-first architecture and export bundles.
7. Launch checklist content included in distillation outputs and project plan.

### New: Research Distillation & Extension Layer
`GET /api/v1/projects/{id}/distillation` returns:
- Poster generator payload (problem/method/equations/architecture/results/limitations/QR/formats).
- Visual notes / knowledge cards (concept cards, equation intuition panels, failure cards).
- Executive takeaways (1-page brief points + 5-slide deck outline).
- Extension engine (what-next ideas, missing-piece detector, research roadmap).
- Personalized learning path (beginner and advanced tracks).

## Run
```bash
python -m paper2product.server
```

Open in browser:
- App UI: `http://127.0.0.1:8000`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

## API sequence
1. `POST /api/v1/projects/ingest`
2. `POST /api/v1/projects/{id}/artifacts`
3. `GET /api/v1/projects/{id}/visual-graph`
4. `GET /api/v1/projects/{id}/distillation`
5. `GET /api/v1/projects/{id}/export.zip`

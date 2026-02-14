const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';

async function fetchJSON(path: string, opts: RequestInit = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...opts.headers as any },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('json')) return res.json();
  return res.blob();
}

export const api = {
  ingest: (data: any) => fetchJSON('/api/v2/projects/ingest', { method: 'POST', body: JSON.stringify(data) }),
  listProjects: () => fetchJSON('/api/v2/projects'),
  getProject: (id: string) => fetchJSON(`/api/v2/projects/${id}`),
  getVisualGraph: (id: string) => fetchJSON(`/api/v2/projects/${id}/visual-graph`),
  getDistillation: (id: string) => fetchJSON(`/api/v2/projects/${id}/distillation`),
  getReproducibility: (id: string) => fetchJSON(`/api/v2/projects/${id}/reproducibility`),
  getCodeScaffold: (id: string) => fetchJSON(`/api/v2/projects/${id}/code-scaffold`),
  getAgentMessages: (id: string) => fetchJSON(`/api/v2/projects/${id}/agent-messages`),
  getApiContract: (id: string) => fetchJSON(`/api/v2/projects/${id}/api-contract`),
  getDeploymentPlan: (id: string) => fetchJSON(`/api/v2/projects/${id}/deployment-plan`),
  getLaunchPackage: (id: string) => fetchJSON(`/api/v2/projects/${id}/launch-package`),
  productize: (id: string, data: any) => fetchJSON(`/api/v2/projects/${id}/productize`, { method: 'POST', body: JSON.stringify(data) }),
  getExperiments: (id: string) => fetchJSON(`/api/v2/projects/${id}/experiments`),
  getExperimentPlans: (id: string) => fetchJSON(`/api/v2/projects/${id}/experiments/plan`),
  createExperiment: (id: string, data: any) => fetchJSON(`/api/v2/projects/${id}/experiments`, { method: 'POST', body: JSON.stringify(data) }),
  getReviews: (id: string) => fetchJSON(`/api/v2/projects/${id}/reviews`),
  addReview: (id: string, data: any) => fetchJSON(`/api/v2/projects/${id}/reviews`, { method: 'POST', body: JSON.stringify(data) }),
  approve: (id: string, approver: string) => fetchJSON(`/api/v2/projects/${id}/approve`, { method: 'POST', body: JSON.stringify({ approver }) }),
  exportZip: (id: string) => `${API_BASE}/api/v2/projects/${id}/export.zip`,
};

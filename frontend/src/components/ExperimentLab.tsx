'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';

interface Props { project: any; }

export function ExperimentLab({ project }: Props) {
  const [experiments, setExperiments] = useState<any>(null);
  const [plans, setPlans] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!project?.id) return;
    setLoading(true);
    Promise.all([api.getExperiments(project.id), api.getExperimentPlans(project.id)])
      .then(([e, p]) => { setExperiments(e); setPlans(p); })
      .finally(() => setLoading(false));
  }, [project?.id]);

  if (!project) return <div className="card"><p style={{ color: 'var(--text-dim)' }}>Ingest a paper first to access experiments.</p></div>;
  if (loading) return <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: 40, color: 'var(--text-dim)' }}><span className="spinner" /> Loading...</div>;

  const handleCreate = async () => {
    const hp = project.paper_spec?.hyperparameters || {};
    await api.createExperiment(project.id, { config: { ...hp, epochs: 1 }, expected_metrics: {} });
    const e = await api.getExperiments(project.id);
    setExperiments(e);
  };

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Experiment Lab</div>
        <div className="page-desc">Run configs, benchmark dashboard, and experiment tracking.</div>
      </div>
      <div className="card">
        <div className="card-header">
          <div className="card-title">Experiment Plans</div>
          <span className="badge badge-blue">{(plans?.plans || []).length} planned</span>
        </div>
        <table>
          <thead><tr><th>Name</th><th>Purpose</th><th>Priority</th></tr></thead>
          <tbody>
            {(plans?.plans || []).map((p: any, i: number) => (
              <tr key={i}><td><strong>{p.name}</strong></td><td style={{ fontSize: 12 }}>{p.purpose}</td><td><span className={`badge badge-${p.priority <= 2 ? 'green' : 'blue'}`}>P{p.priority}</span></td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="card">
        <div className="card-header">
          <div className="card-title">Experiment Runs</div>
          <button className="btn btn-sm btn-primary" onClick={handleCreate}>New Experiment</button>
        </div>
        {(experiments?.experiments || []).length === 0 ? (
          <p style={{ color: 'var(--text-dim)', fontSize: 13 }}>No experiments run yet.</p>
        ) : (
          <table>
            <thead><tr><th>ID</th><th>Status</th><th>Config</th></tr></thead>
            <tbody>
              {(experiments?.experiments || []).map((e: any) => (
                <tr key={e.id}><td style={{ fontFamily: 'monospace', fontSize: 12 }}>{(e.id || '').slice(0, 8)}</td><td><span className={`badge badge-${e.status === 'completed' ? 'green' : e.status === 'running' ? 'blue' : 'orange'}`}>{e.status}</span></td><td style={{ fontSize: 12 }}>{JSON.stringify(e.config || {}).slice(0, 80)}</td></tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

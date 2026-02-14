'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';

interface Props { project: any; }

export function LaunchCenter({ project }: Props) {
  const [deploy, setDeploy] = useState<any>(null);
  const [launch, setLaunch] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!project?.id) return;
    setLoading(true);
    Promise.all([api.getDeploymentPlan(project.id), api.getLaunchPackage(project.id)])
      .then(([d, l]) => { setDeploy(d); setLaunch(l); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [project?.id]);

  if (!project) return <div className="card"><p style={{ color: 'var(--text-dim)' }}>Ingest a paper first.</p></div>;
  if (loading) return <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: 40, color: 'var(--text-dim)' }}><span className="spinner" /> Loading...</div>;

  if (!deploy?.architecture_type && !launch?.web_deploy) {
    return (
      <div>
        <div className="page-header"><div className="page-title">Launch Center</div></div>
        <div className="card">
          <p style={{ color: 'var(--text-dim)', marginBottom: 12 }}>No deployment plan yet.</p>
          <button className="btn btn-primary" onClick={async () => {
            setLoading(true);
            try { await api.productize(project.id, { architecture_type: 'server' }); const [d, l] = await Promise.all([api.getDeploymentPlan(project.id), api.getLaunchPackage(project.id)]); setDeploy(d); setLaunch(l); } finally { setLoading(false); }
          }}>Generate Launch Package</button>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header"><div className="page-title">Launch Center</div><div className="page-desc">Web deploy, mobile release, and staging config.</div></div>

      <div className="grid grid-2" style={{ marginBottom: 16 }}>
        <div className="card stat-card"><div className="stat-value" style={{ color: 'var(--accent-light)' }}>{deploy?.architecture_type || 'server'}</div><div className="stat-label">Architecture</div></div>
        <div className="card stat-card"><div className="stat-value" style={{ color: 'var(--green)' }}>${deploy?.estimated_cost_monthly || 0}/mo</div><div className="stat-label">Estimated Cost</div></div>
      </div>

      <div className="card">
        <div className="card-title">Security Checklist</div>
        {(deploy?.security_checklist || []).map((item: any, i: number) => (
          <div key={i} className="checklist-item">
            <div className="checklist-check">{item.status === 'required' ? '!' : '?'}</div>
            <span>{item.item}</span>
            <span className={`badge badge-${item.priority === 'high' ? 'red' : item.priority === 'medium' ? 'orange' : 'green'}`} style={{ marginLeft: 8 }}>{item.priority}</span>
          </div>
        ))}
      </div>

      <div className="grid grid-2">
        <div className="card">
          <div className="card-title">Web Deployment</div>
          {(launch?.web_deploy?.deployment_steps || []).map((s: string, i: number) => (
            <div key={i} className="checklist-item"><div className="checklist-check">{i + 1}</div><span>{s}</span></div>
          ))}
        </div>
        <div className="card">
          <div className="card-title">Mobile Release</div>
          <div style={{ marginBottom: 8 }}><strong>Platform:</strong> {launch?.mobile_release?.platform}</div>
          <div style={{ marginBottom: 8 }}><strong>Package:</strong> <code>{launch?.mobile_release?.package_name}</code></div>
          <div><strong>Integrations:</strong></div>
          {(launch?.mobile_release?.integrations || []).map((i: string, idx: number) => (
            <div key={idx} className="checklist-item"><div className="checklist-check">+</div><span>{i}</span></div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="card-title">Release Checklist</div>
        {(launch?.release_checklist || []).map((item: any, i: number) => (
          <div key={i} className="checklist-item">
            <div className="checklist-check">{item.required ? '!' : '-'}</div>
            <span>{item.item}</span>
            <span className="badge badge-blue" style={{ marginLeft: 8 }}>{item.category}</span>
            {item.required && <span className="badge badge-red" style={{ marginLeft: 4 }}>required</span>}
          </div>
        ))}
      </div>

      <div style={{ marginTop: 12 }}>
        <a className="btn btn-success" href={api.exportZip(project.id)} download>Download Full Launch Package</a>
      </div>
    </div>
  );
}

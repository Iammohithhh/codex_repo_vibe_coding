'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';

interface Props { project: any; }

export function ApiBuilder({ project }: Props) {
  const [contract, setContract] = useState<any>(null);
  const [tab, setTab] = useState('endpoints');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!project?.id) return;
    setLoading(true);
    api.getApiContract(project.id).then(setContract).catch(() => {}).finally(() => setLoading(false));
  }, [project?.id]);

  const handleProductize = async () => {
    if (!project?.id) return;
    setLoading(true);
    try {
      const result = await api.productize(project.id, { architecture_type: 'server' });
      setContract(result.api_contract);
    } finally { setLoading(false); }
  };

  if (!project) return <div className="card"><p style={{ color: 'var(--text-dim)' }}>Ingest a paper first.</p></div>;
  if (loading) return <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: 40, color: 'var(--text-dim)' }}><span className="spinner" /> Loading...</div>;

  if (!contract || !contract.endpoints || contract.endpoints.length === 0) {
    return (
      <div>
        <div className="page-header"><div className="page-title">API Builder</div></div>
        <div className="card">
          <p style={{ color: 'var(--text-dim)', marginBottom: 12 }}>No API contract yet.</p>
          <button className="btn btn-primary" onClick={handleProductize}>Generate Productization Assets</button>
        </div>
      </div>
    );
  }

  const tabs = [{ id: 'endpoints', label: 'Endpoints' }, { id: 'sdk', label: 'SDK Stubs' }, { id: 'spec', label: 'OpenAPI Spec' }];

  return (
    <div>
      <div className="page-header"><div className="page-title">API Builder</div><div className="page-desc">Generated OpenAPI specs, SDK stubs, and API contracts.</div></div>
      <div className="tabs">
        {tabs.map(t => <button key={t.id} className={`tab ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>)}
      </div>

      {tab === 'endpoints' && (
        <>
          <div className="card">
            <div className="card-title">API Endpoints</div>
            <table>
              <thead><tr><th>Method</th><th>Path</th><th>Summary</th></tr></thead>
              <tbody>
                {(contract.endpoints || []).map((ep: any, i: number) => (
                  <tr key={i}><td><span className={`badge badge-${ep.method === 'GET' ? 'green' : 'blue'}`}>{ep.method}</span></td><td><code>{ep.path}</code></td><td style={{ fontSize: 13 }}>{ep.summary}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="grid grid-2">
            <div className="card">
              <div className="card-title">Rate Limits</div>
              {Object.entries(contract.rate_limits || {}).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', fontSize: 13 }}><span>{k}</span><span>{v as number} req/min</span></div>
              ))}
            </div>
            <div className="card">
              <div className="card-title">Auth Config</div>
              {Object.entries(contract.auth_config || {}).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', fontSize: 13 }}><span>{k}</span><span>{v as string}</span></div>
              ))}
            </div>
          </div>
        </>
      )}

      {tab === 'sdk' && Object.entries(contract.sdk_stubs || {}).map(([lang, code]) => (
        <div key={lang} className="card">
          <div className="card-title">{lang.charAt(0).toUpperCase() + lang.slice(1)} SDK</div>
          <pre>{code as string}</pre>
        </div>
      ))}

      {tab === 'spec' && (
        <div className="card">
          <div className="card-title">OpenAPI 3.0 Specification</div>
          <pre>{JSON.stringify(contract.openapi_spec || {}, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

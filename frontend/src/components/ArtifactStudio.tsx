'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { KnowledgeGraph } from './KnowledgeGraph';

interface Props {
  project: any;
}

export function ArtifactStudio({ project }: Props) {
  const [tab, setTab] = useState('code');
  const [scaffold, setScaffold] = useState<any>(null);
  const [visual, setVisual] = useState<any>(null);
  const [distill, setDistill] = useState<any>(null);
  const [selectedFile, setSelectedFile] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!project?.id) return;
    setLoading(true);
    Promise.all([
      api.getCodeScaffold(project.id),
      api.getVisualGraph(project.id),
      api.getDistillation(project.id),
    ]).then(([s, v, d]) => {
      setScaffold(s);
      setVisual(v);
      setDistill(d);
    }).finally(() => setLoading(false));
  }, [project?.id]);

  if (!project) return <div className="card"><p style={{ color: 'var(--text-dim)' }}>Ingest a paper first to see generated artifacts.</p></div>;
  if (loading) return <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: 40, color: 'var(--text-dim)' }}><span className="spinner" /> Loading artifacts...</div>;

  const tabs = [
    { id: 'code', label: 'Code Scaffold' },
    { id: 'visuals', label: 'Visual Graphs' },
    { id: 'distill', label: 'Distillation' },
    { id: 'summaries', label: 'Summaries' },
  ];

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Artifact Studio</div>
        <div className="page-desc">Generated code, visuals, summaries, and reports.</div>
      </div>

      <div className="tabs">
        {tabs.map(t => (
          <button key={t.id} className={`tab ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </div>

      {tab === 'code' && scaffold && (
        <>
          <div className="card">
            <div className="card-header">
              <div className="card-title">Generated Files ({(scaffold.files || []).length})</div>
              <span className="badge badge-blue">{scaffold.framework}</span>
            </div>
            {(scaffold.files || []).map((f: any) => (
              <div key={f.path} className="file-item" onClick={() => {
                const file = project.code_scaffold?.files?.find((x: any) => x.path === f.path);
                setSelectedFile(file || f);
              }}>
                {f.path} <span style={{ color: 'var(--text-dim)', fontSize: 11 }}>{f.language || ''}</span>
              </div>
            ))}
          </div>
          {scaffold.architecture_mermaid && (
            <div className="card">
              <div className="card-title">Architecture Diagram</div>
              <pre>{scaffold.architecture_mermaid}</pre>
            </div>
          )}
          {selectedFile && (
            <div className="card">
              <div className="card-header">
                <div className="card-title">{selectedFile.path}</div>
                <span className="badge badge-blue">{selectedFile.language || 'text'}</span>
              </div>
              <pre>{selectedFile.content || 'No content available (fetch from export ZIP)'}</pre>
            </div>
          )}
        </>
      )}

      {tab === 'visuals' && visual && (
        <>
          <div className="card">
            <div className="card-title">Knowledge Graph</div>
            <KnowledgeGraph graphData={visual.architecture_graph || {}} />
          </div>
          {Object.entries(visual.mermaid_diagrams || {}).map(([name, diagram]) => (
            <div key={name} className="card">
              <div className="card-title">{name.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase())} Diagram</div>
              <pre>{diagram as string}</pre>
            </div>
          ))}
          <div className="card">
            <div className="card-title">Data Flow Timeline</div>
            {(visual.data_flow_timeline || []).map((step: any) => (
              <div key={step.step} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '8px 0', borderBottom: '1px solid var(--border)' }}>
                <span className="badge badge-blue" style={{ minWidth: 28, textAlign: 'center' }}>{step.step}</span>
                <div><strong>{step.name}</strong><div style={{ fontSize: 12, color: 'var(--text-dim)' }}>{step.description}</div></div>
              </div>
            ))}
          </div>
          <div className="card">
            <div className="card-title">Failure Mode Map</div>
            <table>
              <thead><tr><th>Scenario</th><th>Impact</th><th>Mitigation</th><th>Severity</th></tr></thead>
              <tbody>
                {(visual.failure_mode_map || []).map((f: any, i: number) => (
                  <tr key={i}>
                    <td>{f.scenario}</td><td>{f.impact}</td><td>{f.mitigation}</td>
                    <td><span className={`badge badge-${f.severity === 'high' ? 'red' : f.severity === 'medium' ? 'orange' : 'green'}`}>{f.severity}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {tab === 'distill' && distill && (
        <>
          <div className="card">
            <div className="card-title">Research Poster</div>
            <div className="grid grid-2" style={{ marginTop: 12 }}>
              <div><strong>Problem:</strong><p style={{ fontSize: 13, marginTop: 4 }}>{distill.poster?.problem}</p></div>
              <div><strong>Method:</strong><p style={{ fontSize: 13, marginTop: 4 }}>{distill.poster?.method}</p></div>
            </div>
          </div>
          <div className="card">
            <div className="card-title">Equation Intuition</div>
            {(distill.knowledge_cards?.equation_intuition_panels || []).map((eq: any, i: number) => (
              <div key={i} style={{ padding: '10px 0', borderBottom: '1px solid var(--border)' }}>
                <code>{eq.equation}</code>
                <p style={{ fontSize: 13, marginTop: 6 }}>{eq.intuition}</p>
              </div>
            ))}
          </div>
          <div className="card">
            <div className="card-title">Extension Engine</div>
            <div className="grid grid-2">
              <div>
                <strong>What&apos;s Next</strong>
                <ul style={{ margin: '8px 0 0 20px', fontSize: 13 }}>{(distill.extension_engine?.what_next || []).map((w: string, i: number) => <li key={i}>{w}</li>)}</ul>
              </div>
              <div>
                <strong>Missing Pieces</strong>
                <ul style={{ margin: '8px 0 0 20px', fontSize: 13 }}>{(distill.extension_engine?.missing_piece_detector || []).map((m: string, i: number) => <li key={i}>{m}</li>)}</ul>
              </div>
            </div>
          </div>
          {(distill.quiz_cards || []).length > 0 && (
            <div className="card">
              <div className="card-title">Quiz Cards</div>
              {distill.quiz_cards.map((q: any, i: number) => (
                <div key={i} style={{ padding: '10px 0', borderBottom: '1px solid var(--border)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <strong style={{ fontSize: 13 }}>{q.question}</strong>
                    <span className={`badge badge-${q.difficulty === 'beginner' ? 'green' : q.difficulty === 'intermediate' ? 'blue' : 'purple'}`}>{q.difficulty}</span>
                  </div>
                  <p style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 6 }}>{q.answer}</p>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {tab === 'summaries' && distill?.summaries && (
        <>
          {Object.entries(distill.summaries).map(([level, text]) => (
            <div key={level} className="card">
              <div className="card-header">
                <div className="card-title">{level === 'eli5' ? 'ELI5 (Simple)' : level === 'practitioner' ? 'Practitioner Level' : 'Researcher Level'}</div>
                <span className={`badge badge-${level === 'eli5' ? 'green' : level === 'practitioner' ? 'blue' : 'purple'}`}>{level}</span>
              </div>
              <div style={{ whiteSpace: 'pre-wrap', fontSize: 14, lineHeight: 1.6 }}>{text as string}</div>
            </div>
          ))}
        </>
      )}
    </div>
  );
}

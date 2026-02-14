'use client';

import { useState } from 'react';
import { api } from '@/lib/api';
import { AgentTrace } from './AgentTrace';
import { ReproducibilityCard } from './ReproducibilityCard';

interface Props {
  onProjectCreated: (project: any) => void;
}

export function ProjectIntake({ onProjectCreated }: Props) {
  const [title, setTitle] = useState('Efficient Distilled Transformers');
  const [abstract, setAbstract] = useState('We propose an efficient transformer compression method combining knowledge distillation with structured pruning for low-latency inference while preserving performance on production NLP tasks. Our approach achieves state-of-the-art accuracy-latency trade-offs.');
  const [method, setMethod] = useState('Our method defines L = alpha * KL(student, teacher) + beta * CE(student, labels). We apply iterative magnitude pruning with quantization-aware training. The student architecture uses 6 transformer layers with 256 hidden dim and 4 attention heads. We train for 50 epochs with batch_size 64 using Adam optimizer with learning_rate 1e-4. We evaluate on CIFAR-10 and SQuAD, reporting accuracy, F1, and latency metrics.');
  const [framework, setFramework] = useState('pytorch');
  const [persona, setPersona] = useState('ml_engineer');
  const [sourceUrl, setSourceUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState('');

  const handleSubmit = async () => {
    setLoading(true);
    setError('');
    try {
      const project = await api.ingest({
        title, abstract, method_text: method, framework, persona,
        source_url: sourceUrl || undefined,
      });
      setResult(project);
      onProjectCreated(project);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Project Intake</div>
        <div className="page-desc">Upload a paper or paste details to start the multi-agent pipeline.</div>
      </div>

      <div className="card">
        <div className="card-title" style={{ marginBottom: 16 }}>Paper Details</div>

        <div className="form-group">
          <label className="form-label">Title</label>
          <input className="form-input" value={title} onChange={e => setTitle(e.target.value)} placeholder="Paper title..." />
        </div>

        <div className="form-group">
          <label className="form-label">Abstract (min 40 chars)</label>
          <textarea className="form-textarea" value={abstract} onChange={e => setAbstract(e.target.value)} rows={3} />
        </div>

        <div className="form-group">
          <label className="form-label">Method Text (min 80 chars)</label>
          <textarea className="form-textarea" value={method} onChange={e => setMethod(e.target.value)} rows={4} />
        </div>

        <div className="grid grid-3">
          <div className="form-group">
            <label className="form-label">Framework</label>
            <select className="form-select" value={framework} onChange={e => setFramework(e.target.value)}>
              <option value="pytorch">PyTorch</option>
              <option value="tensorflow">TensorFlow</option>
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Persona</label>
            <select className="form-select" value={persona} onChange={e => setPersona(e.target.value)}>
              <option value="ml_engineer">ML Engineer</option>
              <option value="founder_pm">Founder / PM</option>
              <option value="designer_educator">Designer / Educator</option>
              <option value="mobile_dev">Mobile Developer</option>
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Source URL (optional)</label>
            <input className="form-input" value={sourceUrl} onChange={e => setSourceUrl(e.target.value)} placeholder="https://arxiv.org/abs/..." />
          </div>
        </div>

        <button className="btn btn-primary" onClick={handleSubmit} disabled={loading}>
          {loading ? <><span className="spinner" /> Running Pipeline...</> : 'Run Multi-Agent Pipeline'}
        </button>

        {error && <p style={{ color: 'var(--red)', marginTop: 12, fontSize: 14 }}>Error: {error}</p>}
      </div>

      {result && <ProjectResult project={result} />}
    </div>
  );
}

function ProjectResult({ project }: { project: any }) {
  const spec = project.paper_spec || {};
  const repro = project.reproducibility || {};
  const msgs = project.agent_messages || [];
  const scaffold = project.code_scaffold || {};
  const confColor = project.confidence_score >= 0.7 ? 'var(--green)' : project.confidence_score >= 0.4 ? 'var(--orange)' : 'var(--red)';

  return (
    <>
      <div className="grid grid-4" style={{ marginBottom: 16 }}>
        <div className="card stat-card">
          <div className="stat-value" style={{ color: confColor }}>{(project.confidence_score * 100).toFixed(0)}%</div>
          <div className="stat-label">Overall Confidence</div>
        </div>
        <div className="card stat-card">
          <div className="stat-value" style={{ color: (repro.overall || 0) >= 0.5 ? 'var(--green)' : 'var(--red)' }}>
            {((repro.overall || 0) * 100).toFixed(0)}%
          </div>
          <div className="stat-label">Reproducibility</div>
        </div>
        <div className="card stat-card">
          <div className="stat-value">{(scaffold.files || []).length}</div>
          <div className="stat-label">Generated Files</div>
        </div>
        <div className="card stat-card">
          <div className="stat-value">{msgs.length}</div>
          <div className="stat-label">Agent Messages</div>
        </div>
      </div>

      <div className="card">
        <div className="card-title">Paper Specification</div>
        <table>
          <tbody>
            <tr><th style={{ width: 160 }}>Problem</th><td>{spec.problem}</td></tr>
            <tr><th>Method</th><td>{spec.method}</td></tr>
            <tr><th>Datasets</th><td>{(spec.datasets || []).map((d: string) => <span key={d} className="badge badge-blue" style={{ marginRight: 4 }}>{d}</span>)}</td></tr>
            <tr><th>Metrics</th><td>{(spec.metrics || []).map((m: string) => <span key={m} className="badge badge-green" style={{ marginRight: 4 }}>{m}</span>)}</td></tr>
            <tr><th>Architecture</th><td>{(spec.architecture_components || []).map((a: string) => <span key={a} className="badge badge-purple" style={{ marginRight: 4 }}>{a}</span>)}</td></tr>
            <tr><th>Equations</th><td>{(spec.key_equations || []).map((eq: any, i: number) => <div key={i}><code>{eq.raw || eq}</code></div>)}</td></tr>
          </tbody>
        </table>
      </div>

      <ReproducibilityCard repro={repro} />
      <AgentTrace messages={msgs} />

      <div style={{ marginTop: 12, display: 'flex', gap: 12 }}>
        <a className="btn btn-success" href={api.exportZip(project.id)} download>Download ZIP Export</a>
      </div>
    </>
  );
}

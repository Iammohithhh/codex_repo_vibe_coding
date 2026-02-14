'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';

interface Props { project: any; }

export function TeamReview({ project }: Props) {
  const [reviews, setReviews] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const load = async () => {
    if (!project?.id) return;
    setLoading(true);
    try { const r = await api.getReviews(project.id); setReviews(r); } finally { setLoading(false); }
  };

  useEffect(() => { load(); }, [project?.id]);

  if (!project) return <div className="card"><p style={{ color: 'var(--text-dim)' }}>Ingest a paper first to start reviews.</p></div>;
  if (loading) return <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: 40, color: 'var(--text-dim)' }}><span className="spinner" /> Loading...</div>;

  const handleAddReview = async () => {
    const content = prompt('Review comment:');
    if (!content) return;
    const author = prompt('Your name:', 'reviewer') || 'reviewer';
    await api.addReview(project.id, { author, content });
    load();
  };

  const handleApprove = async () => {
    const approver = prompt('Approver name:', 'admin');
    if (!approver) return;
    const result = await api.approve(project.id, approver);
    if (result.error) alert(result.error);
    else alert('Project approved!');
  };

  return (
    <div>
      <div className="page-header"><div className="page-title">Team Review</div><div className="page-desc">Comments, approvals, and governance workflows.</div></div>

      <div className="card">
        <div className="card-header">
          <div className="card-title">Reviews</div>
          <button className="btn btn-sm btn-primary" onClick={handleAddReview}>Add Review</button>
        </div>
        {(reviews?.reviews || []).length === 0 ? (
          <p style={{ color: 'var(--text-dim)', fontSize: 13 }}>No reviews yet.</p>
        ) : (
          (reviews?.reviews || []).map((r: any, i: number) => (
            <div key={i} style={{ padding: '12px 0', borderBottom: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <strong style={{ fontSize: 13 }}>{r.author || 'anonymous'}</strong>
                <span className={`badge badge-${r.status === 'approved' ? 'green' : r.status === 'rejected' ? 'red' : 'orange'}`}>{r.status}</span>
              </div>
              <p style={{ fontSize: 13 }}>{r.content}</p>
              {r.artifact_path && <div style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 4 }}>File: {r.artifact_path}{r.line_number ? `:${r.line_number}` : ''}</div>}
            </div>
          ))
        )}
      </div>

      <div className="card">
        <div className="card-title">Approval</div>
        <p style={{ fontSize: 13, color: 'var(--text-dim)', marginBottom: 12 }}>Approve this project for production export.</p>
        <button className="btn btn-success" onClick={handleApprove}>Approve for Production</button>
      </div>
    </div>
  );
}

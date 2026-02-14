'use client';

interface Props {
  repro: any;
}

export function ReproducibilityCard({ repro }: Props) {
  if (!repro || repro.overall === undefined) return null;

  const fields = ['code_available', 'data_available', 'hyperparams_complete', 'compute_specified'];

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">Reproducibility Scorecard</div>
        <span className={`badge ${repro.overall >= 0.7 ? 'badge-green' : repro.overall >= 0.4 ? 'badge-orange' : 'badge-red'}`}>
          {(repro.overall * 100).toFixed(0)}%
        </span>
      </div>

      <div className="grid grid-4" style={{ marginBottom: 12 }}>
        {fields.map(k => (
          <div key={k}>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width: `${(repro[k] || 0) * 100}%`,
                  background: (repro[k] || 0) >= 0.5 ? 'var(--green)' : 'var(--orange)',
                }}
              />
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 4 }}>
              {k.replace(/_/g, ' ')} — {((repro[k] || 0) * 100).toFixed(0)}%
            </div>
          </div>
        ))}
      </div>

      {(repro.blockers || []).length > 0 && (
        <div style={{ marginTop: 8 }}>
          <strong style={{ color: 'var(--red)' }}>Blockers:</strong>
          <ul style={{ margin: '4px 0 0 20px' }}>
            {repro.blockers.map((b: string, i: number) => <li key={i} style={{ fontSize: 13 }}>{b}</li>)}
          </ul>
        </div>
      )}

      {(repro.fixes || []).length > 0 && (
        <div style={{ marginTop: 8 }}>
          <strong style={{ color: 'var(--green)' }}>Suggested Fixes:</strong>
          <ul style={{ margin: '4px 0 0 20px' }}>
            {repro.fixes.map((f: string, i: number) => <li key={i} style={{ fontSize: 13 }}>{f}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}

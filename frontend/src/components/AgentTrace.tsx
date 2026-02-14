'use client';

interface Props {
  messages: any[];
}

export function AgentTrace({ messages }: Props) {
  if (!messages.length) return null;

  return (
    <div className="card">
      <div className="card-title">Agent Pipeline Trace</div>
      <div style={{ fontSize: 13, color: 'var(--text-dim)', marginBottom: 12 }}>
        {messages.length} messages from 5 agents
      </div>
      {messages.map((m: any, i: number) => (
        <div key={i} className={`agent-msg ${m.role || 'unknown'}`}>
          <div className="agent-role">
            {m.role}
            <span className="badge badge-blue" style={{ marginLeft: 8 }}>
              {((m.confidence || 1) * 100).toFixed(0)}%
            </span>
          </div>
          {m.content}
        </div>
      ))}
    </div>
  );
}

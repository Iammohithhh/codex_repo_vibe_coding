'use client';

interface SidebarProps {
  currentPage: string;
  onNavigate: (page: string) => void;
  projects: any[];
  onSelectProject: (p: any) => void;
  currentProject: any;
}

const navItems = [
  { id: 'intake', icon: '+', label: 'Project Intake' },
  { id: 'studio', icon: '\u2699', label: 'Artifact Studio' },
  { id: 'lab', icon: '\u2697', label: 'Experiment Lab' },
  { id: 'api', icon: '\u276F', label: 'API Builder' },
  { id: 'launch', icon: '\u25B2', label: 'Launch Center' },
  { id: 'review', icon: '\u2713', label: 'Team Review' },
];

export function Sidebar({ currentPage, onNavigate, projects, onSelectProject, currentProject }: SidebarProps) {
  return (
    <div className="sidebar">
      <div className="logo">Paper2Product</div>
      <div className="logo-sub">AI Research OS v2.0</div>

      <div className="nav-section">Modules</div>
      {navItems.map(item => (
        <button
          key={item.id}
          className={`nav-item ${currentPage === item.id ? 'active' : ''}`}
          onClick={() => onNavigate(item.id)}
        >
          <span style={{ width: 18, textAlign: 'center' }}>{item.icon}</span>
          {item.label}
        </button>
      ))}

      <div className="nav-section">Projects</div>
      <div>
        {projects.length === 0 ? (
          <div style={{ fontSize: 12, color: 'var(--text-dim)', padding: '6px 12px' }}>No projects yet</div>
        ) : (
          projects.map((p, i) => (
            <div
              key={p.id || i}
              className="file-item"
              style={{
                color: currentProject?.id === p.id ? 'var(--accent-light)' : undefined,
                fontWeight: currentProject?.id === p.id ? 500 : undefined,
              }}
              onClick={() => onSelectProject(p)}
            >
              {p.ingest?.title || p.id?.slice(0, 8)}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

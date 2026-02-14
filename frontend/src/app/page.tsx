'use client';

import { useState } from 'react';
import { Sidebar } from '@/components/Sidebar';
import { ProjectIntake } from '@/components/ProjectIntake';
import { ArtifactStudio } from '@/components/ArtifactStudio';
import { ExperimentLab } from '@/components/ExperimentLab';
import { ApiBuilder } from '@/components/ApiBuilder';
import { LaunchCenter } from '@/components/LaunchCenter';
import { TeamReview } from '@/components/TeamReview';

export default function Home() {
  const [page, setPage] = useState('intake');
  const [project, setProject] = useState<any>(null);
  const [projects, setProjects] = useState<any[]>([]);

  const handleProjectCreated = (p: any) => {
    setProject(p);
    setProjects(prev => [p, ...prev]);
  };

  const pages: Record<string, React.ReactNode> = {
    intake: <ProjectIntake onProjectCreated={handleProjectCreated} />,
    studio: <ArtifactStudio project={project} />,
    lab: <ExperimentLab project={project} />,
    api: <ApiBuilder project={project} />,
    launch: <LaunchCenter project={project} />,
    review: <TeamReview project={project} />,
  };

  return (
    <div className="app">
      <Sidebar
        currentPage={page}
        onNavigate={setPage}
        projects={projects}
        onSelectProject={(p) => setProject(p)}
        currentProject={project}
      />
      <main className="main">
        {pages[page]}
      </main>
    </div>
  );
}

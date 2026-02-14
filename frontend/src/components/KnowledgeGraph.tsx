'use client';

import { useCallback, useMemo } from 'react';
import ReactFlow, { Background, Controls, MiniMap, Node, Edge } from 'reactflow';
import 'reactflow/dist/style.css';

interface Props {
  graphData: { nodes?: any[]; edges?: any[] };
}

const typeColors: Record<string, string> = {
  problem: '#ef4444',
  method: '#3b82f6',
  equation: '#a855f7',
  dataset: '#22c55e',
  metric: '#22c55e',
  architecture: '#f59e0b',
  claim: '#6366f1',
  default: '#8b8fa3',
};

export function KnowledgeGraph({ graphData }: Props) {
  const nodes: Node[] = useMemo(() => {
    return (graphData.nodes || []).map((n: any, i: number) => ({
      id: n.id || `node-${i}`,
      data: { label: n.label || n.id },
      position: { x: 150 + (i % 4) * 250, y: 50 + Math.floor(i / 4) * 120 },
      style: {
        background: '#1a1d27',
        color: '#e4e6f0',
        border: `2px solid ${typeColors[n.type] || typeColors.default}`,
        borderRadius: 8,
        padding: '8px 12px',
        fontSize: 12,
        width: 200,
      },
    }));
  }, [graphData.nodes]);

  const edges: Edge[] = useMemo(() => {
    return (graphData.edges || []).map((e: any, i: number) => ({
      id: `edge-${i}`,
      source: e.source,
      target: e.target,
      label: e.label,
      animated: true,
      style: { stroke: '#2d3148' },
      labelStyle: { fill: '#8b8fa3', fontSize: 10 },
    }));
  }, [graphData.edges]);

  if (!nodes.length) return <div style={{ color: 'var(--text-dim)', fontSize: 13 }}>No graph data available.</div>;

  return (
    <div style={{ height: 500 }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        attributionPosition="bottom-left"
      >
        <Background color="#2d3148" gap={20} />
        <Controls />
        <MiniMap
          style={{ background: '#0f1117' }}
          nodeColor={(n) => {
            const type = graphData.nodes?.find((gn: any) => gn.id === n.id)?.type;
            return typeColors[type] || typeColors.default;
          }}
        />
      </ReactFlow>
    </div>
  );
}

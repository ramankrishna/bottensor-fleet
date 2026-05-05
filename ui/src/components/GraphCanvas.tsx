import { useCallback, useMemo } from 'react'
import ReactFlow, {
  Background,
  Controls,
  Handle,
  Position,
  type NodeProps,
  type Node,
  type Edge,
} from 'reactflow'
import 'reactflow/dist/style.css'
import { useFleetStore, type NodeStatus } from '../store/fleetStore'

// ── Status badge ─────────────────────────────────────────────────────────────

const STATUS_COLOR: Record<NodeStatus, string> = {
  idle:    '#8a94a6',
  running: '#00dbb8',
  done:    '#3ecf8e',
  error:   '#f87171',
  waiting: '#f5c842',
}

const STATUS_BG: Record<NodeStatus, string> = {
  idle:    'rgba(138,148,166,0.12)',
  running: 'rgba(0,219,184,0.12)',
  done:    'rgba(62,207,142,0.12)',
  error:   'rgba(248,113,113,0.12)',
  waiting: 'rgba(245,200,66,0.12)',
}

function StatusBadge({ status }: { status: NodeStatus }) {
  return (
    <span
      style={{
        fontFamily: 'var(--font-mono)',
        fontSize: 10,
        color: STATUS_COLOR[status],
        background: STATUS_BG[status],
        border: `1px solid ${STATUS_COLOR[status]}33`,
        borderRadius: 4,
        padding: '1px 6px',
        letterSpacing: '0.04em',
      }}
    >
      {status}
    </span>
  )
}

// ── Custom AgentNode ──────────────────────────────────────────────────────────

function AgentNode({ data }: NodeProps) {
  const status: NodeStatus = data.status ?? 'idle'
  const isActive  = status === 'running'
  const isWaiting = status === 'waiting'

  const borderColor = isWaiting
    ? 'var(--accent-gold)'
    : isActive
      ? 'var(--accent-teal)'
      : 'var(--border)'

  return (
    <div
      style={{
        background: 'var(--bg-elevated)',
        border: `1.5px solid ${borderColor}`,
        borderRadius: 'var(--radius-md)',
        padding: '10px 14px',
        minWidth: 140,
        boxShadow: isActive
          ? '0 0 12px rgba(0,219,184,0.25)'
          : isWaiting
            ? '0 0 12px rgba(245,200,66,0.20)'
            : 'none',
        transition: 'border-color 0.2s, box-shadow 0.2s',
        cursor: 'pointer',
      }}
      onClick={() => data.onSelect(data.name)}
    >
      <Handle type="target" position={Position.Top} style={{ background: 'var(--border)' }} />

      <div style={{ fontFamily: 'var(--font-display)', fontWeight: 500, fontSize: 13, color: 'var(--text-primary)', marginBottom: 6 }}>
        {data.label}
      </div>
      <StatusBadge status={status} />

      <Handle type="source" position={Position.Bottom} style={{ background: 'var(--border)' }} />
    </div>
  )
}

const NODE_TYPES = { agent: AgentNode }

// ── GraphCanvas ───────────────────────────────────────────────────────────────

export function GraphCanvas() {
  const nodes    = useFleetStore((s) => s.nodes)
  const setSelected = useFleetStore((s) => s.setSelectedNode)

  const onSelect = useCallback((name: string) => setSelected(name), [setSelected])

  // Lay nodes out in a simple vertical grid when there's no position info
  const rfNodes: Node[] = useMemo(() => {
    const entries = Object.values(nodes)
    return entries.map((n, i) => ({
      id: n.name,
      type: 'agent',
      position: { x: 120 + (i % 3) * 200, y: 80 + Math.floor(i / 3) * 130 },
      data: { name: n.name, label: n.name, status: n.status, onSelect },
    }))
  }, [nodes, onSelect])

  const rfEdges: Edge[] = useMemo(() => {
    const entries = Object.values(nodes)
    if (entries.length < 2) return []
    return entries.slice(1).map((n, i) => ({
      id: `e-${entries[i].name}-${n.name}`,
      source: entries[i].name,
      target: n.name,
      style: { stroke: 'var(--border)', strokeWidth: 1.5 },
    }))
  }, [nodes])

  return (
    <div style={{ width: '100%', height: '100%', background: 'var(--bg-base)' }}>
      <ReactFlow
        nodes={rfNodes}
        edges={rfEdges}
        nodeTypes={NODE_TYPES}
        fitView
        proOptions={{ hideAttribution: true }}
        style={{ background: 'var(--bg-base)' }}
      >
        <Background color="var(--border)" gap={28} size={1} />
        <Controls
          style={{
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
          }}
        />
      </ReactFlow>

      {rfNodes.length === 0 && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--text-muted)',
            fontFamily: 'var(--font-display)',
            fontSize: 14,
            pointerEvents: 'none',
          }}
        >
          No agents running — start a run to see the graph.
        </div>
      )}
    </div>
  )
}

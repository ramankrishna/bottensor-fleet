/**
 * Bidirectional mapping between ReactFlow canvas state and the GraphSpec
 * JSON format the Python loader accepts.
 *
 * The shapes match `src/fleet/graphspec/spec.py` 1:1 — keep them in sync.
 */
import type { Node as RFNode, Edge as RFEdge } from 'reactflow'

// ── GraphSpec wire types ───────────────────────────────────────────────────

export interface AgentSpecJSON {
  name: string
  provider: string
  model: string
  system: string
  tools: string[]
  memory_bank: boolean
  base_url: string | null
  max_iters?: number
}

export interface PositionJSON {
  x: number
  y: number
}

export interface NodeSpecJSON {
  id: string
  type: 'agent'
  agent: AgentSpecJSON
  position?: PositionJSON | null
}

export interface EdgeSpecJSON {
  src: string
  dst: string
  cond?: string | null
}

export interface GraphSpecJSON {
  version: string
  name: string
  nodes: NodeSpecJSON[]
  edges: EdgeSpecJSON[]
  entry: string
  exit: string
}

// ── Canvas-side node data ──────────────────────────────────────────────────

export interface AgentNodeData {
  agent: AgentSpecJSON
  isEntry: boolean
  isExit: boolean
}

export interface AgentEdgeData {
  cond: string | null
}

export type BuilderNode = RFNode<AgentNodeData>
export type BuilderEdge = RFEdge<AgentEdgeData>

// ── canvasToSpec ───────────────────────────────────────────────────────────

export interface CanvasState {
  graphName: string
  nodes: BuilderNode[]
  edges: BuilderEdge[]
}

export interface CanvasConversionError {
  message: string
}

export function canvasToSpec(
  canvas: CanvasState,
): { spec: GraphSpecJSON } | { error: string } {
  const { graphName, nodes, edges } = canvas

  if (!graphName.trim()) {
    return { error: 'Graph needs a name.' }
  }
  if (nodes.length === 0) {
    return { error: 'Add at least one agent node.' }
  }

  const entry = nodes.find((n) => n.data.isEntry)
  const exit = nodes.find((n) => n.data.isExit)
  if (!entry) return { error: 'Pick an entry node.' }
  if (!exit) return { error: 'Pick an exit node.' }

  const ids = new Set(nodes.map((n) => n.id))
  if (ids.size !== nodes.length) {
    return { error: 'Duplicate node ids — give each agent a unique id.' }
  }

  const specNodes: NodeSpecJSON[] = nodes.map((n) => ({
    id: n.id,
    type: 'agent',
    agent: { ...n.data.agent },
    position: n.position ? { x: n.position.x, y: n.position.y } : null,
  }))

  const specEdges: EdgeSpecJSON[] = edges.map((e) => ({
    src: e.source,
    dst: e.target,
    cond: e.data?.cond ?? null,
  }))

  // Surface dangling edges client-side so the UI doesn't have to wait for the
  // server's 422 to give feedback.
  for (const e of specEdges) {
    if (!ids.has(e.src)) return { error: `Edge from unknown node '${e.src}'.` }
    if (!ids.has(e.dst)) return { error: `Edge to unknown node '${e.dst}'.` }
  }

  return {
    spec: {
      version: '0.3',
      name: graphName.trim(),
      nodes: specNodes,
      edges: specEdges,
      entry: entry.id,
      exit: exit.id,
    },
  }
}

// ── specToCanvas ───────────────────────────────────────────────────────────

export function specToCanvas(spec: GraphSpecJSON): CanvasState {
  const nodes: BuilderNode[] = spec.nodes.map((n, i) => ({
    id: n.id,
    type: 'agent',
    position: n.position ?? { x: 120 + (i % 3) * 220, y: 80 + Math.floor(i / 3) * 140 },
    data: {
      agent: n.agent,
      isEntry: n.id === spec.entry,
      isExit: n.id === spec.exit,
    },
  }))

  const edges: BuilderEdge[] = spec.edges.map((e, i) => ({
    id: `e_${i}_${e.src}_${e.dst}`,
    source: e.src,
    target: e.dst,
    type: e.cond ? 'smoothstep' : 'default',
    label: e.cond ?? undefined,
    data: { cond: e.cond ?? null },
  }))

  return { graphName: spec.name, nodes, edges }
}

// ── Helpers used by the side panel ─────────────────────────────────────────

export function newAgentSpec(id: string): AgentSpecJSON {
  return {
    name: id,
    provider: 'anthropic',
    model: 'claude-sonnet-4-6',
    system: '',
    tools: [],
    memory_bank: false,
    base_url: null,
    max_iters: 10,
  }
}

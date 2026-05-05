import { create } from 'zustand'

// ── event shapes emitted by the server ──────────────────────────────────────
export interface FleetEvent {
  type: string
  run_id?: string
  node?: string
  step?: number
  status?: string
  error?: string
  [key: string]: unknown
}

// ── per-agent / per-node state ───────────────────────────────────────────────
export type NodeStatus = 'idle' | 'running' | 'done' | 'error' | 'waiting'

export interface NodeState {
  name: string
  status: NodeStatus
  lastStep?: number
  events: FleetEvent[]
}

// ── message log entry ────────────────────────────────────────────────────────
export interface LogEntry {
  id: string
  ts: number
  role: 'user' | 'assistant' | 'tool' | 'system'
  content: string
  toolName?: string
  isError?: boolean
}

// ── provider keys ────────────────────────────────────────────────────────────
export interface ProviderKey {
  provider: string
  key: string
}

// ── run meta ─────────────────────────────────────────────────────────────────
export interface RunMeta {
  runId: string
  goal: string
  graphModule: string
  status: 'running' | 'paused' | 'done' | 'killed' | 'error'
  messageCount: number
}

// ── store shape ──────────────────────────────────────────────────────────────
export interface FleetStore {
  // Discovery
  providers: string[]
  setProviders: (p: string[]) => void

  graphs: string[]
  setGraphs: (g: string[]) => void

  // Provider keys (in-memory only, never persisted)
  providerKeys: ProviderKey[]
  setKey: (provider: string, key: string) => void
  clearKeys: () => void

  // Active run
  activeRunId: string | null
  setActiveRunId: (id: string | null) => void

  runs: Record<string, RunMeta>
  upsertRun: (run: RunMeta) => void

  // Per-node state (keyed by node name)
  nodes: Record<string, NodeState>
  upsertNode: (name: string, patch: Partial<NodeState>) => void
  resetNodes: () => void

  // Selected node (for AgentPanel)
  selectedNode: string | null
  setSelectedNode: (name: string | null) => void

  // Log entries per run
  logs: Record<string, LogEntry[]>
  appendLog: (runId: string, entry: LogEntry) => void
  clearLogs: (runId: string) => void

  // Dispatch a raw WS event into the store
  dispatchEvent: (event: FleetEvent) => void
}

let _logCounter = 0
function nextId() { return `log_${++_logCounter}` }

export const useFleetStore = create<FleetStore>((set, get) => ({
  providers: [],
  setProviders: (p) => set({ providers: p }),

  graphs: [],
  setGraphs: (g) => set({ graphs: g }),

  providerKeys: [],
  setKey: (provider, key) => set((s) => ({
    providerKeys: [
      ...s.providerKeys.filter((k) => k.provider !== provider),
      { provider, key },
    ],
  })),
  clearKeys: () => set({ providerKeys: [] }),

  activeRunId: null,
  setActiveRunId: (id) => set({ activeRunId: id }),

  runs: {},
  upsertRun: (run) => set((s) => ({ runs: { ...s.runs, [run.runId]: run } })),

  nodes: {},
  upsertNode: (name, patch) => set((s) => {
    const existing = s.nodes[name] ?? { name, status: 'idle' as NodeStatus, events: [] }
    return { nodes: { ...s.nodes, [name]: { ...existing, ...patch } } }
  }),
  resetNodes: () => set({ nodes: {} }),

  selectedNode: null,
  setSelectedNode: (name) => set({ selectedNode: name }),

  logs: {},
  appendLog: (runId, entry) => set((s) => ({
    logs: { ...s.logs, [runId]: [...(s.logs[runId] ?? []), entry] },
  })),
  clearLogs: (runId) => set((s) => ({ logs: { ...s.logs, [runId]: [] } })),

  dispatchEvent: (event) => {
    const { upsertNode, appendLog } = get()
    const runId = event.run_id ?? 'unknown'

    switch (event.type) {
      case 'node_started':
        upsertNode(event.node ?? '', { status: 'running' })
        break
      case 'node_finished':
        upsertNode(event.node ?? '', {
          status: 'done',
          lastStep: event.step as number | undefined,
        })
        break
      case 'tool_called':
        upsertNode(event.node ?? '', { status: 'waiting' })
        appendLog(runId, {
          id: nextId(), ts: Date.now(), role: 'tool',
          content: `→ ${event.tool_name ?? '(tool)'}`,
          toolName: event.tool_name as string | undefined,
        })
        break
      case 'tool_returned':
        upsertNode(event.node ?? '', { status: 'running' })
        appendLog(runId, {
          id: nextId(), ts: Date.now(), role: 'tool',
          content: `← ${event.tool_name ?? '(tool)'}: ${String(event.result ?? '').slice(0, 120)}`,
          toolName: event.tool_name as string | undefined,
        })
        break
      case 'run_finished':
        set((s) => {
          const existing = s.runs[runId]
          if (!existing) return {}
          return {
            runs: {
              ...s.runs,
              [runId]: { ...existing, status: (event.status as RunMeta['status']) ?? 'done' },
            },
          }
        })
        break
    }
  },
}))

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ReactFlow, {
  Background,
  Controls,
  Handle,
  Position,
  ReactFlowProvider,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  useReactFlow,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
  type NodeProps,
} from 'reactflow'
import 'reactflow/dist/style.css'

import { useFleetStore, type NodeStatus } from '../store/fleetStore'
import {
  canvasToSpec,
  newAgentSpec,
  specToCanvas,
  type AgentEdgeData,
  type AgentNodeData,
  type BuilderEdge,
  type BuilderNode,
  type GraphSpecJSON,
} from '../lib/graphSpec'

// ---------------------------------------------------------------------------
// Custom node
// ---------------------------------------------------------------------------

const STATUS_COLOR: Record<NodeStatus, string> = {
  idle: '#8a94a6',
  running: '#00dbb8',
  done: '#3ecf8e',
  error: '#f87171',
  waiting: '#f5c842',
}

interface AgentNodeRenderData extends AgentNodeData {
  liveStatus?: NodeStatus
  onSelect: (id: string) => void
  selected: boolean
}

function BuilderAgentNode({ id, data }: NodeProps<AgentNodeRenderData>) {
  const status = data.liveStatus ?? 'idle'
  const isSelected = data.selected
  return (
    <div
      onClick={() => data.onSelect(id)}
      style={{
        background: 'var(--bg-elevated)',
        border: `1.5px solid ${isSelected ? 'var(--accent-teal)' : 'var(--border)'}`,
        borderRadius: 'var(--radius-md)',
        padding: '10px 14px',
        minWidth: 150,
        boxShadow: isSelected ? '0 0 12px rgba(0,219,184,0.25)' : 'none',
        cursor: 'pointer',
      }}
    >
      <Handle type="target" position={Position.Top} style={{ background: 'var(--border)' }} />
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          marginBottom: 4,
        }}
      >
        <span
          style={{
            fontFamily: 'var(--font-display)',
            fontWeight: 500,
            fontSize: 13,
            color: 'var(--text-primary)',
          }}
        >
          {data.agent.name || id}
        </span>
        {data.isEntry && <Pill label="entry" color="var(--accent-teal)" />}
        {data.isExit && <Pill label="exit" color="var(--accent-gold)" />}
      </div>
      <div
        style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 10,
          color: 'var(--text-muted)',
        }}
      >
        {data.agent.provider}/{data.agent.model}
      </div>
      <div style={{ marginTop: 4 }}>
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: STATUS_COLOR[status],
          }}
        >
          ● {status}
        </span>
      </div>
      <Handle type="source" position={Position.Bottom} style={{ background: 'var(--border)' }} />
    </div>
  )
}

function Pill({ label, color }: { label: string; color: string }) {
  return (
    <span
      style={{
        fontFamily: 'var(--font-mono)',
        fontSize: 9,
        color,
        background: `${color}22`,
        border: `1px solid ${color}55`,
        borderRadius: 4,
        padding: '0 4px',
        letterSpacing: '0.04em',
      }}
    >
      {label}
    </span>
  )
}

const NODE_TYPES = { agent: BuilderAgentNode }

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

interface ProviderDetail {
  name: string
  requires_base_url: boolean
  default_base_url: string | null
}

function GraphBuilderInner() {
  const [graphName, setGraphName] = useState('untitled_graph')
  const [nodes, setNodes] = useState<BuilderNode[]>([])
  const [edges, setEdges] = useState<BuilderEdge[]>([])
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [goal, setGoal] = useState('')

  const [providers, setProviders] = useState<ProviderDetail[]>([])
  const [conditions, setConditions] = useState<string[]>([])
  const [parametrics, setParametrics] = useState<string[]>([])
  const [tools, setTools] = useState<string[]>([])

  const setActiveRunId = useFleetStore((s) => s.setActiveRunId)
  const upsertRun = useFleetStore((s) => s.upsertRun)
  const resetNodes = useFleetStore((s) => s.resetNodes)
  const liveNodes = useFleetStore((s) => s.nodes)

  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const counterRef = useRef(0)
  const { screenToFlowPosition } = useReactFlow()

  // ─── Load registries once ─────────────────────────────────────────────
  useEffect(() => {
    fetch('/api/providers')
      .then((r) => r.json())
      .then((data) => {
        const list = (data.details as ProviderDetail[] | undefined) ??
          (data.providers as string[] | undefined)?.map((name) => ({
            name,
            requires_base_url: name === 'custom',
            default_base_url: null,
          })) ?? []
        setProviders(list)
      })
      .catch(() => {})

    fetch('/api/conditions')
      .then((r) => r.json())
      .then((data) => {
        setConditions(data.conditions ?? [])
        setParametrics(data.parametric ?? [])
      })
      .catch(() => {})

    fetch('/api/tools')
      .then((r) => r.json())
      .then((data) => setTools(data.tools ?? []))
      .catch(() => {})
  }, [])

  // ─── ReactFlow change handlers ────────────────────────────────────────
  const onNodesChange = useCallback(
    (changes: NodeChange[]) => setNodes((n) => applyNodeChanges(changes, n) as BuilderNode[]),
    [],
  )
  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => setEdges((e) => applyEdgeChanges(changes, e) as BuilderEdge[]),
    [],
  )
  const onConnect = useCallback(
    (params: Connection) =>
      setEdges((e) =>
        addEdge({ ...params, data: { cond: null } as AgentEdgeData }, e) as BuilderEdge[],
      ),
    [],
  )

  // ─── Palette: drag handlers ───────────────────────────────────────────
  const onDragStart = (event: React.DragEvent<HTMLDivElement>) => {
    event.dataTransfer.setData('application/fleet-node', 'agent')
    event.dataTransfer.effectAllowed = 'move'
  }

  const onDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault()
      const type = event.dataTransfer.getData('application/fleet-node')
      if (type !== 'agent') return
      const position = screenToFlowPosition({ x: event.clientX, y: event.clientY })
      const id = nextNodeId(counterRef, nodes)
      const isFirst = nodes.length === 0
      const newNode: BuilderNode = {
        id,
        type: 'agent',
        position,
        data: { agent: newAgentSpec(id), isEntry: isFirst, isExit: isFirst },
      }
      setNodes((n) => [...n, newNode])
      setSelectedNodeId(id)
    },
    [screenToFlowPosition, nodes],
  )

  // ─── Selection / clicks ───────────────────────────────────────────────
  const handleSelectNode = useCallback((nodeId: string) => {
    setSelectedNodeId(nodeId)
    setSelectedEdgeId(null)
  }, [])

  // ─── Editing helpers ──────────────────────────────────────────────────
  const updateSelectedNode = useCallback(
    (patch: Partial<AgentNodeData>) => {
      if (!selectedNodeId) return
      setNodes((nlist) =>
        nlist.map((n) =>
          n.id === selectedNodeId
            ? { ...n, data: { ...n.data, ...patch, agent: { ...n.data.agent, ...(patch.agent ?? {}) } } }
            : n,
        ),
      )
    },
    [selectedNodeId],
  )

  const setEntryNode = useCallback((nodeId: string) => {
    setNodes((nlist) => nlist.map((n) => ({ ...n, data: { ...n.data, isEntry: n.id === nodeId } })))
  }, [])

  const setExitNode = useCallback((nodeId: string) => {
    setNodes((nlist) => nlist.map((n) => ({ ...n, data: { ...n.data, isExit: n.id === nodeId } })))
  }, [])

  const deleteSelectedNode = useCallback(() => {
    if (!selectedNodeId) return
    setNodes((nlist) => nlist.filter((n) => n.id !== selectedNodeId))
    setEdges((elist) =>
      elist.filter((e) => e.source !== selectedNodeId && e.target !== selectedNodeId),
    )
    setSelectedNodeId(null)
  }, [selectedNodeId])

  const updateSelectedEdgeCond = useCallback(
    (cond: string | null) => {
      if (!selectedEdgeId) return
      setEdges((elist) =>
        elist.map((e) =>
          e.id === selectedEdgeId
            ? {
                ...e,
                data: { cond },
                label: cond ?? undefined,
                type: cond ? 'smoothstep' : 'default',
              }
            : e,
        ),
      )
    },
    [selectedEdgeId],
  )

  const deleteSelectedEdge = useCallback(() => {
    if (!selectedEdgeId) return
    setEdges((elist) => elist.filter((e) => e.id !== selectedEdgeId))
    setSelectedEdgeId(null)
  }, [selectedEdgeId])

  // ─── Serialization ────────────────────────────────────────────────────
  const buildSpec = useCallback(() => {
    return canvasToSpec({ graphName, nodes, edges })
  }, [graphName, nodes, edges])

  // ─── Live-status overlay (drive node colors from WS events) ───────────
  const decoratedNodes = useMemo<Node[]>(() => {
    return nodes.map((n) => ({
      ...n,
      data: {
        ...n.data,
        liveStatus: liveNodes[n.id]?.status,
        onSelect: handleSelectNode,
        selected: n.id === selectedNodeId,
      },
    }))
  }, [nodes, liveNodes, handleSelectNode, selectedNodeId])

  const decoratedEdges = useMemo<Edge[]>(() => {
    return edges.map((e) => ({
      ...e,
      style: {
        stroke: e.id === selectedEdgeId ? 'var(--accent-teal)' : 'var(--border)',
        strokeWidth: e.id === selectedEdgeId ? 2 : 1.5,
      },
    }))
  }, [edges, selectedEdgeId])

  // ─── Run / export / save / open ───────────────────────────────────────
  const handleRun = useCallback(async () => {
    const result = buildSpec()
    if ('error' in result) {
      setError(result.error)
      return
    }
    if (!goal.trim()) {
      setError('Set a goal first.')
      return
    }
    setError(null)
    setStatus('Starting…')
    resetNodes()
    try {
      const r = await fetch('/api/runs/from-spec', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ spec: result.spec, goal: goal.trim() }),
      })
      if (!r.ok) {
        const body = await r.text()
        throw new Error(body || `HTTP ${r.status}`)
      }
      const { run_id } = (await r.json()) as { run_id: string }
      upsertRun({
        runId: run_id,
        goal: goal.trim(),
        graphModule: `builder:${graphName}`,
        status: 'running',
        messageCount: 0,
      })
      setActiveRunId(run_id)
      setStatus(`Running ${run_id}`)
    } catch (e) {
      setError(String(e))
      setStatus(null)
    }
  }, [buildSpec, goal, graphName, resetNodes, setActiveRunId, upsertRun])

  const handleExportPython = useCallback(async () => {
    const result = buildSpec()
    if ('error' in result) {
      setError(result.error)
      return
    }
    setError(null)
    try {
      const r = await fetch('/api/export-python', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ spec: result.spec }),
      })
      if (!r.ok) throw new Error(await r.text())
      const { source } = (await r.json()) as { source: string }
      downloadFile(`${graphName}.py`, source, 'text/x-python')
      setStatus(`Exported ${graphName}.py`)
    } catch (e) {
      setError(String(e))
    }
  }, [buildSpec, graphName])

  const handleSaveJSON = useCallback(() => {
    const result = buildSpec()
    if ('error' in result) {
      setError(result.error)
      return
    }
    setError(null)
    downloadFile(`${graphName}.json`, JSON.stringify(result.spec, null, 2), 'application/json')
    setStatus(`Saved ${graphName}.json`)
  }, [buildSpec, graphName])

  const handleOpenJSON = useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  const handleFileChosen = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const f = event.target.files?.[0]
      if (!f) return
      const reader = new FileReader()
      reader.onload = () => {
        try {
          const parsed = JSON.parse(String(reader.result ?? ''))
          const canvas = specToCanvas(parsed as GraphSpecJSON)
          setGraphName(canvas.graphName)
          setNodes(canvas.nodes)
          setEdges(canvas.edges)
          setSelectedNodeId(null)
          setSelectedEdgeId(null)
          setStatus(`Loaded ${f.name}`)
          setError(null)
        } catch (err) {
          setError(`Failed to parse ${f.name}: ${String(err)}`)
        }
      }
      reader.readAsText(f)
      event.target.value = ''
    },
    [],
  )

  const selectedNode = nodes.find((n) => n.id === selectedNodeId) ?? null
  const selectedEdge = edges.find((e) => e.id === selectedEdgeId) ?? null

  // ─────────────────────────────────────────────────────────────────────
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '180px 1fr 320px',
        gridTemplateRows: '48px 1fr',
        height: '100%',
        background: 'var(--bg-base)',
      }}
    >
      {/* Toolbar */}
      <div
        style={{
          gridColumn: '1 / -1',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '0 12px',
          borderBottom: '1px solid var(--border)',
          background: 'var(--bg-elevated)',
        }}
      >
        <input
          value={graphName}
          onChange={(e) => setGraphName(e.target.value)}
          style={{
            background: 'var(--bg-base)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            color: 'var(--text-primary)',
            fontFamily: 'var(--font-mono)',
            fontSize: 12,
            padding: '5px 8px',
            width: 200,
            outline: 'none',
          }}
        />
        <input
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          placeholder="Goal for this run…"
          style={{
            flex: 1,
            background: 'var(--bg-base)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            color: 'var(--text-primary)',
            fontFamily: 'var(--font-display)',
            fontSize: 13,
            padding: '5px 8px',
            outline: 'none',
          }}
        />
        <ToolbarButton label="Run" variant="teal" onClick={handleRun} />
        <ToolbarButton label="Export .py" onClick={handleExportPython} />
        <ToolbarButton label="Save .json" onClick={handleSaveJSON} />
        <ToolbarButton label="Open" onClick={handleOpenJSON} />
        <input
          ref={fileInputRef}
          type="file"
          accept=".json,application/json"
          style={{ display: 'none' }}
          onChange={handleFileChosen}
        />
      </div>

      {/* Palette */}
      <aside
        style={{
          borderRight: '1px solid var(--border)',
          padding: 12,
          background: 'var(--bg-elevated)',
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
        }}
      >
        <span
          style={{
            fontFamily: 'var(--font-display)',
            fontWeight: 600,
            fontSize: 11,
            color: 'var(--text-muted)',
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
          }}
        >
          Palette
        </span>
        <div
          draggable
          onDragStart={onDragStart}
          style={{
            background: 'var(--bg-base)',
            border: '1.5px dashed var(--border)',
            borderRadius: 'var(--radius-md)',
            padding: '12px 10px',
            cursor: 'grab',
            display: 'flex',
            flexDirection: 'column',
            gap: 4,
          }}
        >
          <span
            style={{
              fontFamily: 'var(--font-display)',
              fontWeight: 500,
              fontSize: 13,
              color: 'var(--text-primary)',
            }}
          >
            Agent
          </span>
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 10,
              color: 'var(--text-muted)',
            }}
          >
            Drag onto canvas
          </span>
        </div>

        {status && (
          <p
            style={{
              margin: 0,
              fontFamily: 'var(--font-mono)',
              fontSize: 10,
              color: 'var(--accent-teal)',
              wordBreak: 'break-word',
            }}
          >
            {status}
          </p>
        )}
        {error && (
          <p
            style={{
              margin: 0,
              fontFamily: 'var(--font-mono)',
              fontSize: 10,
              color: '#f87171',
              wordBreak: 'break-word',
            }}
          >
            {error}
          </p>
        )}
      </aside>

      {/* Canvas */}
      <main
        style={{ position: 'relative' }}
        onDragOver={onDragOver}
        onDrop={onDrop}
      >
        <ReactFlow
          nodes={decoratedNodes}
          edges={decoratedEdges}
          nodeTypes={NODE_TYPES}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onEdgeClick={(_, edge) => {
            setSelectedEdgeId(edge.id)
            setSelectedNodeId(null)
          }}
          onPaneClick={() => {
            setSelectedNodeId(null)
            setSelectedEdgeId(null)
          }}
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
        {nodes.length === 0 && (
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
            Drag an Agent from the palette to begin.
          </div>
        )}
      </main>

      {/* Side panel */}
      <aside
        style={{
          borderLeft: '1px solid var(--border)',
          background: 'var(--bg-elevated)',
          overflowY: 'auto',
        }}
      >
        {selectedNode ? (
          <NodePanel
            node={selectedNode}
            providers={providers}
            tools={tools}
            onChange={updateSelectedNode}
            onSetEntry={() => setEntryNode(selectedNode.id)}
            onSetExit={() => setExitNode(selectedNode.id)}
            onDelete={deleteSelectedNode}
          />
        ) : selectedEdge ? (
          <EdgePanel
            edge={selectedEdge}
            conditions={conditions}
            parametrics={parametrics}
            onChange={updateSelectedEdgeCond}
            onDelete={deleteSelectedEdge}
          />
        ) : (
          <EmptyPanel />
        )}
      </aside>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Side panels
// ---------------------------------------------------------------------------

function NodePanel({
  node,
  providers,
  tools,
  onChange,
  onSetEntry,
  onSetExit,
  onDelete,
}: {
  node: BuilderNode
  providers: ProviderDetail[]
  tools: string[]
  onChange: (patch: Partial<AgentNodeData>) => void
  onSetEntry: () => void
  onSetExit: () => void
  onDelete: () => void
}) {
  const agent = node.data.agent
  const providerDetail = providers.find((p) => p.name === agent.provider)
  const baseUrlRequired = providerDetail?.requires_base_url ?? false

  return (
    <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
      <PanelHeader title={node.id} subtitle="agent node" />

      <Field label="name">
        <TextInput
          value={agent.name}
          onChange={(v) => onChange({ agent: { ...agent, name: v } })}
        />
      </Field>

      <Field label="provider">
        <Select
          value={agent.provider}
          options={providers.map((p) => p.name)}
          onChange={(v) => {
            const next = providers.find((p) => p.name === v)
            const defaultBaseUrl = next?.default_base_url ?? null
            onChange({
              agent: {
                ...agent,
                provider: v,
                base_url: defaultBaseUrl,
              },
            })
          }}
        />
      </Field>

      <Field label="model">
        <TextInput
          value={agent.model}
          onChange={(v) => onChange({ agent: { ...agent, model: v } })}
        />
      </Field>

      {(baseUrlRequired || agent.base_url) && (
        <Field
          label={baseUrlRequired ? 'base_url (required)' : 'base_url'}
        >
          <TextInput
            value={agent.base_url ?? ''}
            onChange={(v) => onChange({ agent: { ...agent, base_url: v || null } })}
          />
        </Field>
      )}

      <Field label="system prompt">
        <textarea
          rows={4}
          value={agent.system}
          onChange={(e) => onChange({ agent: { ...agent, system: e.target.value } })}
          style={textareaStyle}
        />
      </Field>

      <Field label="tools">
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {tools.length === 0 ? (
            <span style={mutedStyle}>(no tools registered)</span>
          ) : (
            tools.map((t) => {
              const selected = agent.tools.includes(t)
              return (
                <button
                  key={t}
                  onClick={() => {
                    const next = selected
                      ? agent.tools.filter((x) => x !== t)
                      : [...agent.tools, t]
                    onChange({ agent: { ...agent, tools: next } })
                  }}
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: 10,
                    background: selected ? 'rgba(0,219,184,0.18)' : 'var(--bg-base)',
                    border: `1px solid ${selected ? 'var(--accent-teal)' : 'var(--border)'}`,
                    color: selected ? 'var(--accent-teal)' : 'var(--text-muted)',
                    borderRadius: 4,
                    padding: '2px 6px',
                    cursor: 'pointer',
                  }}
                >
                  {t}
                </button>
              )
            })
          )}
        </div>
        {agent.tools.includes('python_exec') && (
          <p
            style={{
              margin: '6px 0 0 0',
              fontFamily: 'var(--font-mono)',
              fontSize: 10,
              color: 'var(--accent-gold)',
              lineHeight: 1.4,
            }}
          >
            ⚠ python_exec runs code in this process, unsandboxed. Only run
            graphs you trust.
          </p>
        )}
      </Field>

      <Field label="memory bank">
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>
          <input
            type="checkbox"
            checked={agent.memory_bank}
            onChange={(e) => onChange({ agent: { ...agent, memory_bank: e.target.checked } })}
          />
          enable ReasoningBank retrieval / writeback
        </label>
      </Field>

      <Field label="entry / exit">
        <div style={{ display: 'flex', gap: 6 }}>
          <button
            onClick={onSetEntry}
            style={pillButtonStyle(node.data.isEntry, 'var(--accent-teal)')}
          >
            {node.data.isEntry ? '✓ entry' : 'set as entry'}
          </button>
          <button
            onClick={onSetExit}
            style={pillButtonStyle(node.data.isExit, 'var(--accent-gold)')}
          >
            {node.data.isExit ? '✓ exit' : 'set as exit'}
          </button>
        </div>
      </Field>

      <button
        onClick={onDelete}
        style={{
          marginTop: 8,
          background: 'rgba(248,113,113,0.12)',
          color: '#f87171',
          border: '1px solid #f87171',
          borderRadius: 'var(--radius-sm)',
          fontFamily: 'var(--font-display)',
          fontSize: 12,
          padding: '6px 0',
          cursor: 'pointer',
        }}
      >
        Delete node
      </button>
    </div>
  )
}

function EdgePanel({
  edge,
  conditions,
  parametrics,
  onChange,
  onDelete,
}: {
  edge: BuilderEdge
  conditions: string[]
  parametrics: string[]
  onChange: (cond: string | null) => void
  onDelete: () => void
}) {
  const [paramKey, setParamKey] = useState('')
  const [selectedParam, setSelectedParam] = useState<string>('')
  const current = edge.data?.cond ?? null

  return (
    <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
      <PanelHeader
        title={`${edge.source} → ${edge.target}`}
        subtitle="edge"
      />

      <Field label="condition">
        <Select
          value={current ?? ''}
          options={['', ...conditions]}
          placeholder="(unconditional)"
          onChange={(v) => onChange(v ? v : null)}
        />
      </Field>

      {parametrics.length > 0 && (
        <Field label="parametric condition">
          <div style={{ display: 'flex', gap: 6 }}>
            <select
              value={selectedParam}
              onChange={(e) => setSelectedParam(e.target.value)}
              style={selectStyle}
            >
              <option value="">--</option>
              {parametrics.map((p) => (
                <option key={p} value={p.replace(':<key>', '')}>{p}</option>
              ))}
            </select>
            <input
              placeholder="key"
              value={paramKey}
              onChange={(e) => setParamKey(e.target.value)}
              style={{ ...textInputStyle, width: 90 }}
            />
            <button
              disabled={!selectedParam || !paramKey}
              onClick={() => onChange(`${selectedParam}:${paramKey}`)}
              style={pillButtonStyle(false, 'var(--accent-teal)')}
            >
              apply
            </button>
          </div>
        </Field>
      )}

      <button
        onClick={onDelete}
        style={{
          marginTop: 8,
          background: 'rgba(248,113,113,0.12)',
          color: '#f87171',
          border: '1px solid #f87171',
          borderRadius: 'var(--radius-sm)',
          fontFamily: 'var(--font-display)',
          fontSize: 12,
          padding: '6px 0',
          cursor: 'pointer',
        }}
      >
        Delete edge
      </button>
    </div>
  )
}

function EmptyPanel() {
  return (
    <div
      style={{
        padding: 24,
        color: 'var(--text-muted)',
        fontFamily: 'var(--font-display)',
        fontSize: 13,
      }}
    >
      Select a node or edge to configure it. Drag from the palette to add a
      new agent. Drag from a node's bottom handle to its neighbour's top
      handle to connect them.
    </div>
  )
}

// ---------------------------------------------------------------------------
// small styled primitives
// ---------------------------------------------------------------------------

function PanelHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div>
      <p
        style={{
          margin: 0,
          fontFamily: 'var(--font-display)',
          fontWeight: 600,
          fontSize: 14,
          color: 'var(--text-primary)',
        }}
      >
        {title}
      </p>
      <p
        style={{
          margin: 0,
          fontFamily: 'var(--font-mono)',
          fontSize: 10,
          color: 'var(--text-muted)',
        }}
      >
        {subtitle}
      </p>
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <span
        style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 10,
          color: 'var(--text-muted)',
          letterSpacing: '0.04em',
          textTransform: 'uppercase',
        }}
      >
        {label}
      </span>
      {children}
    </div>
  )
}

const textInputStyle: React.CSSProperties = {
  background: 'var(--bg-base)',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius-sm)',
  color: 'var(--text-primary)',
  fontFamily: 'var(--font-mono)',
  fontSize: 12,
  padding: '5px 8px',
  outline: 'none',
}

const textareaStyle: React.CSSProperties = {
  ...textInputStyle,
  fontFamily: 'var(--font-display)',
  fontSize: 12,
  resize: 'vertical',
}

const selectStyle: React.CSSProperties = {
  ...textInputStyle,
  flex: 1,
}

const mutedStyle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: 10,
  color: 'var(--text-muted)',
}

function TextInput({
  value,
  onChange,
}: {
  value: string
  onChange: (v: string) => void
}) {
  return (
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={textInputStyle}
    />
  )
}

function Select({
  value,
  options,
  onChange,
  placeholder,
}: {
  value: string
  options: string[]
  onChange: (v: string) => void
  placeholder?: string
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={selectStyle}
    >
      {options.map((opt) =>
        opt === '' ? (
          <option key="__blank__" value="">
            {placeholder ?? '--'}
          </option>
        ) : (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ),
      )}
    </select>
  )
}

function ToolbarButton({
  label,
  onClick,
  variant = 'muted',
}: {
  label: string
  onClick: () => void
  variant?: 'teal' | 'muted'
}) {
  const color = variant === 'teal' ? 'var(--accent-teal)' : 'var(--text-muted)'
  return (
    <button
      onClick={onClick}
      style={{
        background: variant === 'teal' ? 'rgba(0,219,184,0.12)' : 'transparent',
        border: `1px solid ${variant === 'teal' ? 'var(--accent-teal)' : 'var(--border)'}`,
        borderRadius: 'var(--radius-sm)',
        color,
        fontFamily: 'var(--font-display)',
        fontWeight: 500,
        fontSize: 12,
        padding: '5px 10px',
        cursor: 'pointer',
      }}
    >
      {label}
    </button>
  )
}

function pillButtonStyle(active: boolean, color: string): React.CSSProperties {
  return {
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    background: active ? `${color}22` : 'var(--bg-base)',
    border: `1px solid ${active ? color : 'var(--border)'}`,
    color: active ? color : 'var(--text-muted)',
    borderRadius: 4,
    padding: '4px 8px',
    cursor: 'pointer',
  }
}

// ---------------------------------------------------------------------------
// id allocation
// ---------------------------------------------------------------------------

function nextNodeId(counter: React.MutableRefObject<number>, nodes: BuilderNode[]): string {
  const taken = new Set(nodes.map((n) => n.id))
  for (;;) {
    counter.current += 1
    const candidate = `agent_${counter.current}`
    if (!taken.has(candidate)) return candidate
  }
}

// ---------------------------------------------------------------------------
// download helper
// ---------------------------------------------------------------------------

function downloadFile(filename: string, content: string, mime: string): void {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

// ---------------------------------------------------------------------------
// Provider wrapper so screenToFlowPosition works
// ---------------------------------------------------------------------------

export function GraphBuilder() {
  return (
    <ReactFlowProvider>
      <GraphBuilderInner />
    </ReactFlowProvider>
  )
}

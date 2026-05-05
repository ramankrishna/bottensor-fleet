import { useState } from 'react'
import { useFleetStore } from '../store/fleetStore'

type BtnVariant = 'teal' | 'gold' | 'danger' | 'muted'

function Btn({
  label, onClick, variant = 'muted', disabled = false,
}: {
  label: string
  onClick: () => void
  variant?: BtnVariant
  disabled?: boolean
}) {
  const COLOR: Record<BtnVariant, string> = {
    teal:   'var(--accent-teal)',
    gold:   'var(--accent-gold)',
    danger: '#f87171',
    muted:  'var(--text-muted)',
  }
  const c = COLOR[variant]

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        flex: 1,
        background: disabled ? 'transparent' : `${c}18`,
        border: `1px solid ${disabled ? 'var(--border)' : c}`,
        borderRadius: 'var(--radius-sm)',
        color: disabled ? 'var(--text-muted)' : c,
        fontFamily: 'var(--font-display)',
        fontWeight: 500,
        fontSize: 12,
        padding: '6px 0',
        cursor: disabled ? 'not-allowed' : 'pointer',
        transition: 'background 0.15s',
      }}
    >
      {label}
    </button>
  )
}

export function RunControls() {
  const graphs       = useFleetStore((s) => s.graphs)
  const setGraphs    = useFleetStore((s) => s.setGraphs)
  const providerKeys = useFleetStore((s) => s.providerKeys)
  const setActiveRunId = useFleetStore((s) => s.setActiveRunId)
  const upsertRun    = useFleetStore((s) => s.upsertRun)
  const resetNodes   = useFleetStore((s) => s.resetNodes)
  const activeRunId  = useFleetStore((s) => s.activeRunId)
  const runs         = useFleetStore((s) => s.runs)

  const [goal, setGoal]         = useState('')
  const [graph, setGraph]       = useState('')
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState<string | null>(null)

  // Lazily load graphs once
  const ensureGraphs = async () => {
    if (graphs.length > 0) return
    try {
      const r = await fetch('/api/graphs')
      const data: string[] = await r.json()
      setGraphs(data)
      if (data.length > 0 && !graph) setGraph(data[0])
    } catch { /* server down */ }
  }

  const activeRun = activeRunId ? runs[activeRunId] : null
  const isRunning = activeRun?.status === 'running'
  const isPaused  = activeRun?.status === 'paused'
  const isTerminal = activeRun && (activeRun.status === 'done' || activeRun.status === 'error' || activeRun.status === 'killed')

  const handleStart = async () => {
    if (!goal.trim() || !graph) return
    setError(null)
    setLoading(true)
    resetNodes()
    try {
      const backend = providerKeys[0]?.provider ?? 'anthropic'
      const r = await fetch('/api/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ goal: goal.trim(), graph_module: graph, backend }),
      })
      if (!r.ok) throw new Error(await r.text())
      const { run_id } = await r.json() as { run_id: string }
      upsertRun({ runId: run_id, goal: goal.trim(), graphModule: graph, status: 'running', messageCount: 0 })
      setActiveRunId(run_id)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  const patchRun = async (action: 'pause' | 'resume' | 'kill') => {
    if (!activeRunId) return
    try {
      await fetch(`/api/runs/${activeRunId}/${action}`, { method: 'POST' })
    } catch { /* ignore */ }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <span
        style={{
          fontFamily: 'var(--font-display)',
          fontWeight: 600,
          fontSize: 12,
          color: 'var(--text-muted)',
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
        }}
      >
        Run
      </span>

      {/* Goal input */}
      <textarea
        rows={3}
        value={goal}
        placeholder="Describe the goal…"
        onChange={(e) => setGoal(e.target.value)}
        style={{
          background: 'var(--bg-base)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-sm)',
          color: 'var(--text-primary)',
          fontFamily: 'var(--font-display)',
          fontSize: 13,
          padding: '8px',
          resize: 'vertical',
          outline: 'none',
          width: '100%',
        }}
        onFocus={(e) => { e.currentTarget.style.borderColor = 'var(--accent-teal)' }}
        onBlur={(e) => { e.currentTarget.style.borderColor = 'var(--border)' }}
      />

      {/* Graph selector */}
      <select
        value={graph}
        onClick={ensureGraphs}
        onChange={(e) => setGraph(e.target.value)}
        style={{
          background: 'var(--bg-base)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-sm)',
          color: graph ? 'var(--text-primary)' : 'var(--text-muted)',
          fontFamily: 'var(--font-mono)',
          fontSize: 12,
          padding: '6px 8px',
          width: '100%',
          outline: 'none',
        }}
      >
        <option value="">-- select graph --</option>
        {graphs.map((g) => <option key={g} value={g}>{g}</option>)}
      </select>

      {/* Action buttons */}
      <div style={{ display: 'flex', gap: 6 }}>
        <Btn
          label={loading ? 'Starting…' : 'Start'}
          variant="teal"
          onClick={handleStart}
          disabled={loading || isRunning || isPaused || !goal.trim() || !graph}
        />
        <Btn
          label="Pause"
          variant="gold"
          onClick={() => patchRun('pause')}
          disabled={!isRunning}
        />
        <Btn
          label="Resume"
          variant="teal"
          onClick={() => patchRun('resume')}
          disabled={!isPaused}
        />
        <Btn
          label="Kill"
          variant="danger"
          onClick={() => patchRun('kill')}
          disabled={!isRunning && !isPaused}
        />
      </div>

      {/* Status row */}
      {activeRun && (
        <p
          style={{
            margin: 0,
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            color: isTerminal
              ? activeRun.status === 'done' ? 'var(--accent-teal)' : '#f87171'
              : 'var(--text-muted)',
          }}
        >
          {activeRunId} · {activeRun.status}
        </p>
      )}

      {error && (
        <p style={{ margin: 0, fontFamily: 'var(--font-mono)', fontSize: 11, color: '#f87171' }}>
          {error}
        </p>
      )}
    </div>
  )
}

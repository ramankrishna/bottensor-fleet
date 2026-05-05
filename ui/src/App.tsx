import { useFleetStore } from './store/fleetStore'
import { useFleetWS } from './hooks/useFleetWS'
import { GraphCanvas } from './components/GraphCanvas'
import { AgentPanel } from './components/AgentPanel'
import { ProviderPicker } from './components/ProviderPicker'
import { RunControls } from './components/RunControls'

export default function App() {
  const activeRunId = useFleetStore((s) => s.activeRunId)
  useFleetWS(activeRunId)

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '240px 1fr 360px',
        gridTemplateRows: '48px 1fr',
        height: '100vh',
        width: '100vw',
        background: 'var(--bg-base)',
        overflow: 'hidden',
      }}
    >
      {/* ── Top bar ─────────────────────────────────────────────────────── */}
      <header
        style={{
          gridColumn: '1 / -1',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '0 20px',
          borderBottom: '1px solid var(--border)',
          background: 'var(--bg-elevated)',
        }}
      >
        <span
          style={{
            fontFamily: 'var(--font-serif)',
            fontStyle: 'italic',
            fontSize: 20,
            color: 'var(--accent-teal)',
            letterSpacing: '-0.01em',
          }}
        >
          bottensor
        </span>
        <span
          style={{
            fontFamily: 'var(--font-display)',
            fontWeight: 300,
            fontSize: 13,
            color: 'var(--text-muted)',
          }}
        >
          fleet
        </span>
      </header>

      {/* ── Left rail ───────────────────────────────────────────────────── */}
      <aside
        style={{
          borderRight: '1px solid var(--border)',
          background: 'var(--bg-elevated)',
          overflowY: 'auto',
          padding: 16,
          display: 'flex',
          flexDirection: 'column',
          gap: 24,
        }}
      >
        <RunControls />
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: 16 }}>
          <ProviderPicker />
        </div>
      </aside>

      {/* ── Center canvas ────────────────────────────────────────────────── */}
      <main style={{ position: 'relative', overflow: 'hidden' }}>
        <GraphCanvas />
      </main>

      {/* ── Right rail ──────────────────────────────────────────────────── */}
      <aside style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <AgentPanel />
      </aside>
    </div>
  )
}

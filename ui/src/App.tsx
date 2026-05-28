import { useState } from 'react'
import { useFleetStore } from './store/fleetStore'
import { useFleetWS } from './hooks/useFleetWS'
import { GraphCanvas } from './components/GraphCanvas'
import { AgentPanel } from './components/AgentPanel'
import { RunControls } from './components/RunControls'
import { MemoryTab } from './components/MemoryTab'

type Tab = 'runs' | 'memory'

export default function App() {
  const activeRunId = useFleetStore((s) => s.activeRunId)
  useFleetWS(activeRunId)
  const [tab, setTab] = useState<Tab>('runs')

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: tab === 'runs' ? '240px 1fr 360px' : '1fr',
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
          gap: 16,
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

        <nav style={{ display: 'flex', gap: 4, marginLeft: 24 }}>
          {(['runs', 'memory'] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              style={{
                background: tab === t ? 'rgba(95,179,167,0.12)' : 'transparent',
                border: 'none',
                borderBottom: tab === t ? '2px solid var(--accent-teal)' : '2px solid transparent',
                color: tab === t ? 'var(--accent-teal)' : 'var(--text-muted)',
                fontFamily: 'var(--font-display)',
                fontWeight: 500,
                fontSize: 13,
                padding: '6px 12px',
                cursor: 'pointer',
                textTransform: 'capitalize',
              }}
            >
              {t}
            </button>
          ))}
        </nav>
      </header>

      {tab === 'runs' ? (
        <>
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
          </aside>

          {/* ── Center canvas ────────────────────────────────────────────────── */}
          <main style={{ position: 'relative', overflow: 'hidden' }}>
            <GraphCanvas />
          </main>

          {/* ── Right rail ──────────────────────────────────────────────────── */}
          <aside style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            <AgentPanel />
          </aside>
        </>
      ) : (
        <main style={{ overflow: 'hidden' }}>
          <MemoryTab />
        </main>
      )}
    </div>
  )
}

import { useFleetStore } from '../store/fleetStore'

const ROLE_COLOR: Record<string, string> = {
  user:      'var(--accent-teal)',
  assistant: 'var(--text-primary)',
  tool:      'var(--accent-gold)',
  system:    'var(--text-muted)',
}

const ROLE_LABEL: Record<string, string> = {
  user:      'user',
  assistant: 'agent',
  tool:      'tool',
  system:    'sys',
}

export function AgentPanel() {
  const selectedNode = useFleetStore((s) => s.selectedNode)
  const nodeState    = useFleetStore((s) => selectedNode ? s.nodes[selectedNode] : null)
  const activeRunId  = useFleetStore((s) => s.activeRunId)
  const logs         = useFleetStore((s) => activeRunId ? (s.logs[activeRunId] ?? []) : [])

  const nodeLogs = selectedNode
    ? logs.filter((l) =>
        l.role === 'tool'
          ? l.toolName !== undefined
          : true
      )
    : logs

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        background: 'var(--bg-elevated)',
        borderLeft: '1px solid var(--border)',
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: '12px 16px',
          borderBottom: '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}
      >
        <span
          style={{
            fontFamily: 'var(--font-display)',
            fontWeight: 600,
            fontSize: 13,
            color: 'var(--text-primary)',
          }}
        >
          {selectedNode ?? 'Activity log'}
        </span>
        {nodeState && (
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 10,
              color: 'var(--text-muted)',
              background: 'rgba(138,148,166,0.12)',
              borderRadius: 4,
              padding: '1px 6px',
            }}
          >
            {nodeState.status}
          </span>
        )}
      </div>

      {/* Log entries */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '8px 0',
        }}
      >
        {nodeLogs.length === 0 ? (
          <p
            style={{
              color: 'var(--text-muted)',
              fontFamily: 'var(--font-display)',
              fontSize: 13,
              textAlign: 'center',
              marginTop: 32,
            }}
          >
            {activeRunId ? 'No messages yet.' : 'Start a run to see logs.'}
          </p>
        ) : (
          nodeLogs.map((entry) => (
            <div
              key={entry.id}
              style={{
                padding: '6px 16px',
                borderBottom: '1px solid var(--border)',
                display: 'flex',
                flexDirection: 'column',
                gap: 2,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: 10,
                    color: ROLE_COLOR[entry.role] ?? 'var(--text-muted)',
                    minWidth: 36,
                  }}
                >
                  {ROLE_LABEL[entry.role] ?? entry.role}
                </span>
                <span
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: 10,
                    color: 'var(--text-muted)',
                  }}
                >
                  {new Date(entry.ts).toLocaleTimeString()}
                </span>
              </div>
              <p
                style={{
                  margin: 0,
                  fontFamily: entry.role === 'tool' ? 'var(--font-mono)' : 'var(--font-display)',
                  fontSize: entry.role === 'tool' ? 11 : 13,
                  color: 'var(--text-primary)',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                }}
              >
                {entry.content}
              </p>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

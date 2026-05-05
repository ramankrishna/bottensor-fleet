import { useEffect, useState } from 'react'
import { useFleetStore } from '../store/fleetStore'

function KeyRow({ provider }: { provider: string }) {
  const setKey = useFleetStore((s) => s.setKey)
  const stored = useFleetStore((s) => s.providerKeys.find((k) => k.provider === provider)?.key ?? '')
  const [value, setValue] = useState('')

  useEffect(() => { setValue(stored) }, [stored])

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '6px 0',
        borderBottom: '1px solid var(--border)',
      }}
    >
      <span
        style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 12,
          color: 'var(--text-primary)',
          minWidth: 90,
          flexShrink: 0,
        }}
      >
        {provider}
      </span>
      <input
        type="password"
        value={value}
        placeholder="sk-…"
        onChange={(e) => setValue(e.target.value)}
        onBlur={() => { if (value.trim()) setKey(provider, value.trim()) }}
        style={{
          flex: 1,
          background: 'var(--bg-base)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-sm)',
          color: 'var(--text-primary)',
          fontFamily: 'var(--font-mono)',
          fontSize: 12,
          padding: '4px 8px',
          outline: 'none',
        }}
        onFocus={(e) => { e.currentTarget.style.borderColor = 'var(--accent-teal)' }}
        onBlurCapture={(e) => { e.currentTarget.style.borderColor = 'var(--border)' }}
      />
      {stored && (
        <span
          style={{
            width: 7, height: 7, borderRadius: '50%',
            background: 'var(--accent-teal)',
            flexShrink: 0,
          }}
          title="Key saved"
        />
      )}
    </div>
  )
}

export function ProviderPicker() {
  const providers  = useFleetStore((s) => s.providers)
  const setProviders = useFleetStore((s) => s.setProviders)
  const clearKeys  = useFleetStore((s) => s.clearKeys)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (providers.length > 0) return
    setLoading(true)
    fetch('/api/providers')
      .then((r) => r.json())
      .then((data: string[]) => setProviders(data))
      .catch(() => {/* server not reachable during dev */})
      .finally(() => setLoading(false))
  }, [providers, setProviders])

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 0,
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: 8,
        }}
      >
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
          API Keys
        </span>
        <button
          onClick={clearKeys}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--text-muted)',
            fontFamily: 'var(--font-display)',
            fontSize: 11,
            cursor: 'pointer',
            padding: '2px 6px',
          }}
        >
          clear all
        </button>
      </div>

      {loading && (
        <p style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-display)', fontSize: 12 }}>
          Loading…
        </p>
      )}

      {providers.map((p) => <KeyRow key={p} provider={p} />)}

      {!loading && providers.length === 0 && (
        <p style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-display)', fontSize: 12, margin: 0 }}>
          No providers found. Is the server running?
        </p>
      )}
    </div>
  )
}

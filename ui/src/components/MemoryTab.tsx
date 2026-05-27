import { useCallback, useEffect, useState } from 'react'

interface MemoryItem {
  id: string
  title: string
  description: string
  content: string
  source: string
  scope: string
  confidence: number
  use_count: number
  created_at: string
  last_used_at?: string | null
  task_signature?: string
  related_ids?: string[]
}

const SOURCE_COLOR: Record<string, string> = {
  success:         'var(--accent-teal)',
  failure:         '#f87171',
  matts_contrast:  'var(--accent-gold)',
  manual:          'var(--text-muted)',
}

export function MemoryTab() {
  const [items, setItems] = useState<MemoryItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  const [editing, setEditing] = useState<string | null>(null)
  const [draft, setDraft] = useState<{title: string; description: string; content: string}>({
    title: '', description: '', content: '',
  })

  const fetchItems = useCallback(async (q?: string) => {
    setLoading(true)
    setError(null)
    try {
      const url = q ? `/api/memory?q=${encodeURIComponent(q)}` : '/api/memory'
      const r = await fetch(url)
      if (!r.ok) throw new Error(await r.text())
      const data = await r.json()
      setItems(data.items ?? [])
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchItems() }, [fetchItems])

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    fetchItems(query.trim() || undefined)
  }

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this memory?')) return
    try {
      await fetch(`/api/memory/${id}`, { method: 'DELETE' })
      setItems((prev) => prev.filter((i) => i.id !== id))
    } catch (e) {
      setError(String(e))
    }
  }

  const startEdit = (item: MemoryItem) => {
    setEditing(item.id)
    setDraft({ title: item.title, description: item.description, content: item.content })
  }

  const saveEdit = async () => {
    if (!editing) return
    try {
      const r = await fetch(`/api/memory/${editing}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(draft),
      })
      if (!r.ok) throw new Error(await r.text())
      const updated: MemoryItem = await r.json()
      setItems((prev) => prev.map((i) => (i.id === updated.id ? updated : i)))
      setEditing(null)
    } catch (e) {
      setError(String(e))
    }
  }

  const handleExport = async () => {
    try {
      const r = await fetch('/api/memory/export', { method: 'POST' })
      const data = await r.json()
      const blob = new Blob([JSON.stringify(data.items, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `memory-${new Date().toISOString().slice(0, 10)}.json`
      a.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      setError(String(e))
    }
  }

  const handleImport = async (file: File) => {
    try {
      const text = await file.text()
      const parsed = JSON.parse(text)
      const items = Array.isArray(parsed) ? parsed : parsed.items
      if (!Array.isArray(items)) throw new Error('Expected a JSON array of memory items')
      const r = await fetch('/api/memory/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ items }),
      })
      if (!r.ok) throw new Error(await r.text())
      await fetchItems()
    } catch (e) {
      setError(String(e))
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      {/* Toolbar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '12px 16px',
          borderBottom: '1px solid var(--border)',
          background: 'var(--bg-elevated)',
        }}
      >
        <form onSubmit={handleSearch} style={{ display: 'flex', flex: 1, gap: 6 }}>
          <input
            type="text"
            value={query}
            placeholder="Search memories…"
            onChange={(e) => setQuery(e.target.value)}
            style={{
              flex: 1,
              background: 'var(--bg-base)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-display)',
              fontSize: 13,
              padding: '6px 10px',
              outline: 'none',
            }}
          />
          <button
            type="submit"
            style={{
              background: 'var(--accent-teal)18',
              border: '1px solid var(--accent-teal)',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--accent-teal)',
              fontFamily: 'var(--font-display)',
              fontSize: 12,
              padding: '6px 12px',
              cursor: 'pointer',
            }}
          >
            Search
          </button>
        </form>
        <button
          onClick={handleExport}
          style={{
            background: 'transparent',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            color: 'var(--text-primary)',
            fontFamily: 'var(--font-display)',
            fontSize: 12,
            padding: '6px 12px',
            cursor: 'pointer',
          }}
        >
          Export
        </button>
        <label
          style={{
            background: 'transparent',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            color: 'var(--text-primary)',
            fontFamily: 'var(--font-display)',
            fontSize: 12,
            padding: '6px 12px',
            cursor: 'pointer',
          }}
        >
          Import
          <input
            type="file"
            accept="application/json"
            style={{ display: 'none' }}
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) handleImport(f)
              e.target.value = ''
            }}
          />
        </label>
      </div>

      {/* Body */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '8px 0' }}>
        {error && (
          <p style={{ padding: '8px 16px', color: '#f87171', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
            {error}
          </p>
        )}
        {loading && !items.length && (
          <p style={{ padding: '8px 16px', color: 'var(--text-muted)' }}>Loading…</p>
        )}
        {!loading && items.length === 0 && (
          <p
            style={{
              padding: 24,
              color: 'var(--text-muted)',
              fontFamily: 'var(--font-display)',
              textAlign: 'center',
            }}
          >
            No memories yet. They'll appear here as agents complete tasks.
          </p>
        )}

        {items.map((item) => {
          const isOpen = expanded[item.id]
          const isEditing = editing === item.id
          return (
            <div
              key={item.id}
              style={{
                padding: '10px 16px',
                borderBottom: '1px solid var(--border)',
                cursor: isEditing ? 'default' : 'pointer',
              }}
              onClick={() => {
                if (isEditing) return
                setExpanded((p) => ({ ...p, [item.id]: !p[item.id] }))
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                <span
                  style={{
                    fontFamily: 'var(--font-display)',
                    fontWeight: 600,
                    fontSize: 13,
                    color: 'var(--text-primary)',
                    flex: 1,
                  }}
                >
                  {item.title}
                </span>
                <span
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: 10,
                    color: SOURCE_COLOR[item.source] ?? 'var(--text-muted)',
                    background: 'rgba(138,148,166,0.12)',
                    borderRadius: 4,
                    padding: '1px 6px',
                  }}
                >
                  {item.source}
                </span>
                <span
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: 10,
                    color: 'var(--text-muted)',
                  }}
                  title="confidence · use count"
                >
                  {item.confidence.toFixed(2)} · {item.use_count}×
                </span>
              </div>
              <p
                style={{
                  margin: 0,
                  fontFamily: 'var(--font-display)',
                  fontSize: 12,
                  color: 'var(--text-muted)',
                }}
              >
                {item.description}
              </p>

              {isOpen && !isEditing && (
                <div style={{ marginTop: 8 }}>
                  <pre
                    style={{
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      fontFamily: 'var(--font-mono)',
                      fontSize: 12,
                      color: 'var(--text-primary)',
                      background: 'var(--bg-base)',
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-sm)',
                      padding: 8,
                      margin: 0,
                    }}
                  >
                    {item.content}
                  </pre>
                  <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
                    <button
                      onClick={(e) => { e.stopPropagation(); startEdit(item) }}
                      style={{
                        background: 'transparent',
                        border: '1px solid var(--border)',
                        borderRadius: 'var(--radius-sm)',
                        color: 'var(--text-primary)',
                        fontFamily: 'var(--font-display)',
                        fontSize: 11,
                        padding: '4px 10px',
                        cursor: 'pointer',
                      }}
                    >
                      Edit
                    </button>
                    <button
                      onClick={(e) => { e.stopPropagation(); handleDelete(item.id) }}
                      style={{
                        background: 'transparent',
                        border: '1px solid #f87171',
                        borderRadius: 'var(--radius-sm)',
                        color: '#f87171',
                        fontFamily: 'var(--font-display)',
                        fontSize: 11,
                        padding: '4px 10px',
                        cursor: 'pointer',
                      }}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              )}

              {isEditing && (
                <div style={{ marginTop: 8 }} onClick={(e) => e.stopPropagation()}>
                  <input
                    value={draft.title}
                    onChange={(e) => setDraft((d) => ({ ...d, title: e.target.value }))}
                    placeholder="Title"
                    style={{
                      width: '100%', marginBottom: 6, padding: 6,
                      background: 'var(--bg-base)', color: 'var(--text-primary)',
                      border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                      fontFamily: 'var(--font-display)', fontSize: 13, outline: 'none',
                    }}
                  />
                  <input
                    value={draft.description}
                    onChange={(e) => setDraft((d) => ({ ...d, description: e.target.value }))}
                    placeholder="Description"
                    style={{
                      width: '100%', marginBottom: 6, padding: 6,
                      background: 'var(--bg-base)', color: 'var(--text-primary)',
                      border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                      fontFamily: 'var(--font-display)', fontSize: 12, outline: 'none',
                    }}
                  />
                  <textarea
                    rows={6}
                    value={draft.content}
                    onChange={(e) => setDraft((d) => ({ ...d, content: e.target.value }))}
                    placeholder="Content"
                    style={{
                      width: '100%', marginBottom: 6, padding: 6,
                      background: 'var(--bg-base)', color: 'var(--text-primary)',
                      border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                      fontFamily: 'var(--font-mono)', fontSize: 12, outline: 'none',
                      resize: 'vertical',
                    }}
                  />
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button
                      onClick={saveEdit}
                      style={{
                        background: 'var(--accent-teal)18',
                        border: '1px solid var(--accent-teal)',
                        borderRadius: 'var(--radius-sm)',
                        color: 'var(--accent-teal)',
                        fontFamily: 'var(--font-display)',
                        fontSize: 11,
                        padding: '4px 12px',
                        cursor: 'pointer',
                      }}
                    >
                      Save
                    </button>
                    <button
                      onClick={() => setEditing(null)}
                      style={{
                        background: 'transparent',
                        border: '1px solid var(--border)',
                        borderRadius: 'var(--radius-sm)',
                        color: 'var(--text-muted)',
                        fontFamily: 'var(--font-display)',
                        fontSize: 11,
                        padding: '4px 12px',
                        cursor: 'pointer',
                      }}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

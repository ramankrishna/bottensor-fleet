import { useEffect, useRef } from 'react'
import { useFleetStore } from '../store/fleetStore'

const BASE_DELAY_MS = 500
const MAX_DELAY_MS = 30_000

/** Connect to ws://localhost:8765/ws/runs/{runId}, dispatch events into the
 *  zustand store, and reconnect with exponential back-off on disconnect. */
export function useFleetWS(runId: string | null): void {
  const dispatchEvent = useFleetStore((s) => s.dispatchEvent)
  const wsRef = useRef<WebSocket | null>(null)
  const delayRef = useRef(BASE_DELAY_MS)
  const stoppedRef = useRef(false)

  useEffect(() => {
    if (!runId) return
    stoppedRef.current = false
    delayRef.current = BASE_DELAY_MS

    function connect() {
      if (stoppedRef.current) return

      const url = `ws://localhost:8765/ws/runs/${runId}`
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onmessage = (ev) => {
        try {
          const event = JSON.parse(ev.data as string)
          dispatchEvent(event)
          // Terminal event — stop reconnecting.
          if (event.type === 'run_finished') {
            stoppedRef.current = true
          }
        } catch {
          // malformed frame — ignore
        }
      }

      ws.onopen = () => {
        delayRef.current = BASE_DELAY_MS  // reset back-off on successful connect
      }

      ws.onclose = () => {
        if (stoppedRef.current) return
        const delay = delayRef.current
        delayRef.current = Math.min(delay * 2, MAX_DELAY_MS)
        setTimeout(connect, delay)
      }

      ws.onerror = () => {
        ws.close()  // triggers onclose → retry
      }
    }

    connect()

    return () => {
      stoppedRef.current = true
      wsRef.current?.close()
    }
  }, [runId, dispatchEvent])
}

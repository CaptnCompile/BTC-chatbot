import { useEffect, useState } from 'react'
import ChatPanel from './components/ChatPanel'
import Tape from './components/Tape'
import TradingViewChart from './components/TradingViewChart'
import { fetchSnapshot } from './api'

// Poll a little faster than the backend's 10s price TTL so the tape stays
// current. Extra polls are cheap: they land on the cache, not the exchange.
const POLL_MS = 8000

export default function App() {
  const [snapshot, setSnapshot] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    const controller = new AbortController()
    let timer

    async function poll() {
      try {
        setSnapshot(await fetchSnapshot(controller.signal))
        setError(null)
      } catch (err) {
        if (err.name !== 'AbortError') setError(err.message)
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, POLL_MS)
      }
    }

    poll()
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [])

  const up = (snapshot?.change_24h_pct ?? 0) >= 0
  const dotClass = error ? 'dot down' : snapshot ? `dot ${up ? 'live' : 'down'}` : 'dot'

  return (
    <div className="app">
      <header className="header">
        <div className="brand">BTC Terminal</div>
        <div className="status">
          <span className={dotClass} />
          {error ? 'feed down' : snapshot ? `live · ${snapshot.source}` : 'connecting…'}
        </div>
      </header>

      <Tape snapshot={snapshot} error={error} />

      <main className="split">
        <TradingViewChart />
        <ChatPanel />
      </main>
    </div>
  )
}

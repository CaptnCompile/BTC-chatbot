import { useEffect, useRef, useState } from 'react'
import { streamChat } from '../api'

const STARTERS = [
  'Is BTC volatile today?',
  "What's the current trend?",
  'Should I be cautious entering now?',
  'Explain what RSI is telling us right now',
]

export default function ChatPanel() {
  const [messages, setMessages] = useState([])
  const [draft, setDraft] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [tool, setTool] = useState(null)
  const logRef = useRef(null)
  const abortRef = useRef(null)

  // Pin to the newest message as tokens arrive.
  useEffect(() => {
    const el = logRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages, tool])

  useEffect(() => () => abortRef.current?.abort(), [])

  async function send(text) {
    const content = text.trim()
    if (!content || streaming) return

    const history = [...messages, { role: 'user', content }]
    setMessages(history)
    setDraft('')
    setStreaming(true)
    setTool(null)

    const controller = new AbortController()
    abortRef.current = controller

    // Open the assistant bubble up front so tokens have somewhere to land.
    setMessages((m) => [...m, { role: 'assistant', content: '' }])

    const appendToLast = (patch) =>
      setMessages((m) => {
        const next = [...m]
        const last = next[next.length - 1]
        next[next.length - 1] = { ...last, ...patch(last) }
        return next
      })

    try {
      await streamChat(
        // Only user/assistant turns go back — error bubbles are UI state, not
        // conversation, and replaying them would confuse the model.
        history.filter((m) => m.role === 'user' || m.role === 'assistant'),
        {
          signal: controller.signal,
          onToken: (t) => appendToLast((last) => ({ content: last.content + t })),
          onTool: (name) => setTool(name),
          onError: (msg) => appendToLast(() => ({ role: 'error', content: msg })),
        }
      )
    } catch (err) {
      if (err.name !== 'AbortError') {
        appendToLast(() => ({ role: 'error', content: err.message }))
      }
    } finally {
      setStreaming(false)
      setTool(null)
      abortRef.current = null
      // A stream that closed before emitting anything would leave an empty
      // bubble; drop it rather than render a blank reply.
      setMessages((m) => m.filter((msg, i) => i !== m.length - 1 || msg.content !== ''))
    }
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send(draft)
    }
  }

  const lastIsStreamingBot =
    streaming && messages.length > 0 && messages[messages.length - 1].role === 'assistant'

  return (
    <section className="chat" aria-label="Market analyst chat">
      <div className="chat-head">Analyst · grounded in the tape above</div>

      <div className="log" ref={logRef} role="log" aria-live="polite">
        {messages.length === 0 && (
          <div className="empty">
            <p>Ask about the tape</p>
            {STARTERS.map((s) => (
              <button key={s} className="chip" onClick={() => send(s)} disabled={streaming}>
                {s}
              </button>
            ))}
          </div>
        )}

        {messages.map((m, i) => {
          const cls = m.role === 'user' ? 'user' : m.role === 'error' ? 'err' : 'bot'
          const isLast = i === messages.length - 1
          return (
            <div key={i} className={`msg ${cls}`}>
              {m.content}
              {isLast && lastIsStreamingBot && m.role === 'assistant' && (
                <span className="caret" aria-hidden="true" />
              )}
            </div>
          )
        })}

        {tool && <div className="tool">↳ fetching {tool.replace(/_/g, ' ')}…</div>}
      </div>

      <div className="composer">
        <textarea
          rows={1}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask about price action, trend, volatility…"
          aria-label="Message"
          disabled={streaming}
        />
        <button
          className="send"
          onClick={() => send(draft)}
          disabled={streaming || !draft.trim()}
          aria-label="Send message"
        >
          ↑
        </button>
      </div>

      <div className="disclaimer">Market context and education. Not financial advice.</div>
    </section>
  )
}

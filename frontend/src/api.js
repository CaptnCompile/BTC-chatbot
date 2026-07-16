/** Backend client. */

export async function fetchSnapshot(signal) {
  const res = await fetch('/api/market/snapshot', { signal })
  if (!res.ok) throw new Error(`snapshot failed: HTTP ${res.status}`)
  return res.json()
}

/**
 * POST the conversation and stream the reply back as Server-Sent Events.
 *
 * EventSource can't be used here: it only issues GET requests and can't send a
 * body, so we parse the SSE framing off a fetch stream ourselves.
 *
 * @param {Array<{role: string, content: string}>} messages
 * @param {{onToken, onTool, onError, signal}} handlers
 */
export async function streamChat(messages, { onToken, onTool, onError, signal }) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
    signal,
  })

  if (!res.ok || !res.body) {
    throw new Error(`chat failed: HTTP ${res.status}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    // Frames are delimited by a blank line. A chunk can split a frame in half,
    // so keep the trailing partial in the buffer for the next read.
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''

    for (const frame of frames) {
      const line = frame.split('\n').find((l) => l.startsWith('data: '))
      if (!line) continue

      let event
      try {
        event = JSON.parse(line.slice(6))
      } catch {
        continue // ignore a malformed frame rather than killing the stream
      }

      if (event.type === 'token') onToken(event.text)
      else if (event.type === 'tool') onTool?.(event.name)
      else if (event.type === 'error') onError?.(event.message)
      else if (event.type === 'done') return
    }
  }
}

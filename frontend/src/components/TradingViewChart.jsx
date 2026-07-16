import { useEffect, useRef, useState } from 'react'

const SCRIPT_SRC = 'https://s3.tradingview.com/tv.js'
const CONTAINER_ID = 'tv-chart-container'

/** Load tv.js once, shared across mounts. */
let scriptPromise = null
function loadTradingView() {
  if (window.TradingView) return Promise.resolve()
  if (scriptPromise) return scriptPromise

  scriptPromise = new Promise((resolve, reject) => {
    const el = document.createElement('script')
    el.src = SCRIPT_SRC
    el.async = true
    el.onload = () => resolve()
    el.onerror = () => {
      scriptPromise = null // let a later mount retry
      reject(new Error('could not load tv.js'))
    }
    document.head.appendChild(el)
  })
  return scriptPromise
}

/**
 * TradingView Advanced Real-Time Chart widget.
 *
 * Symbol is BINANCE:BTCUSDT to match the backend's primary feed, so the candles
 * on screen and the numbers the assistant reasons over come from the same
 * venue. Pointing the chart at a different exchange would put the two subtly
 * out of step — a user could read one price on the chart and hear another in
 * the chat.
 */
export default function TradingViewChart() {
  const mounted = useRef(false)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let cancelled = false

    loadTradingView()
      .then(() => {
        if (cancelled || mounted.current || !window.TradingView) return
        mounted.current = true

        // The widget can't be re-configured responsively after mount, so read
        // the viewport once here. On a phone the drawing-tools rail is dead
        // weight that steals width from the candles.
        const narrow = window.matchMedia('(max-width: 900px)').matches

        new window.TradingView.widget({
          container_id: CONTAINER_ID,
          symbol: 'BINANCE:BTCUSDT',
          interval: '60',
          timezone: 'Etc/UTC',
          theme: 'dark',
          style: '1', // candlesticks
          locale: 'en',
          autosize: true,
          hide_side_toolbar: narrow,
          hide_legend: narrow,
          allow_symbol_change: false,
          enable_publishing: false,
          withdateranges: true,
          details: false,
          studies: ['RSI@tv-basicstudies', 'MASimple@tv-basicstudies'],
          backgroundColor: '#0F1216',
          gridColor: 'rgba(35, 42, 52, 0.6)',
        })
      })
      .catch(() => {
        if (!cancelled) setFailed(true)
      })

    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div className="chart">
      <div id={CONTAINER_ID} />
      {failed && (
        <div className="chart-fallback">
          <div>CHART UNAVAILABLE</div>
          <div>tv.js could not load — check the network connection.</div>
          <div>Market data and chat are unaffected.</div>
        </div>
      )}
    </div>
  )
}

/**
 * The tape — a terminal readout of the live market state.
 *
 * Every field here is one the assistant receives in its context on each turn.
 * That's the point: it makes the grounding legible, so a user can check the
 * bot's claims against the same numbers it was given rather than taking them
 * on faith.
 */

const fmtUsd = (v, digits = 2) =>
  v == null
    ? '—'
    : `$${v.toLocaleString('en-US', {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })}`

const fmtPct = (v, digits = 2) => (v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(digits)}%`)

function Cell({ label, value, tone = 'plain', wide = false, small = false }) {
  return (
    <div className={`cell${wide ? ' wide' : ''}`}>
      <div className="cell-label">{label}</div>
      <div className={`cell-value tabular ${tone}${small ? ' small' : ''}`}>{value}</div>
    </div>
  )
}

export default function Tape({ snapshot, error }) {
  if (error) {
    return (
      <div className="tape">
        <Cell label="feed" value="unavailable" tone="down" wide />
        <Cell label="detail" value={error} small wide />
      </div>
    )
  }

  if (!snapshot) {
    return (
      <div className="tape">
        <Cell label="price" value="loading…" tone="plain" />
      </div>
    )
  }

  const h1 = snapshot.timeframes?.['1h']
  const d1 = snapshot.timeframes?.['1d']
  const up = (snapshot.change_24h_pct ?? 0) >= 0

  return (
    <div className="tape">
      {/* Price is the only amber value: the one number everything else qualifies. */}
      <Cell label="BTC / USD" value={fmtUsd(snapshot.price)} tone="" wide />
      <Cell label="24h" value={fmtPct(snapshot.change_24h_pct)} tone={up ? 'up' : 'down'} />
      <Cell
        label="24h range"
        value={`${fmtUsd(snapshot.low_24h, 0)} – ${fmtUsd(snapshot.high_24h, 0)}`}
        wide
      />
      <Cell label="rsi 1h" value={h1?.rsi_14 != null ? h1.rsi_14.toFixed(1) : '—'} />
      <Cell
        label="real vol 1h"
        value={h1?.realized_vol_annual_pct != null ? `${h1.realized_vol_annual_pct.toFixed(1)}%` : '—'}
      />
      <Cell label="atr 1h" value={h1?.atr_pct != null ? `${h1.atr_pct.toFixed(2)}%` : '—'} />
      <Cell label="trend 1h" value={h1?.trend ?? '—'} small wide />
      <Cell label="trend 1d" value={d1?.trend ?? '—'} small wide />
    </div>
  )
}

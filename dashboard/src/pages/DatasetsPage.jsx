import { useEffect, useState } from 'react'
import { getMarkets, getThemes, getEquityTickers, getTokens, getDatasetsSummary } from '../api'
import { Card, CardHeader, CardBody } from '../components/Card'
import { StatCard } from '../components/StatCard'
import { Badge } from '../components/Badge'
import { LoadingState, ErrorState, EmptyState } from '../components/States'
import { Database, Search, Filter, ChevronLeft, ChevronRight } from 'lucide-react'

export default function DatasetsPage() {
  const [tab, setTab] = useState('markets')
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    getDatasetsSummary()
      .then(setSummary)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Datasets</h1>
        <p className="text-slate-400 mt-1">Browse and inspect Polymarket and equity datasets</p>
      </div>

      {/* Summary row */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Markets" value={summary.polymarket.markets} color="purple" />
          <StatCard label="Tokens" value={summary.polymarket.tokens} color="blue" />
          <StatCard label="Price Points" value={summary.polymarket.prices} color="indigo" />
          <StatCard label="Equity Rows" value={summary.equity.rows} color="green" />
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 bg-[#1e293b] rounded-lg p-1 w-fit">
        {['markets', 'tokens', 'equity'].map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium rounded-md transition ${
              tab === t
                ? 'bg-indigo-600 text-white'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
            }`}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {tab === 'markets' && <MarketsTab />}
      {tab === 'tokens' && <TokensTab />}
      {tab === 'equity' && <EquityTab />}
    </div>
  )
}

function MarketsTab() {
  const [markets, setMarkets] = useState([])
  const [themes, setThemes] = useState([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedTheme, setSelectedTheme] = useState('')
  const [page, setPage] = useState(0)
  const [search, setSearch] = useState('')
  const limit = 25

  const load = () => {
    setLoading(true)
    const params = { limit, offset: page * limit }
    if (selectedTheme) params.theme = selectedTheme
    Promise.all([getMarkets(params), themes.length ? Promise.resolve(themes) : getThemes()])
      .then(([res, th]) => {
        setMarkets(res.data)
        setTotal(res.total)
        if (Array.isArray(th) && th.length && !themes.length) setThemes(th)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(load, [page, selectedTheme])

  const filtered = search
    ? markets.filter(m => m.question?.toLowerCase().includes(search.toLowerCase()))
    : markets

  if (error) return <ErrorState message={error} onRetry={load} />

  return (
    <Card>
      <CardHeader className="flex flex-col sm:flex-row gap-3 sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              placeholder="Search markets..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="pl-9 pr-4 py-2 bg-slate-800 border border-slate-600 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:border-indigo-500 focus:outline-none w-64"
            />
          </div>
          <select
            value={selectedTheme}
            onChange={e => { setSelectedTheme(e.target.value); setPage(0) }}
            className="px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-sm text-slate-200 focus:border-indigo-500 focus:outline-none"
          >
            <option value="">All Themes</option>
            {themes.map(t => (
              <option key={t.theme} value={t.theme}>{t.theme} ({t.count})</option>
            ))}
          </select>
        </div>
        <span className="text-xs text-slate-400">{total} markets total</span>
      </CardHeader>
      <div className="overflow-x-auto">
        {loading ? (
          <LoadingState />
        ) : filtered.length === 0 ? (
          <EmptyState icon={Database} title="No markets found" />
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left px-6 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Question</th>
                <th className="text-left px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Theme</th>
                <th className="text-center px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Status</th>
                <th className="text-right px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Tokens</th>
                <th className="text-right px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Prices</th>
                <th className="text-right px-6 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Volume</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/50">
              {filtered.map((m, i) => (
                <tr key={i} className="hover:bg-slate-700/30 transition">
                  <td className="px-6 py-3 text-slate-200 max-w-md truncate" title={m.question}>
                    {m.question?.slice(0, 80) || '—'}
                  </td>
                  <td className="px-4 py-3">
                    <Badge variant="info">{m.theme || 'unknown'}</Badge>
                  </td>
                  <td className="px-4 py-3 text-center">
                    {m.closed ? (
                      <Badge variant="default">Closed</Badge>
                    ) : m.active ? (
                      <Badge variant="success">Active</Badge>
                    ) : (
                      <Badge variant="warning">Inactive</Badge>
                    )}
                  </td>
                  <td className="px-4 py-3 text-right text-slate-300 font-mono">{m.token_count}</td>
                  <td className="px-4 py-3 text-right text-slate-300 font-mono">{m.total_prices?.toLocaleString()}</td>
                  <td className="px-6 py-3 text-right text-slate-300 font-mono">
                    {m.volumeNum ? `$${(m.volumeNum / 1000).toFixed(0)}K` : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {/* Pagination */}
      <div className="flex items-center justify-between px-6 py-3 border-t border-slate-700">
        <span className="text-xs text-slate-400">
          Showing {page * limit + 1}–{Math.min((page + 1) * limit, total)} of {total}
        </span>
        <div className="flex gap-2">
          <button
            disabled={page === 0}
            onClick={() => setPage(p => p - 1)}
            className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 disabled:opacity-40 rounded-md text-xs text-slate-300 transition flex items-center gap-1"
          >
            <ChevronLeft size={14} /> Prev
          </button>
          <button
            disabled={(page + 1) * limit >= total}
            onClick={() => setPage(p => p + 1)}
            className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 disabled:opacity-40 rounded-md text-xs text-slate-300 transition flex items-center gap-1"
          >
            Next <ChevronRight size={14} />
          </button>
        </div>
      </div>
    </Card>
  )
}

function TokensTab() {
  const [tokens, setTokens] = useState([])
  const [loading, setLoading] = useState(true)
  const [minRows, setMinRows] = useState(200)

  useEffect(() => {
    setLoading(true)
    getTokens({ min_rows: minRows, limit: 100 })
      .then(setTokens)
      .finally(() => setLoading(false))
  }, [minRows])

  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-200">Tokens with Price Data</h3>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">Min rows:</span>
          <input
            type="number"
            value={minRows}
            onChange={e => setMinRows(Number(e.target.value))}
            className="w-20 px-2 py-1 bg-slate-800 border border-slate-600 rounded text-sm text-slate-200 focus:border-indigo-500 focus:outline-none"
          />
        </div>
      </CardHeader>
      <div className="overflow-x-auto">
        {loading ? <LoadingState /> : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left px-6 py-3 text-xs font-semibold text-slate-400 uppercase">Question</th>
                <th className="text-left px-4 py-3 text-xs font-semibold text-slate-400 uppercase">Theme</th>
                <th className="text-left px-4 py-3 text-xs font-semibold text-slate-400 uppercase">Token ID</th>
                <th className="text-right px-6 py-3 text-xs font-semibold text-slate-400 uppercase">Prices</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/50">
              {tokens.map((t, i) => (
                <tr key={i} className="hover:bg-slate-700/30 transition">
                  <td className="px-6 py-3 text-slate-200 max-w-sm truncate">{t.question || '—'}</td>
                  <td className="px-4 py-3"><Badge variant="info">{t.theme || '?'}</Badge></td>
                  <td className="px-4 py-3 font-mono text-xs text-slate-400">{t.token_id?.slice(0, 20)}…</td>
                  <td className="px-6 py-3 text-right font-mono text-slate-300">{t.n_prices?.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </Card>
  )
}

function EquityTab() {
  const [tickers, setTickers] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getEquityTickers().then(setTickers).finally(() => setLoading(false))
  }, [])

  return (
    <Card>
      <CardHeader>
        <h3 className="text-sm font-semibold text-slate-200">Equity Tickers</h3>
      </CardHeader>
      <div className="overflow-x-auto">
        {loading ? <LoadingState /> : tickers.length === 0 ? (
          <EmptyState icon={Database} title="No equity data loaded" description="Import BBO-1m data to get started" />
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left px-6 py-3 text-xs font-semibold text-slate-400 uppercase">Ticker</th>
                <th className="text-right px-4 py-3 text-xs font-semibold text-slate-400 uppercase">Rows</th>
                <th className="text-left px-4 py-3 text-xs font-semibold text-slate-400 uppercase">Start Date</th>
                <th className="text-left px-6 py-3 text-xs font-semibold text-slate-400 uppercase">End Date</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/50">
              {tickers.map((t, i) => (
                <tr key={i} className="hover:bg-slate-700/30 transition">
                  <td className="px-6 py-3 font-mono font-bold text-indigo-400">{t.ticker}</td>
                  <td className="px-4 py-3 text-right font-mono text-slate-300">{t.rows?.toLocaleString()}</td>
                  <td className="px-4 py-3 text-slate-300 text-xs">{t.start_date?.slice(0, 19)}</td>
                  <td className="px-6 py-3 text-slate-300 text-xs">{t.end_date?.slice(0, 19)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </Card>
  )
}

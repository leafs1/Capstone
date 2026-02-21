import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getResults } from '../api'
import { Card, CardHeader } from '../components/Card'
import { Badge } from '../components/Badge'
import { LoadingState, ErrorState, EmptyState } from '../components/States'
import { BarChart3, Search, ExternalLink, ArrowUpDown, ChevronLeft, ChevronRight } from 'lucide-react'

export default function ResultsPage() {
  const [results, setResults] = useState([])
  const [total, setTotal] = useState(0)
  const [totalAnalyzed, setTotalAnalyzed] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [direction, setDirection] = useState('any')
  const [sigLevel, setSigLevel] = useState('0.05')
  const [search, setSearch] = useState('')
  const [sortCol, setSortCol] = useState(null)
  const [sortAsc, setSortAsc] = useState(true)

  const load = () => {
    setLoading(true)
    setError(null)
    getResults({ direction, sig_level: sigLevel, limit: 500 })
      .then(res => {
        setResults(res.data)
        setTotal(res.total_filtered)
        setTotalAnalyzed(res.total_analyzed)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(load, [direction, sigLevel])

  const handleSort = (col) => {
    if (sortCol === col) setSortAsc(!sortAsc)
    else { setSortCol(col); setSortAsc(true) }
  }

  let filtered = search
    ? results.filter(r =>
        r.question?.toLowerCase().includes(search.toLowerCase()) ||
        r.theme?.toLowerCase().includes(search.toLowerCase()) ||
        r.token_id?.includes(search)
      )
    : [...results]

  if (sortCol) {
    filtered.sort((a, b) => {
      const va = a[sortCol] ?? 0
      const vb = b[sortCol] ?? 0
      return sortAsc ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1)
    })
  }

  // Pagination
  const [page, setPage] = useState(0)
  const perPage = 25
  const paged = filtered.slice(page * perPage, (page + 1) * perPage)
  const totalPages = Math.ceil(filtered.length / perPage)

  useEffect(() => setPage(0), [search, direction, sigLevel, sortCol, sortAsc])

  const SortHeader = ({ col, children, className = '' }) => (
    <th
      className={`px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider cursor-pointer hover:text-slate-200 transition select-none ${className}`}
      onClick={() => handleSort(col)}
    >
      <span className="flex items-center gap-1">
        {children}
        {sortCol === col && <ArrowUpDown size={12} className="text-indigo-400" />}
      </span>
    </th>
  )

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Results</h1>
        <p className="text-slate-400 mt-1">
          Granger causality results — {totalAnalyzed} pairs analyzed, {total} significant
        </p>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 items-center">
        <div className="relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            placeholder="Search question, theme, token..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="pl-9 pr-4 py-2 bg-[#1e293b] border border-slate-600 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:border-indigo-500 focus:outline-none w-72"
          />
        </div>

        <select
          value={direction}
          onChange={e => setDirection(e.target.value)}
          className="px-3 py-2 bg-[#1e293b] border border-slate-600 rounded-lg text-sm text-slate-200 focus:border-indigo-500 focus:outline-none"
        >
          <option value="any">Any Direction</option>
          <option value="poly_to_eq">Poly → Equity</option>
          <option value="eq_to_poly">Equity → Poly</option>
          <option value="both">Bidirectional</option>
        </select>

        <select
          value={sigLevel}
          onChange={e => setSigLevel(e.target.value)}
          className="px-3 py-2 bg-[#1e293b] border border-slate-600 rounded-lg text-sm text-slate-200 focus:border-indigo-500 focus:outline-none"
        >
          <option value="0.05">p ≤ 0.05</option>
          <option value="0.01">p ≤ 0.01</option>
          <option value="0.001">p ≤ 0.001</option>
          <option value="1.0">All results</option>
        </select>

        <span className="text-xs text-slate-500 ml-auto">{filtered.length} results shown</span>
      </div>

      {/* Table */}
      <Card>
        {loading ? (
          <LoadingState />
        ) : error ? (
          <ErrorState message={error} onRetry={load} />
        ) : filtered.length === 0 ? (
          <EmptyState icon={BarChart3} title="No results found" description="Adjust filters or run an analysis first" />
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="text-left px-6 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Question</th>
                    <SortHeader col="theme">Theme</SortHeader>
                    <SortHeader col="n_obs" className="text-right">Obs</SortHeader>
                    <SortHeader col="p_poly_to_eq_corrected" className="text-right">Poly→Eq p</SortHeader>
                    <SortHeader col="lag_poly_to_eq" className="text-right">Lag</SortHeader>
                    <SortHeader col="p_eq_to_poly_corrected" className="text-right">Eq→Poly p</SortHeader>
                    <SortHeader col="lag_eq_to_poly" className="text-right">Lag</SortHeader>
                    <th className="text-center px-4 py-3 text-xs font-semibold text-slate-400 uppercase">Quality</th>
                    <th className="px-4 py-3"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-700/50">
                  {paged.map((r, i) => (
                    <tr key={i} className="hover:bg-slate-700/30 transition group">
                      <td className="px-6 py-3 max-w-xs">
                        <p className="text-slate-200 truncate text-xs" title={r.question}>
                          {r.question?.slice(0, 70) || '—'}
                        </p>
                      </td>
                      <td className="px-4 py-3">
                        <Badge variant="info">{r.theme}</Badge>
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-xs text-slate-300">
                        {r.n_obs?.toLocaleString()}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className={`font-mono text-xs ${r.sig_poly_to_eq ? 'text-green-400 font-bold' : 'text-slate-400'}`}>
                          {r.p_poly_to_eq_corrected?.toFixed(4)}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-xs text-slate-300">
                        {r.lag_poly_to_eq}m
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className={`font-mono text-xs ${r.sig_eq_to_poly ? 'text-green-400 font-bold' : 'text-slate-400'}`}>
                          {r.p_eq_to_poly_corrected?.toFixed(4)}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-xs text-slate-300">
                        {r.lag_eq_to_poly}m
                      </td>
                      <td className="px-4 py-3 text-center">
                        {r.poly_stationary && r.eq_stationary ? (
                          <Badge variant="success">✓</Badge>
                        ) : (
                          <Badge variant="warning">⚠</Badge>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <Link
                          to={`/market/${encodeURIComponent(r.token_id)}`}
                          className="opacity-0 group-hover:opacity-100 transition text-indigo-400 hover:text-indigo-300"
                        >
                          <ExternalLink size={14} />
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            <div className="flex items-center justify-between px-6 py-3 border-t border-slate-700">
              <span className="text-xs text-slate-400">
                Page {page + 1} of {totalPages} ({filtered.length} results)
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
                  disabled={page + 1 >= totalPages}
                  onClick={() => setPage(p => p + 1)}
                  className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 disabled:opacity-40 rounded-md text-xs text-slate-300 transition flex items-center gap-1"
                >
                  Next <ChevronRight size={14} />
                </button>
              </div>
            </div>
          </>
        )}
      </Card>
    </div>
  )
}

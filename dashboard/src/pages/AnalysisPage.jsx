import { useEffect, useState, useRef } from 'react'
import { runAnalysis, runSingleAnalysis, getJobs, getJobStatus, getTokens, getEquityTickers } from '../api'
import { Card, CardHeader, CardBody } from '../components/Card'
import { Badge } from '../components/Badge'
import { LoadingState } from '../components/States'
import { Play, RefreshCw, CheckCircle, XCircle, Clock, Loader2 } from 'lucide-react'

export default function AnalysisPage() {
  const [tab, setTab] = useState('batch')

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Analysis</h1>
        <p className="text-slate-400 mt-1">Run Granger causality analysis on your datasets</p>
      </div>

      <div className="flex gap-1 bg-[#1e293b] rounded-lg p-1 w-fit">
        {['batch', 'single', 'jobs'].map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium rounded-md transition ${
              tab === t ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
            }`}
          >
            {t === 'batch' ? 'Batch Run' : t === 'single' ? 'Single Token' : 'Job History'}
          </button>
        ))}
      </div>

      {tab === 'batch' && <BatchRunTab />}
      {tab === 'single' && <SingleRunTab />}
      {tab === 'jobs' && <JobsTab />}
    </div>
  )
}

function BatchRunTab() {
  const [ticker, setTicker] = useState('SPY')
  const [maxlag, setMaxlag] = useState(30)
  const [minRows, setMinRows] = useState(200)
  const [limit, setLimit] = useState('')
  const [running, setRunning] = useState(false)
  const [activeJob, setActiveJob] = useState(null)
  const [tickers, setTickers] = useState([])
  const intervalRef = useRef(null)

  useEffect(() => {
    getEquityTickers().then(setTickers).catch(() => {})
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [])

  const handleRun = async () => {
    setRunning(true)
    try {
      const result = await runAnalysis({
        ticker,
        maxlag,
        min_rows: minRows,
        limit: limit || undefined,
      })
      setActiveJob(result)

      // Poll for status
      intervalRef.current = setInterval(async () => {
        try {
          const status = await getJobStatus(result.job_id)
          setActiveJob(status)
          if (status.status !== 'running') {
            clearInterval(intervalRef.current)
            setRunning(false)
          }
        } catch {
          clearInterval(intervalRef.current)
          setRunning(false)
        }
      }, 2000)
    } catch (e) {
      setRunning(false)
      alert('Failed to start analysis: ' + e.message)
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card>
        <CardHeader>
          <h3 className="text-sm font-semibold text-slate-200">Run Batch Granger Analysis</h3>
          <p className="text-xs text-slate-400 mt-1">
            Analyze all tokens with sufficient data against an equity ticker
          </p>
        </CardHeader>
        <CardBody className="space-y-5">
          {/* Ticker */}
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5">Equity Ticker</label>
            <select
              value={ticker}
              onChange={e => setTicker(e.target.value)}
              className="w-full px-3 py-2.5 bg-slate-800 border border-slate-600 rounded-lg text-sm text-slate-200 focus:border-indigo-500 focus:outline-none"
            >
              {tickers.length > 0 ? tickers.map(t => (
                <option key={t.ticker} value={t.ticker}>{t.ticker} ({t.rows?.toLocaleString()} rows)</option>
              )) : (
                <option value="SPY">SPY</option>
              )}
            </select>
          </div>

          {/* Max Lag */}
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5">
              Max Lag (minutes): <span className="text-indigo-400 font-mono">{maxlag}</span>
            </label>
            <input
              type="range" min="5" max="60" value={maxlag}
              onChange={e => setMaxlag(Number(e.target.value))}
              className="w-full accent-indigo-500"
            />
            <div className="flex justify-between text-xs text-slate-500 mt-1"><span>5</span><span>60</span></div>
          </div>

          {/* Min Rows */}
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5">Minimum Data Points</label>
            <input
              type="number" value={minRows}
              onChange={e => setMinRows(Number(e.target.value))}
              className="w-full px-3 py-2.5 bg-slate-800 border border-slate-600 rounded-lg text-sm text-slate-200 focus:border-indigo-500 focus:outline-none"
            />
          </div>

          {/* Limit */}
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5">
              Token Limit <span className="text-slate-500">(optional, blank = all)</span>
            </label>
            <input
              type="number" value={limit} placeholder="All tokens"
              onChange={e => setLimit(e.target.value)}
              className="w-full px-3 py-2.5 bg-slate-800 border border-slate-600 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:border-indigo-500 focus:outline-none"
            />
          </div>

          <button
            onClick={handleRun}
            disabled={running}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-medium rounded-lg transition"
          >
            {running ? <Loader2 size={18} className="animate-spin" /> : <Play size={18} />}
            {running ? 'Running...' : 'Start Analysis'}
          </button>
        </CardBody>
      </Card>

      {/* Active Job Status */}
      <Card>
        <CardHeader>
          <h3 className="text-sm font-semibold text-slate-200">Job Status</h3>
        </CardHeader>
        <CardBody>
          {!activeJob ? (
            <div className="text-center py-12 text-slate-500 text-sm">
              No active job. Configure and start an analysis.
            </div>
          ) : (
            <div className="space-y-6">
              <div className="flex items-center gap-3">
                {activeJob.status === 'running' && <Loader2 size={20} className="animate-spin text-indigo-400" />}
                {activeJob.status === 'completed' && <CheckCircle size={20} className="text-green-400" />}
                {activeJob.status === 'failed' && <XCircle size={20} className="text-red-400" />}
                <div>
                  <p className="text-sm font-medium text-slate-200">{activeJob.id}</p>
                  <p className="text-xs text-slate-400">Started: {activeJob.started}</p>
                </div>
              </div>

              {/* Progress bar */}
              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>Progress</span>
                  <span>{activeJob.progress}%</span>
                </div>
                <div className="w-full bg-slate-700 rounded-full h-3">
                  <div
                    className={`h-3 rounded-full transition-all duration-500 ${
                      activeJob.status === 'completed' ? 'bg-green-500' :
                      activeJob.status === 'failed' ? 'bg-red-500' : 'bg-indigo-500'
                    }`}
                    style={{ width: `${activeJob.progress}%` }}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-800 rounded-lg p-3">
                  <p className="text-xs text-slate-400">Processed</p>
                  <p className="text-lg font-bold text-slate-200">{activeJob.processed} / {activeJob.total}</p>
                </div>
                <div className="bg-slate-800 rounded-lg p-3">
                  <p className="text-xs text-slate-400">Results Found</p>
                  <p className="text-lg font-bold text-green-400">{activeJob.results_count}</p>
                </div>
              </div>

              {activeJob.error && (
                <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
                  <p className="text-xs text-red-400">{activeJob.error}</p>
                </div>
              )}
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  )
}

function SingleRunTab() {
  const [tokens, setTokens] = useState([])
  const [selectedToken, setSelectedToken] = useState('')
  const [ticker, setTicker] = useState('SPY')
  const [maxlag, setMaxlag] = useState(30)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [search, setSearch] = useState('')

  useEffect(() => {
    getTokens({ min_rows: 50, limit: 200 }).then(setTokens).catch(() => {})
  }, [])

  const handleRun = async () => {
    if (!selectedToken) return alert('Select a token first')
    setRunning(true)
    setResult(null)
    setError(null)
    try {
      const res = await runSingleAnalysis({ token_id: selectedToken, ticker, maxlag })
      setResult(res.result)
    } catch (e) {
      setError(e.message)
    } finally {
      setRunning(false)
    }
  }

  const filteredTokens = search
    ? tokens.filter(t => t.question?.toLowerCase().includes(search.toLowerCase()) || t.token_id?.includes(search))
    : tokens

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card>
        <CardHeader>
          <h3 className="text-sm font-semibold text-slate-200">Run Single Token Analysis</h3>
        </CardHeader>
        <CardBody className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5">Search & Select Token</label>
            <input
              type="text"
              placeholder="Search by question or token ID..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:border-indigo-500 focus:outline-none mb-2"
            />
            <div className="max-h-48 overflow-y-auto bg-slate-800 rounded-lg border border-slate-600">
              {filteredTokens.slice(0, 50).map(t => (
                <button
                  key={t.token_id}
                  onClick={() => setSelectedToken(t.token_id)}
                  className={`w-full text-left px-3 py-2 text-xs hover:bg-slate-700 transition border-b border-slate-700/50 ${
                    selectedToken === t.token_id ? 'bg-indigo-500/20 text-indigo-300' : 'text-slate-300'
                  }`}
                >
                  <p className="truncate font-medium">{t.question || 'Unknown'}</p>
                  <p className="text-slate-500 font-mono mt-0.5">{t.token_id?.slice(0, 24)}… ({t.n_prices} prices)</p>
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Ticker</label>
              <input
                value={ticker} onChange={e => setTicker(e.target.value)}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-sm text-slate-200 focus:border-indigo-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Max Lag</label>
              <input
                type="number" value={maxlag} onChange={e => setMaxlag(Number(e.target.value))}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-sm text-slate-200 focus:border-indigo-500 focus:outline-none"
              />
            </div>
          </div>

          <button
            onClick={handleRun}
            disabled={running || !selectedToken}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-medium rounded-lg transition"
          >
            {running ? <Loader2 size={18} className="animate-spin" /> : <Play size={18} />}
            {running ? 'Analyzing...' : 'Run Analysis'}
          </button>
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <h3 className="text-sm font-semibold text-slate-200">Result</h3>
        </CardHeader>
        <CardBody>
          {!result && !error && (
            <div className="text-center py-12 text-slate-500 text-sm">
              Select a token and run analysis to see results
            </div>
          )}
          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}
          {result && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <ResultBox
                  label="Poly → Equity"
                  sig={result.sig_poly_to_eq}
                  lag={result.lag_poly_to_eq}
                  pval={result.p_poly_to_eq_corrected}
                />
                <ResultBox
                  label="Equity → Poly"
                  sig={result.sig_eq_to_poly}
                  lag={result.lag_eq_to_poly}
                  pval={result.p_eq_to_poly_corrected}
                />
              </div>
              <div className="bg-slate-800 rounded-lg p-3 space-y-2 text-xs">
                <Row label="Observations" value={result.n_obs?.toLocaleString()} />
                <Row label="Window" value={`${result.start?.slice(0, 10)} → ${result.end?.slice(0, 10)}`} />
                <Row label="Poly Stationary" value={result.poly_stationary ? '✓ Yes' : '✗ No'} />
                <Row label="Equity Stationary" value={result.eq_stationary ? '✓ Yes' : '✗ No'} />
              </div>
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  )
}

function ResultBox({ label, sig, lag, pval }) {
  return (
    <div className={`rounded-lg p-4 border ${sig ? 'bg-green-500/10 border-green-500/30' : 'bg-slate-800 border-slate-700'}`}>
      <p className="text-xs font-medium text-slate-400 mb-2">{label}</p>
      <Badge variant={sig ? 'success' : 'default'}>{sig ? 'Significant' : 'Not Significant'}</Badge>
      <div className="mt-3 space-y-1 text-xs">
        <p className="text-slate-300">Lag: <span className="font-mono text-white">{lag} min</span></p>
        <p className="text-slate-300">p-value: <span className="font-mono text-white">{pval?.toFixed(6)}</span></p>
      </div>
    </div>
  )
}

function Row({ label, value }) {
  return (
    <div className="flex justify-between">
      <span className="text-slate-400">{label}</span>
      <span className="text-slate-200 font-mono">{value}</span>
    </div>
  )
}

function JobsTab() {
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getJobs().then(setJobs).catch(() => {}).finally(() => setLoading(false))
  }, [])

  if (loading) return <LoadingState />

  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-200">Job History</h3>
        <button
          onClick={() => getJobs().then(setJobs)}
          className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 rounded-md text-xs text-slate-300 transition flex items-center gap-1"
        >
          <RefreshCw size={12} /> Refresh
        </button>
      </CardHeader>
      <div className="overflow-x-auto">
        {jobs.length === 0 ? (
          <div className="text-center py-12 text-slate-500 text-sm">No jobs found</div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left px-6 py-3 text-xs font-semibold text-slate-400 uppercase">Job ID</th>
                <th className="text-center px-4 py-3 text-xs font-semibold text-slate-400 uppercase">Status</th>
                <th className="text-right px-4 py-3 text-xs font-semibold text-slate-400 uppercase">Progress</th>
                <th className="text-right px-4 py-3 text-xs font-semibold text-slate-400 uppercase">Results</th>
                <th className="text-left px-6 py-3 text-xs font-semibold text-slate-400 uppercase">Started</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/50">
              {jobs.map(j => (
                <tr key={j.id} className="hover:bg-slate-700/30 transition">
                  <td className="px-6 py-3 font-mono text-xs text-slate-300">{j.id}</td>
                  <td className="px-4 py-3 text-center">
                    <Badge variant={j.status === 'completed' ? 'success' : j.status === 'failed' ? 'danger' : 'warning'}>
                      {j.status}
                    </Badge>
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-slate-300">{j.progress}%</td>
                  <td className="px-4 py-3 text-right font-mono text-slate-300">{j.results_count}</td>
                  <td className="px-6 py-3 text-xs text-slate-400">{j.started}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </Card>
  )
}

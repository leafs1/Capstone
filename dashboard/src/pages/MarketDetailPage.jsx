import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  ScatterChart, Scatter, ZAxis, ReferenceLine,
} from 'recharts'
import { ArrowLeft, Download, TrendingUp, TrendingDown, CheckCircle, AlertTriangle } from 'lucide-react'
import { getTimeSeries, getScatterData, getPlotPng, getResults } from '../api'
import { Card, CardHeader, CardBody } from '../components/Card'
import { Badge } from '../components/Badge'
import { StatCard } from '../components/StatCard'
import { LoadingState, ErrorState } from '../components/States'

const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-400 mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }}>
          {p.name}: <span className="font-bold">{typeof p.value === 'number' ? p.value.toFixed(4) : p.value}</span>
        </p>
      ))}
    </div>
  )
}

export default function MarketDetailPage() {
  const { tokenId } = useParams()
  const [meta, setMeta] = useState(null)
  const [tsData, setTsData] = useState(null)
  const [scatterData, setScatterData] = useState(null)
  const [pngData, setPngData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeChart, setActiveChart] = useState('interactive')

  useEffect(() => {
    setLoading(true)
    setError(null)

    // Load metadata from results
    getResults({ limit: 500, sig_level: '1.0' })
      .then(res => {
        const match = res.data.find(r => r.token_id === tokenId)
        setMeta(match)
      })
      .catch(() => {})

    // Load chart data in parallel
    Promise.all([
      getTimeSeries(tokenId).catch(() => null),
      getScatterData(tokenId).catch(() => null),
    ])
      .then(([ts, sc]) => {
        setTsData(ts)
        setScatterData(sc)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [tokenId])

  const loadPng = () => {
    getPlotPng(tokenId).then(setPngData).catch(e => setError(e.message))
  }

  if (loading) return <LoadingState message="Loading market data..." />

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start gap-4">
        <Link to="/results" className="mt-1 p-2 rounded-lg bg-slate-700 hover:bg-slate-600 transition">
          <ArrowLeft size={18} className="text-slate-300" />
        </Link>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-white leading-tight">
            {meta?.question || 'Market Detail'}
          </h1>
          <div className="flex items-center gap-3 mt-2">
            {meta?.theme && <Badge variant="info">{meta.theme}</Badge>}
            <span className="text-xs text-slate-400 font-mono">{tokenId.slice(0, 30)}…</span>
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      {meta && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard
            icon={TrendingUp}
            label="Poly → Equity"
            value={meta.sig_poly_to_eq ? 'Significant' : 'Not Sig.'}
            sub={`p=${meta.p_poly_to_eq_corrected?.toFixed(6)}, lag=${meta.lag_poly_to_eq}min`}
            color={meta.sig_poly_to_eq ? 'green' : 'red'}
          />
          <StatCard
            icon={TrendingDown}
            label="Equity → Poly"
            value={meta.sig_eq_to_poly ? 'Significant' : 'Not Sig.'}
            sub={`p=${meta.p_eq_to_poly_corrected?.toFixed(6)}, lag=${meta.lag_eq_to_poly}min`}
            color={meta.sig_eq_to_poly ? 'green' : 'red'}
          />
          <StatCard
            label="Observations"
            value={meta.n_obs}
            sub={`${meta.start_ts?.slice(0, 10)} → ${meta.end_ts?.slice(0, 10)}`}
            color="blue"
          />
          <StatCard
            icon={meta.poly_stationary && meta.eq_stationary ? CheckCircle : AlertTriangle}
            label="Stationarity"
            value={meta.poly_stationary && meta.eq_stationary ? 'Both OK' : 'Warning'}
            sub={`Poly: ${meta.poly_stationary ? '✓' : '✗'} | Eq: ${meta.eq_stationary ? '✓' : '✗'}`}
            color={meta.poly_stationary && meta.eq_stationary ? 'green' : 'amber'}
          />
        </div>
      )}

      {/* Chart tabs */}
      <div className="flex gap-1 bg-[#1e293b] rounded-lg p-1 w-fit">
        {[
          { key: 'interactive', label: 'Time Series' },
          { key: 'scatter', label: 'Returns Scatter' },
          { key: 'matplotlib', label: 'Matplotlib Plot' },
        ].map(t => (
          <button
            key={t.key}
            onClick={() => { setActiveChart(t.key); if (t.key === 'matplotlib' && !pngData) loadPng() }}
            className={`px-4 py-2 text-sm font-medium rounded-md transition ${
              activeChart === t.key ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Charts */}
      {activeChart === 'interactive' && (
        <Card>
          <CardHeader>
            <h3 className="text-sm font-semibold text-slate-200">
              Polymarket Probability vs {meta?.ticker || 'SPY'} Price
            </h3>
          </CardHeader>
          <CardBody>
            {!tsData?.data?.length ? (
              <div className="text-center py-16 text-slate-500 text-sm">No time series data available</div>
            ) : (
              <ResponsiveContainer width="100%" height={420}>
                <LineChart data={tsData.data} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis
                    dataKey="timestamp"
                    tick={{ fill: '#94a3b8', fontSize: 10 }}
                    tickFormatter={v => v?.slice(5, 10)}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    yAxisId="left"
                    domain={[0, 1]}
                    tick={{ fill: '#818cf8', fontSize: 11 }}
                    tickFormatter={v => (v * 100).toFixed(0) + '%'}
                  />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    tick={{ fill: '#f87171', fontSize: 11 }}
                    tickFormatter={v => '$' + v?.toFixed(0)}
                  />
                  <Tooltip content={<ChartTooltip />} />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="poly"
                    stroke="#818cf8"
                    strokeWidth={2}
                    dot={false}
                    name="Polymarket"
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="equity"
                    stroke="#f87171"
                    strokeWidth={2}
                    dot={false}
                    name={meta?.ticker || 'SPY'}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </CardBody>
        </Card>
      )}

      {activeChart === 'scatter' && (
        <Card>
          <CardHeader className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-200">Returns Correlation</h3>
            {scatterData?.correlation != null && (
              <Badge variant={Math.abs(scatterData.correlation) > 0.1 ? 'info' : 'default'}>
                r = {scatterData.correlation.toFixed(4)} ({scatterData.n_points} points)
              </Badge>
            )}
          </CardHeader>
          <CardBody>
            {!scatterData?.data?.length ? (
              <div className="text-center py-16 text-slate-500 text-sm">No scatter data available</div>
            ) : (
              <ResponsiveContainer width="100%" height={420}>
                <ScatterChart margin={{ top: 10, right: 30, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis
                    type="number"
                    dataKey="eq_return"
                    name="Equity Return"
                    tick={{ fill: '#94a3b8', fontSize: 11 }}
                    label={{ value: 'Equity Return (log)', position: 'bottom', fill: '#94a3b8', fontSize: 11 }}
                  />
                  <YAxis
                    type="number"
                    dataKey="poly_return"
                    name="Poly Return"
                    tick={{ fill: '#94a3b8', fontSize: 11 }}
                    label={{ value: 'Poly Δ', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 11 }}
                  />
                  <ZAxis range={[20, 20]} />
                  <Tooltip content={<ChartTooltip />} />
                  <ReferenceLine x={0} stroke="#475569" strokeDasharray="3 3" />
                  <ReferenceLine y={0} stroke="#475569" strokeDasharray="3 3" />
                  <Scatter data={scatterData.data} fill="#6366f1" fillOpacity={0.5} />
                </ScatterChart>
              </ResponsiveContainer>
            )}
          </CardBody>
        </Card>
      )}

      {activeChart === 'matplotlib' && (
        <Card>
          <CardHeader className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-200">Matplotlib Dual-Axis Plot</h3>
            {pngData?.image && (
              <a
                href={pngData.image}
                download={`granger_${tokenId.slice(0, 16)}.png`}
                className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 rounded-md text-xs text-slate-300 transition flex items-center gap-1"
              >
                <Download size={12} /> Download PNG
              </a>
            )}
          </CardHeader>
          <CardBody>
            {pngData?.image ? (
              <img src={pngData.image} alt="Granger plot" className="w-full rounded-lg" />
            ) : (
              <LoadingState message="Generating plot..." />
            )}
          </CardBody>
        </Card>
      )}

      {/* Detailed stats table */}
      {meta && (
        <Card>
          <CardHeader>
            <h3 className="text-sm font-semibold text-slate-200">Full Analysis Details</h3>
          </CardHeader>
          <CardBody>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <h4 className="text-xs font-semibold text-slate-400 uppercase">Polymarket → Equity</h4>
                <DetailRow label="Significant" value={meta.sig_poly_to_eq ? 'Yes ✓' : 'No'} highlight={meta.sig_poly_to_eq} />
                <DetailRow label="Optimal Lag" value={`${meta.lag_poly_to_eq} minutes`} />
                <DetailRow label="Raw p-value" value={meta.p_poly_to_eq_raw?.toFixed(8)} />
                <DetailRow label="Corrected p-value" value={meta.p_poly_to_eq_corrected?.toFixed(8)} />
              </div>
              <div className="space-y-3">
                <h4 className="text-xs font-semibold text-slate-400 uppercase">Equity → Polymarket</h4>
                <DetailRow label="Significant" value={meta.sig_eq_to_poly ? 'Yes ✓' : 'No'} highlight={meta.sig_eq_to_poly} />
                <DetailRow label="Optimal Lag" value={`${meta.lag_eq_to_poly} minutes`} />
                <DetailRow label="Raw p-value" value={meta.p_eq_to_poly_raw?.toFixed(8)} />
                <DetailRow label="Corrected p-value" value={meta.p_eq_to_poly_corrected?.toFixed(8)} />
              </div>
            </div>
            <div className="mt-6 pt-4 border-t border-slate-700 space-y-3">
              <h4 className="text-xs font-semibold text-slate-400 uppercase">Diagnostics</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <DetailRow label="Poly ADF p" value={meta.poly_adf_p?.toFixed(6)} />
                <DetailRow label="Equity ADF p" value={meta.eq_adf_p?.toFixed(6)} />
                <DetailRow label="Poly Stationary" value={meta.poly_stationary ? '✓' : '✗'} highlight={meta.poly_stationary} />
                <DetailRow label="Equity Stationary" value={meta.eq_stationary ? '✓' : '✗'} highlight={meta.eq_stationary} />
              </div>
            </div>
          </CardBody>
        </Card>
      )}
    </div>
  )
}

function DetailRow({ label, value, highlight }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-xs text-slate-400">{label}</span>
      <span className={`text-xs font-mono ${highlight ? 'text-green-400' : highlight === false ? 'text-red-400' : 'text-slate-200'}`}>
        {value ?? '—'}
      </span>
    </div>
  )
}

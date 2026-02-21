import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from 'recharts'
import {
  Database, BarChart3, ArrowRight, TrendingUp, TrendingDown,
  GitBranch, CheckCircle, AlertTriangle
} from 'lucide-react'
import { getDatasetsSummary, getResultsStats } from '../api'
import { StatCard } from '../components/StatCard'
import { Card, CardHeader, CardBody } from '../components/Card'
import { LoadingState, ErrorState } from '../components/States'

const COLORS = ['#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#3b82f6', '#a855f7', '#ec4899', '#14b8a6']

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-300 font-medium">{label ?? payload[0]?.name}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }} className="mt-1">
          {p.name}: <span className="font-bold">{p.value}</span>
        </p>
      ))}
    </div>
  )
}

export default function DashboardPage() {
  const [summary, setSummary] = useState(null)
  const [stats, setStats] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  const load = () => {
    setLoading(true)
    setError(null)
    Promise.all([getDatasetsSummary(), getResultsStats()])
      .then(([s, st]) => { setSummary(s); setStats(st) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(load, [])

  if (loading) return <LoadingState message="Loading dashboard..." />
  if (error) return <ErrorState message={error} onRetry={load} />

  const sigRate = stats.total > 0
    ? ((stats.sig_poly_to_eq + stats.sig_eq_to_poly) / (stats.total * 2) * 100).toFixed(1)
    : 0

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white">Dashboard</h1>
        <p className="text-slate-400 mt-1">
          Overview of your Polymarket ↔ Equity Granger causality research
        </p>
      </div>

      {/* Top Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          icon={Database}
          label="Polymarket Markets"
          value={summary.polymarket.markets}
          sub={`${summary.polymarket.tokens} tokens, ${summary.polymarket.prices.toLocaleString()} prices`}
          color="purple"
        />
        <StatCard
          icon={TrendingUp}
          label="Equity Data Points"
          value={summary.equity.rows}
          sub={`${summary.equity.tickers} ticker(s)`}
          color="blue"
        />
        <StatCard
          icon={BarChart3}
          label="Pairs Analyzed"
          value={stats.total}
          sub={`${stats.avg_observations?.toLocaleString() ?? '—'} avg observations`}
          color="indigo"
        />
        <StatCard
          icon={CheckCircle}
          label="Significant Results"
          value={stats.sig_poly_to_eq + stats.sig_eq_to_poly}
          sub={`${sigRate}% discovery rate`}
          color="green"
        />
      </div>

      {/* Significance Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard
          icon={TrendingUp}
          label="Poly → Equity"
          value={stats.sig_poly_to_eq}
          sub="Polymarket predicts equity moves"
          color="cyan"
        />
        <StatCard
          icon={TrendingDown}
          label="Equity → Poly"
          value={stats.sig_eq_to_poly}
          sub="Equity predicts Polymarket moves"
          color="amber"
        />
        <StatCard
          icon={GitBranch}
          label="Bidirectional"
          value={stats.sig_both}
          sub="Feedback loops (both directions)"
          color="purple"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Lag Distribution */}
        {stats.lag_dist_poly_to_eq?.length > 0 && (
          <Card>
            <CardHeader>
              <h3 className="text-sm font-semibold text-slate-200">
                Lag Distribution — Poly → Equity (minutes)
              </h3>
            </CardHeader>
            <CardBody>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={stats.lag_dist_poly_to_eq}>
                  <XAxis dataKey="lag" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardBody>
          </Card>
        )}

        {/* Theme Breakdown */}
        {stats.theme_breakdown?.length > 0 && (
          <Card>
            <CardHeader>
              <h3 className="text-sm font-semibold text-slate-200">
                Results by Theme
              </h3>
            </CardHeader>
            <CardBody>
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie
                    data={stats.theme_breakdown.slice(0, 8)}
                    dataKey="total"
                    nameKey="theme"
                    cx="50%"
                    cy="50%"
                    outerRadius={90}
                    strokeWidth={2}
                    stroke="#1e293b"
                  >
                    {stats.theme_breakdown.slice(0, 8).map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                  <Legend
                    wrapperStyle={{ fontSize: 11, color: '#94a3b8' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </CardBody>
          </Card>
        )}
      </div>

      {/* P-value distribution */}
      {stats.pvalue_distribution?.length > 0 && (
        <Card>
          <CardHeader>
            <h3 className="text-sm font-semibold text-slate-200">
              P-Value Distribution (Bonferroni-corrected)
            </h3>
          </CardHeader>
          <CardBody>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={stats.pvalue_distribution} layout="vertical">
                <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis dataKey="bucket" type="category" width={100} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardBody>
        </Card>
      )}

      {/* Data quality + Quick links */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <h3 className="text-sm font-semibold text-slate-200">Data Quality</h3>
          </CardHeader>
          <CardBody className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-slate-400">Stationary pairs</span>
              <span className="text-sm font-mono text-green-400">
                {stats.stationary} / {stats.total}
                {stats.total > 0 && ` (${(stats.stationary / stats.total * 100).toFixed(0)}%)`}
              </span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div
                className="bg-green-500 h-2 rounded-full transition-all"
                style={{ width: `${stats.total > 0 ? (stats.stationary / stats.total * 100) : 0}%` }}
              />
            </div>
            {stats.total - stats.stationary > 0 && (
              <div className="flex items-center gap-2 text-xs text-amber-400 mt-2">
                <AlertTriangle size={14} />
                {stats.total - stats.stationary} results may have non-stationary series
              </div>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader>
            <h3 className="text-sm font-semibold text-slate-200">Quick Actions</h3>
          </CardHeader>
          <CardBody className="space-y-2">
            <Link
              to="/results"
              className="flex items-center justify-between px-4 py-3 rounded-lg bg-slate-700/50 hover:bg-slate-700 transition text-sm"
            >
              <span className="text-slate-300">View all Granger results</span>
              <ArrowRight size={16} className="text-slate-400" />
            </Link>
            <Link
              to="/analysis"
              className="flex items-center justify-between px-4 py-3 rounded-lg bg-slate-700/50 hover:bg-slate-700 transition text-sm"
            >
              <span className="text-slate-300">Run new analysis</span>
              <ArrowRight size={16} className="text-slate-400" />
            </Link>
            <Link
              to="/datasets"
              className="flex items-center justify-between px-4 py-3 rounded-lg bg-slate-700/50 hover:bg-slate-700 transition text-sm"
            >
              <span className="text-slate-300">Browse datasets</span>
              <ArrowRight size={16} className="text-slate-400" />
            </Link>
          </CardBody>
        </Card>
      </div>
    </div>
  )
}

import clsx from 'clsx'

export function StatCard({ icon: Icon, label, value, sub, color = 'indigo' }) {
  const colors = {
    indigo: 'from-indigo-500/20 to-indigo-600/5 border-indigo-500/30 text-indigo-400',
    green: 'from-green-500/20 to-green-600/5 border-green-500/30 text-green-400',
    blue: 'from-blue-500/20 to-blue-600/5 border-blue-500/30 text-blue-400',
    amber: 'from-amber-500/20 to-amber-600/5 border-amber-500/30 text-amber-400',
    red: 'from-red-500/20 to-red-600/5 border-red-500/30 text-red-400',
    purple: 'from-purple-500/20 to-purple-600/5 border-purple-500/30 text-purple-400',
    cyan: 'from-cyan-500/20 to-cyan-600/5 border-cyan-500/30 text-cyan-400',
  }

  return (
    <div
      className={clsx(
        'bg-gradient-to-br border rounded-xl p-5 flex flex-col gap-2',
        colors[color]
      )}
    >
      <div className="flex items-center gap-2">
        {Icon && <Icon size={16} className="opacity-70" />}
        <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
          {label}
        </span>
      </div>
      <span className="text-2xl font-bold text-slate-100">
        {typeof value === 'number' ? value.toLocaleString() : value ?? '—'}
      </span>
      {sub && <span className="text-xs text-slate-400">{sub}</span>}
    </div>
  )
}

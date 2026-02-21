import { NavLink, Outlet } from 'react-router-dom'
import {
  LayoutDashboard,
  Database,
  FlaskConical,
  BarChart3,
  Activity,
} from 'lucide-react'
import { useEffect, useState } from 'react'
import { healthCheck } from '../api'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/datasets', icon: Database, label: 'Datasets' },
  { to: '/analysis', icon: FlaskConical, label: 'Analysis' },
  { to: '/results', icon: BarChart3, label: 'Results' },
]

export default function Layout() {
  const [apiOk, setApiOk] = useState(null)

  useEffect(() => {
    healthCheck().then(() => setApiOk(true)).catch(() => setApiOk(false))
    const iv = setInterval(() => {
      healthCheck().then(() => setApiOk(true)).catch(() => setApiOk(false))
    }, 15000)
    return () => clearInterval(iv)
  }, [])

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 flex-shrink-0 bg-[#1e293b] border-r border-slate-700 flex flex-col">
        <div className="p-6 border-b border-slate-700">
          <h1 className="text-xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
            Research Dashboard
          </h1>
          <p className="text-xs text-slate-400 mt-1">Granger Causality Analysis</p>
        </div>

        <nav className="flex-1 p-4 space-y-1">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'
                }`
              }
            >
              <Icon size={18} />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="p-4 border-t border-slate-700">
          <div className="flex items-center gap-2 text-xs">
            <Activity size={14} className={apiOk ? 'text-green-400' : apiOk === false ? 'text-red-400' : 'text-yellow-400'} />
            <span className="text-slate-400">
              API: {apiOk ? 'Connected' : apiOk === false ? 'Disconnected' : 'Checking...'}
            </span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto bg-[#0f172a]">
        <div className="p-8 max-w-[1600px] mx-auto">
          <Outlet />
        </div>
      </main>
    </div>
  )
}

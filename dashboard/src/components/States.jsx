export function Spinner({ size = 'md' }) {
  const sizes = { sm: 'w-4 h-4', md: 'w-8 h-8', lg: 'w-12 h-12' }
  return (
    <div className={`${sizes[size]} animate-spin rounded-full border-2 border-slate-600 border-t-indigo-400`} />
  )
}

export function LoadingState({ message = 'Loading...' }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-4">
      <Spinner />
      <span className="text-sm text-slate-400">{message}</span>
    </div>
  )
}

export function EmptyState({ icon: Icon, title, description }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-3 text-center">
      {Icon && <Icon size={40} className="text-slate-600" />}
      <h3 className="text-lg font-medium text-slate-300">{title}</h3>
      {description && <p className="text-sm text-slate-500 max-w-md">{description}</p>}
    </div>
  )
}

export function ErrorState({ message, onRetry }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-3 text-center">
      <div className="w-12 h-12 rounded-full bg-red-500/20 flex items-center justify-center">
        <span className="text-red-400 text-xl">!</span>
      </div>
      <h3 className="text-lg font-medium text-slate-300">Something went wrong</h3>
      <p className="text-sm text-slate-500 max-w-md">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition"
        >
          Retry
        </button>
      )}
    </div>
  )
}

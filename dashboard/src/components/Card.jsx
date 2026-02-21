import clsx from 'clsx'

export function Card({ children, className, ...props }) {
  return (
    <div
      className={clsx(
        'bg-[#1e293b] border border-slate-700 rounded-xl',
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}

export function CardHeader({ children, className }) {
  return (
    <div className={clsx('px-6 py-4 border-b border-slate-700', className)}>
      {children}
    </div>
  )
}

export function CardBody({ children, className }) {
  return (
    <div className={clsx('px-6 py-4', className)}>
      {children}
    </div>
  )
}

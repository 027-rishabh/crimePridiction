type KpiCardProps = {
  title: string
  value: string
  subtitle?: string
  accent?: 'primary' | 'danger' | 'muted'
}

export function KpiCard({ title, value, subtitle, accent = 'primary' }: KpiCardProps) {
  return (
    <div className={`card kpi-card kpi-${accent}`}>
      <div className="kpi-label">{title}</div>
      <div className="kpi-value">{value}</div>
      {subtitle && <div className="kpi-subtitle">{subtitle}</div>}
    </div>
  )
}
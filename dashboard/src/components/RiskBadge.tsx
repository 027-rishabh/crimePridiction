type RiskBadgeProps = {
  value: number
}

export function RiskBadge({ value }: RiskBadgeProps) {
  let label = 'Low'
  let cls = 'risk-badge-low'
  if (value >= 0.7) {
    label = 'High risk'
    cls = 'risk-badge-high'
  } else if (value >= 0.4) {
    label = 'Moderate'
    cls = 'risk-badge-medium'
  }
  return <span className={`risk-badge ${cls}`}>{label}</span>
}
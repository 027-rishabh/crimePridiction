import { useFilters } from '../lib/filtersContext'

const YEARS: { label: string; value: 'all' | 2021 | 2022 | 2023 }[] = [
  { label: 'All years', value: 'all' },
  { label: '2021', value: 2021 },
  { label: '2022', value: 2022 },
  { label: '2023', value: 2023 },
]

const GROUPS: { label: string; value: 'all' | 'Women' | 'SC' | 'ST' | 'Children' | 'Senior Citizens' }[] = [
  { label: 'All groups', value: 'all' },
  { label: 'Women', value: 'Women' },
  { label: 'SC', value: 'SC' },
  { label: 'ST', value: 'ST' },
  { label: 'Children', value: 'Children' },
  { label: 'Senior citizens', value: 'Senior Citizens' },
]

export function GlobalFilterBar() {
  const { year, group, setYear, setGroup } = useFilters()

  return (
    <div className="global-filters">
      <div className="global-filters-left">
        <span className="global-filters-label">Filters</span>
        <span className="global-filters-pill">Indian metros · 2021–2023 · Crime vulnerability</span>
      </div>
      <div className="global-filters-controls">
        <label className="global-filter-control">
          <span>Year</span>
          <select value={year} onChange={(e) => setYear(e.target.value as any)}>
            {YEARS.map((y) => (
              <option key={y.value} value={y.value}>
                {y.label}
              </option>
            ))}
          </select>
        </label>
        <label className="global-filter-control">
          <span>Vulnerable group</span>
          <select value={group} onChange={(e) => setGroup(e.target.value as any)}>
            {GROUPS.map((g) => (
              <option key={g.value} value={g.value}>
                {g.label}
              </option>
            ))}
          </select>
        </label>
      </div>
    </div>
  )
}
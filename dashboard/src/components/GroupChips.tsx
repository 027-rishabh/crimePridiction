import type { GroupFilter } from '../lib/filtersContext'

const GROUP_OPTIONS: { label: string; value: GroupFilter }[] = [
  { label: 'All', value: 'all' },
  { label: 'Women', value: 'Women' },
  { label: 'SC', value: 'SC' },
  { label: 'ST', value: 'ST' },
  { label: 'Children', value: 'Children' },
  { label: 'Senior citizens', value: 'Senior Citizens' },
]

type Props = {
  value: GroupFilter
  onChange: (value: GroupFilter) => void
}

export function GroupChips({ value, onChange }: Props) {
  return (
    <div className="chip-row">
      {GROUP_OPTIONS.map((g) => (
        <button
          key={g.value}
          type="button"
          className={value === g.value ? 'chip chip-active' : 'chip'}
          onClick={() => onChange(g.value)}
        >
          {g.label}
        </button>
      ))}
    </div>
  )
}
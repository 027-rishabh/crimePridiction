import { createContext, useContext, useState, type ReactNode } from 'react'

export type YearFilter = 'all' | 2021 | 2022 | 2023
export type GroupFilter = 'all' | 'Women' | 'SC' | 'ST' | 'Children' | 'Senior Citizens'

export type FiltersState = {
  year: YearFilter
  group: GroupFilter
  setYear: (year: YearFilter) => void
  setGroup: (group: GroupFilter) => void
}

const FiltersContext = createContext<FiltersState | undefined>(undefined)

export function FiltersProvider({ children }: { children: ReactNode }) {
  const [year, setYear] = useState<YearFilter>('all')
  const [group, setGroup] = useState<GroupFilter>('all')

  return (
    <FiltersContext.Provider value={{ year, group, setYear, setGroup }}>{children}</FiltersContext.Provider>
  )
}

export function useFilters(): FiltersState {
  const ctx = useContext(FiltersContext)
  if (!ctx) {
    throw new Error('useFilters must be used within FiltersProvider')
  }
  return ctx
}
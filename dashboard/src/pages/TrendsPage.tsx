import { useEffect, useMemo, useState } from 'react'
import { loadCityYearGroupTrends, type CityYearGroupRecord } from '../lib/dataClient'
import { useFilters } from '../lib/filtersContext'
import { GroupChips } from '../components/GroupChips'
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend, BarChart, Bar } from 'recharts'

export function TrendsPage() {
  const [trends, setTrends] = useState<CityYearGroupRecord[]>([])
  const [selectedCity, setSelectedCity] = useState<string>('Bhopal')
  const [selectedGroup, setSelectedGroup] = useState<string>('Women')
  const { year, group } = useFilters()

  useEffect(() => {
    loadCityYearGroupTrends().then(setTrends).catch(console.error)
  }, [])

  const cities = useMemo(() => Array.from(new Set(trends.map((d) => d.City))).sort(), [trends])

  const effectiveGroup = group === 'all' ? selectedGroup : group
  const effectiveYear = year === 'all' ? undefined : year

  const cityGroupSeries = trends.filter((d) => {
    const cityOk = d.City === selectedCity
    const groupOk = d.Group === effectiveGroup
    const yearOk = effectiveYear === undefined || d.Year === effectiveYear
    return cityOk && groupOk && yearOk
  })

  const latestYear = useMemo(() => {
    if (effectiveYear !== undefined) return effectiveYear
    return trends.length ? Math.max(...trends.map((d) => d.Year)) : 2023
  }, [trends, effectiveYear])

  const latestYearCityComparison = trends.filter((d) => {
    const groupOk = d.Group === effectiveGroup
    const yearOk = d.Year === latestYear
    return groupOk && yearOk
  })

  return (
    <div className="page">
      <header className="page-header">
        <h1>Trends &amp; Comparisons</h1>
        <p>
          Time-series and cross-sectional views of crime exposure by vulnerable group and city. These plots underpin the
          descriptive part of a research paper.
        </p>
      </header>

      <section className="controls-row">
        <label>
          City
          <select value={selectedCity} onChange={(e) => setSelectedCity(e.target.value)}>
            {cities.map((city) => (
              <option key={city} value={city}>
                {city}
              </option>
            ))}
          </select>
        </label>
        <label className="chip-label">
          Vulnerable group
          <GroupChips value={effectiveGroup as any} onChange={(g) => setSelectedGroup(g === 'all' ? 'Women' : g)} />
        </label>
      </section>

      <section className="chart-section">
        <header>
          <h2>Trend for selected city &amp; group</h2>
          <p>
            Year-on-year crime rate for the chosen city and vulnerable group. This helps identify increasing, decreasing,
            or stable exposure.
          </p>
        </header>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={cityGroupSeries} margin={{ left: 8, right: 16, top: 16, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="Year" stroke="#ccc" />
              <YAxis stroke="#ccc" />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="Crime_Rate" stroke="#ff7043" strokeWidth={2} dot />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="chart-section">
        <header>
          <h2>City comparison in {latestYear} ({selectedGroup})</h2>
          <p>
            Cross-sectional comparison of cities for a fixed year and vulnerable group. This supports city-level
            benchmarking.
          </p>
        </header>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={latestYearCityComparison} margin={{ left: 8, right: 16, top: 16, bottom: 120 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="City" angle={-45} textAnchor="end" height={120} stroke="#ccc" />
              <YAxis stroke="#ccc" />
              <Tooltip />
              <Bar dataKey="Crime_Rate" fill="#64b5f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  )
}

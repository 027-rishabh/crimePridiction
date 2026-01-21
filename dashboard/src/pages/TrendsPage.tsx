import { useEffect, useMemo, useState } from 'react'
import { loadCityYearGroupTrends, type CityYearGroupRecord } from '../lib/dataClient'
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend, BarChart, Bar } from 'recharts'

export function TrendsPage() {
  const [trends, setTrends] = useState<CityYearGroupRecord[]>([])
  const [selectedCity, setSelectedCity] = useState<string>('Bhopal')
  const [selectedGroup, setSelectedGroup] = useState<string>('Women')

  useEffect(() => {
    loadCityYearGroupTrends().then(setTrends).catch(console.error)
  }, [])

  const cities = useMemo(() => Array.from(new Set(trends.map((d) => d.City))).sort(), [trends])
  const groups = useMemo(() => Array.from(new Set(trends.map((d) => d.Group))).sort(), [trends])

  const cityGroupSeries = trends.filter((d) => d.City === selectedCity && d.Group === selectedGroup)
  const latestYear = useMemo(() => (trends.length ? Math.max(...trends.map((d) => d.Year)) : 2023), [trends])
  const latestYearCityComparison = trends.filter((d) => d.Year === latestYear && d.Group === selectedGroup)

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
        <label>
          Vulnerable group
          <select value={selectedGroup} onChange={(e) => setSelectedGroup(e.target.value)}>
            {groups.map((group) => (
              <option key={group} value={group}>
                {group}
              </option>
            ))}
          </select>
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

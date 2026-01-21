import { useEffect, useState } from 'react'
import { loadCityYearGroupTrends, loadPolicyInsights, type CityYearGroupRecord, type PolicyInsights } from '../lib/dataClient'
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts'

export function OverviewPage() {
  const [trends, setTrends] = useState<CityYearGroupRecord[]>([])
  const [insights, setInsights] = useState<PolicyInsights | null>(null)

  useEffect(() => {
    loadCityYearGroupTrends().then(setTrends).catch(console.error)
    loadPolicyInsights().then(setInsights).catch(console.error)
  }, [])

  const years = Array.from(new Set(trends.map((d) => d.Year))).sort()
  const summaryByYear = years.map((year) => {
    const subset = trends.filter((d) => d.Year === year)
    const avg = subset.reduce((acc, d) => acc + d.Crime_Rate, 0) / (subset.length || 1)
    return { Year: year, AvgCrimeRate: avg }
  })

  return (
    <div className="page">
      <header className="page-header">
        <h1>Overview</h1>
        <p>
          Research-grade exploratory dashboard summarising crime exposure across vulnerable groups in Indian
          metropolitan cities (2021–2023).
        </p>
      </header>

      <section className="cards-row">
        <div className="card">
          <h2>Coverage</h2>
          <p>
            {insights ? (
              <>
                <strong>{insights.n_cities}</strong> cities, 3 years, 5 vulnerable groups
              </>
            ) : (
              'Loading…'
            )}
          </p>
        </div>
        <div className="card">
          <h2>Highest-risk cities (overall CVI)</h2>
          <ul>
            {insights?.top_high_risk_cities.slice(0, 3).map((c) => (
              <li key={c.City}>
                <strong>{c.City}</strong> – CVI {c.CVI_Overall.toFixed(3)}
              </li>
            )) || <li>Loading…</li>}
          </ul>
        </div>
        <div className="card">
          <h2>Most vulnerable groups</h2>
          <ul>
            {insights?.most_vulnerable_groups.map((g) => (
              <li key={g.Group}>
                <strong>{g.Group}</strong>
              </li>
            )) || <li>Loading…</li>}
          </ul>
        </div>
      </section>

      <section className="chart-section">
        <header>
          <h2>Total crime intensity over time</h2>
          <p>
            Average crime rate across all cities and vulnerable groups, by year. This provides a high-level sense of
            whether exposure is rising, falling, or stable over the study window.
          </p>
        </header>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={summaryByYear} margin={{ left: 8, right: 16, top: 16, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="Year" stroke="#ccc" />
              <YAxis stroke="#ccc" />
              <Tooltip />
              <Line type="monotone" dataKey="AvgCrimeRate" stroke="#ff7043" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  )
}

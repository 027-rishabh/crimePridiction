import { useEffect, useState, useMemo } from 'react'
import { loadCityYearGroupTrends, loadPolicyInsights, type CityYearGroupRecord, type PolicyInsights } from '../lib/dataClient'
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts'
import { useFilters } from '../lib/filtersContext'
import { KpiCard } from '../components/KpiCard'

export function OverviewPage() {
  const [trends, setTrends] = useState<CityYearGroupRecord[]>([])
  const [insights, setInsights] = useState<PolicyInsights | null>(null)
  const { year, group } = useFilters()

  useEffect(() => {
    loadCityYearGroupTrends().then(setTrends).catch(console.error)
    loadPolicyInsights().then(setInsights).catch(console.error)
  }, [])

  const filteredTrends = useMemo(() => {
    return trends.filter((d) => {
      const yearOk = year === 'all' || d.Year === year
      const groupOk = group === 'all' || d.Group === group
      return yearOk && groupOk
    })
  }, [trends, year, group])

  const years = Array.from(new Set(filteredTrends.map((d) => d.Year))).sort()
  const summaryByYear = years.map((y) => {
    const subset = filteredTrends.filter((d) => d.Year === y)
    const avg = subset.reduce((acc, d) => acc + d.Crime_Rate, 0) / (subset.length || 1)
    return { Year: y, AvgCrimeRate: avg }
  })

  const totalCities = useMemo(() => new Set(trends.map((d) => d.City)).size, [trends])

  return (
    <div className="page">
      <header className="page-header">
        <h1>Overview</h1>
        <p>
          Research-grade exploratory dashboard summarising crime exposure across vulnerable groups in Indian
          metropolitan cities (2021–2023).
        </p>
        <p className="method-note">
          Note: all crime metrics are expressed as crime rates (incidents per lakh population of the relevant group),
          not absolute case counts, to allow fair comparison across cities of different sizes.
        </p>
      </header>

      <section className="cards-row">
        <KpiCard
          title="Coverage"
          value={insights ? `${insights.n_cities} cities` : '–'}
          subtitle="3 years · 5 vulnerable groups"
        />
        <KpiCard
          title="Highest-risk city (overall CVI)"
          value={insights ? insights.top_high_risk_cities[0]?.City ?? '–' : '–'}
          subtitle={
            insights ? `CVI ${insights.top_high_risk_cities[0]?.CVI_Overall.toFixed(3) ?? ''}` : 'Based on composite index'
          }
          accent="danger"
        />
        <KpiCard
          title="Most vulnerable group"
          value={insights ? insights.most_vulnerable_groups[0]?.Group ?? '–' : '–'}
          subtitle={`${totalCities} cities in sample`}
        />
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

import { useEffect, useMemo, useState } from 'react'
import { fetchCsv } from '../lib/dataClient'
import { useFilters } from '../lib/filtersContext'
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, BarChart, Bar } from 'recharts'

 type CviYearRecord = {
  City: string
  Year: number
  CVI: number
}

 type CviOverallRecord = {
  City: string
  CVI_Overall: number
}

export function CviPage() {
  const [cviYear, setCviYear] = useState<CviYearRecord[]>([])
  const [, setCviOverall] = useState<CviOverallRecord[]>([])
  const [selectedCity, setSelectedCity] = useState<string>('Bhopal')
  const [selectedYear, setSelectedYear] = useState<number>(2023)
  const { year } = useFilters()

  useEffect(() => {
    fetchCsv<CviYearRecord>('cvi_city_year.csv').then(setCviYear).catch(console.error)
    fetchCsv<CviOverallRecord>('cvi_city_overall.csv').then(setCviOverall).catch(console.error)
  }, [])

  const cities = useMemo(() => Array.from(new Set(cviYear.map((d) => d.City))).sort(), [cviYear])
  const years = useMemo(() => Array.from(new Set(cviYear.map((d) => d.Year))).sort(), [cviYear])

  const effectiveYear = year === 'all' ? selectedYear : year

  const seriesCity = cviYear.filter((d) => d.City === selectedCity)
  const seriesYear = cviYear.filter((d) => d.Year === effectiveYear).sort((a, b) => b.CVI - a.CVI)

  return (
    <div className="page">
      <header className="page-header">
        <h1>Crime Vulnerability Index (CVI)</h1>
        <p>
          Composite index combining normalized crime rates and group-specific weights. Higher CVI indicates greater
          relative vulnerability within the metropolitan system.
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
          Year (for cross-sectional view)
          <select value={selectedYear} onChange={(e) => setSelectedYear(Number(e.target.value))}>
            {years.map((year) => (
              <option key={year} value={year}>
                {year}
              </option>
            ))}
          </select>
        </label>
      </section>

      <section className="chart-section">
        <header>
          <h2>CVI over time for selected city</h2>
          <p>
            Tracks how aggregate vulnerability for a city evolves across years, after adjusting for group weights and
            normalization.
          </p>
        </header>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={seriesCity} margin={{ left: 8, right: 16, top: 16, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="Year" stroke="#ccc" />
              <YAxis stroke="#ccc" />
              <Tooltip />
              <Line type="monotone" dataKey="CVI" stroke="#ff7043" strokeWidth={2} dot />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="chart-section">
        <header>
          <h2>CVI by city in {effectiveYear}</h2>
          <p>
            Cross-sectional ranking of cities by CVI in a given year. This can be used to identify high-priority cities
            for intervention.
          </p>
        </header>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={seriesYear} margin={{ left: 8, right: 16, top: 16, bottom: 120 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="City" angle={-45} textAnchor="end" height={120} stroke="#ccc" />
              <YAxis stroke="#ccc" />
              <Tooltip
                formatter={(value) => [
                  typeof value === 'number' ? value.toFixed(3) : String(value ?? ''),
                  'CVI',
                ]}
                labelFormatter={(label) => String(label)}
              />
              <Bar dataKey="CVI" fill="#f97316" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  )
}

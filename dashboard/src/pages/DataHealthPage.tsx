import { useEffect, useState } from 'react'
import { loadDataHealthSummary, type DataHealthSummary } from '../lib/dataClient'
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts'

export function DataHealthPage() {
  const [summary, setSummary] = useState<DataHealthSummary | null>(null)

  useEffect(() => {
    loadDataHealthSummary().then(setSummary).catch(console.error)
  }, [])

  const missingData = summary
    ? Object.entries(summary.missing_by_column).map(([key, value]) => ({ column: key, missing: value }))
    : []

  return (
    <div className="page">
      <header className="page-header">
        <h1>Data Health</h1>
        <p>
          Sanity checks on the integrated crime dataset: missing values, duplicates, and distributional anomalies. This
          page supports methodological transparency for research use.
        </p>
      </header>

      {summary && (
        <section className="cards-row">
          <div className="card">
            <h2>Structure</h2>
            <p>
              {summary.rows.toLocaleString()} rows × {summary.columns} columns
            </p>
          </div>
          <div className="card">
            <h2>Cities &amp; Years</h2>
            <p>
              {summary.cities} cities over {summary.years.length} years
            </p>
          </div>
          <div className="card">
            <h2>Duplicates</h2>
            <p>{summary.duplicate_rows} duplicated (City, Year, Group) records</p>
          </div>
        </section>
      )}

      <section className="chart-section">
        <header>
          <h2>Missing values by column</h2>
          <p>
            Columns with non-zero missingness should be examined carefully before drawing substantive conclusions. In
            this curated dataset, missingness is expected to be minimal.
          </p>
        </header>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={missingData} margin={{ left: 8, right: 16, top: 16, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="column" angle={-45} textAnchor="end" height={80} stroke="#ccc" />
              <YAxis stroke="#ccc" />
              <Tooltip />
              <Bar dataKey="missing" fill="#ffb74d" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  )
}

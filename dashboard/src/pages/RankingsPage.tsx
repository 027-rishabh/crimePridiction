import { useEffect, useState } from 'react'
import { fetchCsv, loadPolicyInsights, type PolicyInsights } from '../lib/dataClient'

 type CityRanking = {
  City: string
  Year: number
  CVI: number
  Rank: number
}

 type GroupRanking = {
  Group: string
  Group_Vulnerability_Score: number
  Rank: number
}

export function RankingsPage() {
  const [cityRankings, setCityRankings] = useState<CityRanking[]>([])
  const [groupRankings, setGroupRankings] = useState<GroupRanking[]>([])
  const [insights, setInsights] = useState<PolicyInsights | null>(null)
  const [selectedYear, setSelectedYear] = useState<number | 'overall'>(2023)

  useEffect(() => {
    fetchCsv<CityRanking>('rankings_cities.csv').then(setCityRankings).catch(console.error)
    fetchCsv<GroupRanking>('rankings_groups.csv').then(setGroupRankings).catch(console.error)
    loadPolicyInsights().then(setInsights).catch(console.error)
  }, [])

  const years = Array.from(new Set(cityRankings.map((r) => r.Year))).sort()

  const filteredCities =
    selectedYear === 'overall'
      ? []
      : cityRankings
          .filter((r) => r.Year === selectedYear)
          .sort((a, b) => a.Rank - b.Rank)
          .slice(0, 20)

  const sortedGroups = [...groupRankings].sort((a, b) => a.Rank - b.Rank)

  return (
    <div className="page">
      <header className="page-header">
        <h1>Rankings &amp; Insights</h1>
        <p>
          Rankings of cities and vulnerable groups based on the Crime Vulnerability Index. These tables provide a
          policy-ready summary of high-risk contexts.
        </p>
      </header>

      <section className="cards-row">
        <div className="card">
          <h2>Interpretation notes</h2>
          <ul>
            {insights?.interpretation_notes.map((note) => (
              <li key={note}>{note}</li>
            )) || <li>Loading…</li>}
          </ul>
        </div>
        <div className="card">
          <h2>Top high-risk cities (overall)</h2>
          <ol>
            {insights?.top_high_risk_cities.map((c) => (
              <li key={c.City}>
                {c.City} (CVI {c.CVI_Overall.toFixed(3)})
              </li>
            )) || <li>Loading…</li>}
          </ol>
        </div>
      </section>

      <section className="table-section">
        <header className="section-header-inline">
          <div>
            <h2>City rankings by year</h2>
            <p>Top 20 cities with highest CVI for the selected year.</p>
          </div>
          <label>
            Year
            <select value={selectedYear} onChange={(e) => setSelectedYear(Number(e.target.value))}>
              {years.map((year) => (
                <option key={year} value={year}>
                  {year}
                </option>
              ))}
            </select>
          </label>
        </header>
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>City</th>
                <th>CVI</th>
              </tr>
            </thead>
            <tbody>
              {filteredCities.map((row) => (
                <tr key={`${row.City}-${row.Year}`}>
                  <td>{row.Rank}</td>
                  <td>{row.City}</td>
                  <td>{row.CVI.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="table-section">
        <header>
          <h2>Vulnerable group rankings</h2>
          <p>
            Groups are ordered by their average normalized vulnerability score across all cities and years, consistent
            with the CVI construction logic.
          </p>
        </header>
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Group</th>
                <th>Vulnerability score</th>
              </tr>
            </thead>
            <tbody>
              {sortedGroups.map((g) => (
                <tr key={g.Group}>
                  <td>{g.Rank}</td>
                  <td>{g.Group}</td>
                  <td>{g.Group_Vulnerability_Score.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}

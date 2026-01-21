import { MapContainer, TileLayer, CircleMarker, Tooltip } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import { useEffect, useMemo, useState } from 'react'
import { fetchCsv } from '../lib/dataClient'

const AnyMapContainer = MapContainer as any
const AnyTileLayer = TileLayer as any
const AnyCircleMarker = CircleMarker as any

 type GeoRecord = {
  City: string
  Year: number
  Group: string
  Crime_Rate: number
  Latitude: number
  Longitude: number
}

export function HeatmapPage() {
  const [data, setData] = useState<GeoRecord[]>([])
  const [selectedYear, setSelectedYear] = useState<number>(2023)
  const [selectedGroup, setSelectedGroup] = useState<string>('Women')

  useEffect(() => {
    fetchCsv<GeoRecord>('geo_city_group_year.csv').then(setData).catch(console.error)
  }, [])

  const years = useMemo(() => Array.from(new Set(data.map((d) => d.Year))).sort(), [data])
  const groups = useMemo(() => Array.from(new Set(data.map((d) => d.Group))).sort(), [data])

  const filtered = data.filter((d) => d.Year === selectedYear && d.Group === selectedGroup)
  const maxCrime = filtered.reduce((acc, d) => Math.max(acc, d.Crime_Rate), 0)

  return (
    <div className="page">
      <header className="page-header">
        <h1>Geographic Heatmap</h1>
        <p>
          India-wide map of metropolitan crime intensity, filtered by vulnerable group and year. Circle size and colour
          reflect relative exposure.
        </p>
      </header>

      <section className="controls-row">
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

      <section className="map-section">
        <AnyMapContainer center={[22.5, 79]} zoom={4.8} className="map-root" scrollWheelZoom={false}>
          <AnyTileLayer
            attribution="&copy; OpenStreetMap contributors"
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {filtered.map((d) => {
            const intensity = maxCrime ? d.Crime_Rate / maxCrime : 0
            const radius = 8 + 24 * intensity
            const color = intensity > 0.66 ? '#ff5252' : intensity > 0.33 ? '#ffb74d' : '#64b5f6'
            return (
              <AnyCircleMarker
                key={`${d.City}-${d.Year}-${d.Group}`}
                center={[d.Latitude, d.Longitude]}
                radius={radius}
                pathOptions={{ color, fillColor: color, fillOpacity: 0.7 }}
              >
                <Tooltip>
                  <div>
                    <strong>{d.City}</strong>
                    <br />Year: {d.Year}
                    <br />Group: {d.Group}
                    <br />Crime rate: {d.Crime_Rate}
                  </div>
                </Tooltip>
              </AnyCircleMarker>
            )
          })}
        </AnyMapContainer>
      </section>
    </div>
  )
}

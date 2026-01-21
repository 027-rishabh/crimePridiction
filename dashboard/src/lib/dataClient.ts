import Papa from 'papaparse'

export type CityYearGroupRecord = {
  City: string
  Year: number
  Group: string
  Crime_Rate: number
  YoY_Change?: number | null
  Crime_Rate_Normalized?: number | null
}

export type DataHealthSummary = {
  rows: number
  columns: number
  missing_by_column: Record<string, number>
  duplicate_rows: number
  outliers: {
    iqr_count: number
    zscore_count: number
  }
  cities: number
  years: number[]
  groups: string[]
}

export type PolicyInsights = {
  top_high_risk_cities: { City: string; CVI_Overall: number }[]
  least_risk_cities: { City: string; CVI_Overall: number }[]
  most_vulnerable_groups: { Group: string; Group_Vulnerability_Score: number }[]
  n_cities: number
  interpretation_notes: string[]
}

function csvUrl(name: string) {
  return `/data/${name}`
}

function jsonUrl(name: string) {
  return `/data/${name}`
}

export async function fetchCsv<T = Record<string, unknown>>(name: string): Promise<T[]> {
  const res = await fetch(csvUrl(name))
  const text = await res.text()

  return new Promise<T[]>((resolve, reject) => {
    Papa.parse<T>(text, {
      header: true,
      dynamicTyping: true,
      complete: (result) => {
        resolve(result.data.filter((d) => Object.keys(d as object).length > 0))
      },
      error: (error: unknown) => reject(error),
    })
  })
}

export async function fetchJson<T = unknown>(name: string): Promise<T> {
  const res = await fetch(jsonUrl(name))
  if (!res.ok) {
    throw new Error(`Failed to fetch ${name}: ${res.status}`)
  }
  return (await res.json()) as T
}

export async function loadCityYearGroupTrends(): Promise<CityYearGroupRecord[]> {
  return fetchCsv<CityYearGroupRecord>('city_year_group_trends.csv')
}

export async function loadDataHealthSummary(): Promise<DataHealthSummary> {
  return fetchJson<DataHealthSummary>('data_health_summary.json')
}

export async function loadPolicyInsights(): Promise<PolicyInsights> {
  return fetchJson<PolicyInsights>('policy_insights.json')
}

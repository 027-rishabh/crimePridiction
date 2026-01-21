# Crime Vulnerability Analysis

Research-grade exploratory data analysis and visualization of crime exposure across vulnerable groups in Indian metropolitan cities (2021–2023).

## Project Structure

- `data/` – Source CSVs and modeling scripts.
  - `data/modeling/masterData.csv` – Long-format master dataset used by the analysis pipeline.
- `analysis/` – Python analysis package.
  - `config.py` – Paths, years, groups, CVI weights, and normalization settings.
  - `loaders.py` – Data loading and basic shape checks.
  - `validation.py` – Missing values, duplicates, and outlier detection.
  - `eda.py` – Core EDA: trends, comparisons, volatility.
  - `cvi.py` – Crime Vulnerability Index construction.
  - `geo.py` – Static geographic enrichment for Indian metros.
  - `ranking.py` – City and group rankings plus policy insights.
  - `pipeline.py` – Orchestrates the full end-to-end analysis.
- `analysis/output/` – Derived CSV/JSON files for the dashboard.
- `dashboard/` – React + TypeScript dashboard (Vite).

## 1. Python Environment & Analysis Pipeline

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\\Scripts\\activate  # Windows PowerShell
```

2. Install core Python dependencies (pandas, numpy, etc. – adjust as needed if you expand the analysis):

```bash
pip install pandas numpy
```

3. Run the analysis pipeline from the project root:

```bash
python -m analysis.pipeline --normalization minmax
```

This will:

- Validate the master dataset and write:
  - `analysis/output/data_health_summary.json`
  - `analysis/output/data_health_issues.csv`
  - `analysis/output/feature_distributions.json`
- Compute core EDA products:
  - `analysis/output/city_year_group_trends.csv`
  - `analysis/output/group_comparison_city_agg.csv`
  - `analysis/output/city_group_volatility.csv`
  - `analysis/output/crime_trend_summary.json`
- Compute the Crime Vulnerability Index (CVI):
  - `analysis/output/city_year_group_normalized.csv`
  - `analysis/output/cvi_city_year.csv`
  - `analysis/output/cvi_city_overall.csv`
  - `analysis/output/cvi_sensitivity.json`
- Enrich with geography:
  - `analysis/output/geo_city_group_year.csv`
  - `analysis/output/geo_cities.csv`
  - `analysis/output/geo_cities.json`
- Build rankings and policy-oriented summaries:
  - `analysis/output/rankings_cities.csv`
  - `analysis/output/rankings_cities_overall.csv`
  - `analysis/output/rankings_groups.csv`
  - `analysis/output/policy_insights.json`

You can switch to z-score based normalization for experimentation:

```bash
python -m analysis.pipeline --normalization zscore
```

## 2. Frontend Dashboard

The `dashboard/` directory contains a Vite + React + TypeScript app that consumes the precomputed analysis outputs.

### Install Node dependencies

From the `dashboard/` directory:

```bash
cd dashboard
npm install
```

(If you have just cloned the repo and not run the analysis yet, run the Python pipeline first.)

### Make analysis outputs available to the dashboard

Copy the contents of `analysis/output/` into the dashboard's `public/data/` folder:

```bash
mkdir -p public/data
cp ../analysis/output/* public/data/
```

The React app expects these files under `/data/` at runtime.

### Run the development server

```bash
npm run dev
```

Then open the URL printed in the terminal (by default `http://localhost:5173/`).

### Build for production

```bash
npm run build
npm run preview
```

This runs a TypeScript check and bundles the dashboard into the `dist/` directory, then serves a local preview.

## 3. Dashboard Pages & Interpretation

The dashboard is organised into pages that align with the research workflow described in the project instructions:

1. **Overview** – High-level coverage, top high-risk cities, most vulnerable groups, and aggregate time trend.
2. **Data Health** – Data sanity checks (missing values, duplicates, basic distribution diagnostics).
3. **Trends & Comparisons** – Time-series for a chosen city and group plus cross-city comparisons for a given year.
4. **Geographic Heatmap** – India map with circle markers sized/coloured by crime intensity for selected year and group.
5. **Vulnerability Index (CVI)** – CVI evolution over time per city and cross-sectional CVI ranking by year.
6. **Rankings & Insights** – Tabular rankings of cities and vulnerable groups plus policy-relevant summary bullets.

Each page includes short explanatory text suitable for use in a methods/results section of an academic write-up.

## 4. Extensibility Notes

- **Weights and normalization**: Adjust group weights or the default normalization strategy in `analysis/config.py`.
- **Additional indicators**: If new socio-legal indicators become available, extend `masterData.csv` and the
  computations in `analysis/eda.py` and `analysis/cvi.py` to incorporate them.
- **Styling and UX**: The dashboard uses a minimalist dark theme with accent colours for severity; you can extend
  styles in `dashboard/src/App.css` and global rules in `dashboard/src/index.css`.

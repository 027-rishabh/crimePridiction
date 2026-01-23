import './App.css'
import { NavLink, Route, Routes } from 'react-router-dom'
import { OverviewPage } from './pages/OverviewPage'
import { DataHealthPage } from './pages/DataHealthPage'
import { TrendsPage } from './pages/TrendsPage'
import { HeatmapPage } from './pages/HeatmapPage'
import { CviPage } from './pages/CviPage'
import { RankingsPage } from './pages/RankingsPage'
import { GlobalFilterBar } from './components/GlobalFilterBar'
import { FiltersProvider } from './lib/filtersContext'

function App() {
  return (
    <FiltersProvider>
      <div className="app-root">
        <aside className="sidebar">
        <div className="brand">
          <span className="brand-title">Crime Vulnerability Analysis</span>
          <span className="brand-subtitle">Indian Metropolitan Cities, 2021–2023</span>
        </div>
        <nav className="nav">
          <NavLink to="/" end className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            Overview
          </NavLink>
          <NavLink to="/data-health" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            Data Health
          </NavLink>
          <NavLink to="/trends" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            Trends &amp; Comparisons
          </NavLink>
          <NavLink to="/heatmap" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            Geographic Heatmap
          </NavLink>
          <NavLink to="/cvi" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            Vulnerability Index
          </NavLink>
          <NavLink to="/rankings" className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            Rankings &amp; Insights
          </NavLink>
        </nav>
      </aside>
      <main className="content">
        <GlobalFilterBar />
        <Routes>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/data-health" element={<DataHealthPage />} />
          <Route path="/trends" element={<TrendsPage />} />
          <Route path="/heatmap" element={<HeatmapPage />} />
          <Route path="/cvi" element={<CviPage />} />
          <Route path="/rankings" element={<RankingsPage />} />
        </Routes>
      </main>
    </div>
    </FiltersProvider>
  )
}

export default App

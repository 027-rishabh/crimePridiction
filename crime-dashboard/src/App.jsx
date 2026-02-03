import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/layout/Navbar';
import Overview from './pages/Overview';
import ModelComparison from './pages/ModelComparison';
import FairnessAnalysis from './pages/FairnessAnalysis';
import GeographicView from './pages/GeographicView';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <main className="max-w-7xl mx-auto">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/models" element={<ModelComparison />} />
            <Route path="/fairness" element={<FairnessAnalysis />} />
            <Route path="/geographic" element={<GeographicView />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import MetricCard from '../components/cards/MetricCard';
import ModelComparisonChart from '../components/charts/ModelComparisonChart';
import { transformModelComparison } from '../utils/dataTransformers';

const Overview = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get('/data/react_dashboard_data.json')
      .then(response => {
        setData(response.data);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error loading data:', error);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-xl">Loading...</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-xl text-red-600">Error loading data</div>
      </div>
    );
  }

  const modelData = transformModelComparison(data);
  const bestModel = modelData.reduce((min, model) => 
    model.mae < min.mae ? model : min
  );

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Dashboard Overview</h1>
      
      {/* Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <MetricCard
          title="Best Model (Lowest MAE)"
          value={bestModel.mae}
          subtitle={bestModel.model}
          icon="🏆"
          color="blue"
          format="number"
        />
        <MetricCard
          title="Models Evaluated"
          value={modelData.length}
          subtitle="Baseline comparisons"
          icon="📊"
          color="green"
          format="number"
        />
        <MetricCard
          title="Lowest Fairness Gap"
          value={Math.min(...modelData.map(m => m.fairnessGap))}
          subtitle="Across all groups"
          icon="⚖️"
          color="yellow"
          format="number"
        />
        <MetricCard
          title="Test Year"
          value={data.metadata.test_year}
          subtitle="Evaluation period"
          icon="📅"
          color="blue"
          format="number"
        />
      </div>

      {/* Model Comparison Chart */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h2 className="text-2xl font-bold mb-4">Model Performance (MAE)</h2>
        <ModelComparisonChart data={modelData} metric="mae" />
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-4">Model Performance (R²)</h2>
        <ModelComparisonChart data={modelData} metric="r2" />
      </div>
    </div>
  );
};

export default Overview;
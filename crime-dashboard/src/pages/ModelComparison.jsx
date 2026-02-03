import React from 'react';

const ModelComparison = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Model Comparison</h1>
      <p className="mb-4">This page displays detailed model comparison metrics including MAE, RMSE, R² scores, and training times.</p>
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4">Coming Soon: Advanced Model Comparison Features</h2>
        <ul className="list-disc pl-6 space-y-2">
          <li>Detailed model performance charts</li>
          <li>Side-by-side model comparisons</li>
          <li>Training time analysis</li>
          <li>Hyperparameter tuning results</li>
        </ul>
      </div>
    </div>
  );
};

export default ModelComparison;
/**
 * Transform raw JSON data for chart consumption
 */

export const transformModelComparison = (data) => {
  if (!data?.model_comparison) return [];
  
  return data.model_comparison.map(model => ({
    model: model.model,
    mae: parseFloat(model.mae.toFixed(2)),
    rmse: parseFloat(model.rmse.toFixed(2)),
    r2: parseFloat(model.r2.toFixed(4)),
    fairnessGap: parseFloat(model.fairness_gap.toFixed(2)),
    fairnessRatio: parseFloat(model.fairness_ratio.toFixed(2)),
    trainingTime: parseFloat(model.training_time.toFixed(2)),
  }));
};

export const transformFairnessBreakdown = (data) => {
  if (!data?.fairness_breakdown) return {};
  
  const breakdown = {};
  Object.entries(data.fairness_breakdown).forEach(([model, groups]) => {
    breakdown[model] = Object.entries(groups).map(([group, metrics]) => ({
      group,
      mae: parseFloat(metrics.mae?.toFixed(2) || 0),
      rmse: parseFloat(metrics.rmse?.toFixed(2) || 0),
      r2: parseFloat(metrics.r2?.toFixed(4) || 0),
      count: metrics.count || 0,
    }));
  });
  
  return breakdown;
};

export const transformGeographicData = (data) => {
  if (!data?.geographic_distribution) return [];
  
  return data.geographic_distribution.map(state => ({
    state: state.state,
    totalCrimes: state.total_crimes,
    numDistricts: state.num_districts,
    crimesPerDistrict: (state.total_crimes / state.num_districts).toFixed(2),
  }));
};

export const transformPredictions = (data, modelName) => {
  if (!data?.predictions?.[modelName]) return [];
  
  return data.predictions[modelName].map(pred => ({
    state: pred.state_name,
    district: pred.district_name,
    group: pred.protected_group,
    year: pred.year,
    actual: pred.actual,
    predicted: parseFloat(pred.predicted.toFixed(2)),
    error: parseFloat((pred.actual - pred.predicted).toFixed(2)),
    absError: Math.abs(pred.actual - pred.predicted),
  }));
};

export const calculateGroupAggregates = (predictions) => {
  const groups = {};
  
  predictions.forEach(pred => {
    if (!groups[pred.group]) {
      groups[pred.group] = {
        group: pred.group,
        totalActual: 0,
        totalPredicted: 0,
        count: 0,
        errors: [],
      };
    }
    
    groups[pred.group].totalActual += pred.actual;
    groups[pred.group].totalPredicted += pred.predicted;
    groups[pred.group].count += 1;
    groups[pred.group].errors.push(pred.absError);
  });

  return Object.values(groups).map(g => ({
    group: g.group,
    avgActual: (g.totalActual / g.count).toFixed(2),
    avgPredicted: (g.totalPredicted / g.count).toFixed(2),
    mae: (g.errors.reduce((a, b) => a + b, 0) / g.count).toFixed(2),
    count: g.count,
  }));
};
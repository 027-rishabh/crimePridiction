/**
 * Consistent color schemes for charts
 */

export const modelColors = {
  'SARIMA': '#ef4444',      // Red
  'Prophet': '#f97316',     // Orange
  'Random Forest': '#10b981', // Green
  'XGBoost': '#3b82f6',     // Blue
  'CNN-LSTM': '#8b5cf6',    // Purple
  'Transformer': '#ec4899', // Pink
  'FC-MT-LSTM': '#06b6d4',  // Cyan
};

export const groupColors = {
  'SC': '#3b82f6',      // Blue
  'ST': '#10b981',      // Green
  'Women': '#ec4899',   // Pink
  'Children': '#f59e0b', // Amber
};

export const metricColors = {
  mae: '#3b82f6',
  rmse: '#10b981',
  r2: '#8b5cf6',
  fairnessGap: '#ef4444',
};

export const d3ColorScale = (domain) => {
  const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];
  return (value) => {
    const index = domain.indexOf(value);
    return colors[index % colors.length];
  };
};

// Sequential color scale for heat maps
export const heatMapColors = [
  '#f0f9ff', '#e0f2fe', '#bae6fd', '#7dd3fc', 
  '#38bdf8', '#0ea5e9', '#0284c7', '#0369a1'
];
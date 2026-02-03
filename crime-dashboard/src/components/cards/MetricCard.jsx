import React from 'react';
import { formatNumber, formatPercentage } from '../utils/formatters';

const MetricCard = ({ 
  title, 
  value, 
  subtitle, 
  icon, 
  trend,
  trendDirection,
  color = 'blue',
  format = 'number' 
}) => {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-600 border-blue-200',
    green: 'bg-green-50 text-green-600 border-green-200',
    red: 'bg-red-50 text-red-600 border-red-200',
    yellow: 'bg-yellow-50 text-yellow-600 border-yellow-200',
  };

  const formatValue = () => {
    switch (format) {
      case 'percentage':
        return formatPercentage(value);
      case 'number':
      default:
        return formatNumber(value, 2);
    }
  };

  return (
    <div className={`p-6 rounded-lg border-2 ${colorClasses[color]}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-600">{title}</span>
        {icon && <span className="text-2xl">{icon}</span>}
      </div>
      
      <div className="text-3xl font-bold mb-1">
        {formatValue()}
      </div>
      
      {subtitle && (
        <div className="text-sm text-gray-500">{subtitle}</div>
      )}
      
      {trend && (
        <div className={`text-xs mt-2 ${trendDirection === 'up' ? 'text-green-600' : 'text-red-600'}`}>
          {trendDirection === 'up' ? '↑' : '↓'} {trend}
        </div>
      )}
    </div>
  );
};

export default MetricCard;
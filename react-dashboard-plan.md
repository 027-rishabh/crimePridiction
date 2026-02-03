# React Dashboard Implementation Plan
## Crime Prediction & Fairness Analysis Visualization

---

## 🎯 PROJECT OVERVIEW

**Goal**: Build a production-ready React dashboard to visualize crime prediction model results, fairness metrics, and geographic distribution using D3.js and SVG.

**Tech Stack**:
- React 18+ (with Hooks)
- D3.js v7 (for data visualization)
- Tailwind CSS (for styling)
- Recharts (for quick standard charts)
- React Router (for navigation)
- Axios (for data fetching)

**Key Features**:
1. Model comparison charts (MAE, RMSE, R², Fairness Gap)
2. Per-group performance breakdown
3. Geographic heat maps (state/district-wise)
4. Time series predictions
5. Fairness metrics dashboard
6. Interactive filters and controls

---

## 📁 PROJECT STRUCTURE

```
crime-dashboard/
├── public/
│   ├── index.html
│   └── data/
│       └── react_dashboard_data.json  # Your existing data
├── src/
│   ├── components/
│   │   ├── charts/
│   │   │   ├── ModelComparisonChart.jsx
│   │   │   ├── FairnessGapChart.jsx
│   │   │   ├── PerGroupBarChart.jsx
│   │   │   ├── GeographicHeatMap.jsx
│   │   │   ├── TimeSeriesChart.jsx
│   │   │   ├── ScatterPlot.jsx
│   │   │   └── ParetoFrontier.jsx
│   │   ├── layout/
│   │   │   ├── Navbar.jsx
│   │   │   ├── Sidebar.jsx
│   │   │   └── Footer.jsx
│   │   ├── filters/
│   │   │   ├── ModelFilter.jsx
│   │   │   ├── GroupFilter.jsx
│   │   │   └── StateFilter.jsx
│   │   └── cards/
│   │       ├── MetricCard.jsx
│   │       └── SummaryCard.jsx
│   ├── pages/
│   │   ├── Overview.jsx
│   │   ├── ModelComparison.jsx
│   │   ├── FairnessAnalysis.jsx
│   │   ├── GeographicView.jsx
│   │   └── Predictions.jsx
│   ├── hooks/
│   │   ├── useChartData.js
│   │   └── useWindowSize.js
│   ├── utils/
│   │   ├── dataTransformers.js
│   │   ├── colorPalette.js
│   │   └── formatters.js
│   ├── App.jsx
│   ├── index.jsx
│   └── index.css
├── package.json
└── tailwind.config.js
```

---

## 🚀 PHASE 1: PROJECT SETUP (Day 1)

### **Step 1.1: Initialize React App**

```bash
# Create React app with Vite (faster than CRA)
npm create vite@latest crime-dashboard -- --template react
cd crime-dashboard

# Install dependencies
npm install

# Install chart libraries
npm install d3 recharts

# Install UI libraries
npm install react-router-dom axios

# Install Tailwind CSS
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

### **Step 1.2: Configure Tailwind CSS**

**File**: `tailwind.config.js`

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          100: '#dbeafe',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        },
        success: '#10b981',
        warning: '#f59e0b',
        danger: '#ef4444',
      },
    },
  },
  plugins: [],
}
```

**File**: `src/index.css`

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: #f1f5f9;
}

::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}
```

### **Step 1.3: Project Structure Setup**

```bash
# Create directory structure
mkdir -p src/components/charts
mkdir -p src/components/layout
mkdir -p src/components/filters
mkdir -p src/components/cards
mkdir -p src/pages
mkdir -p src/hooks
mkdir -p src/utils
mkdir -p public/data

# Copy your data file
cp react_dashboard_data.json public/data/
```

---

## 📊 PHASE 2: DATA UTILITIES (Day 1-2)

### **Step 2.1: Data Transformer**

**File**: `src/utils/dataTransformers.js`

```javascript
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
```

### **Step 2.2: Color Palette**

**File**: `src/utils/colorPalette.js`

```javascript
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
```

### **Step 2.3: Formatters**

**File**: `src/utils/formatters.js`

```javascript
/**
 * Number and text formatting utilities
 */

export const formatNumber = (num, decimals = 2) => {
  if (num === null || num === undefined) return 'N/A';
  return Number(num).toFixed(decimals);
};

export const formatLargeNumber = (num) => {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
  return num.toString();
};

export const formatPercentage = (num, decimals = 1) => {
  if (num === null || num === undefined) return 'N/A';
  return `${(num * 100).toFixed(decimals)}%`;
};

export const formatTime = (seconds) => {
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${(seconds / 60).toFixed(1)}m`;
};

export const truncateText = (text, maxLength = 20) => {
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength)}...`;
};
```

---

## 🎨 PHASE 3: REUSABLE COMPONENTS (Day 2-3)

### **Step 3.1: Metric Card Component**

**File**: `src/components/cards/MetricCard.jsx`

```jsx
import React from 'react';
import { formatNumber, formatPercentage } from '../../utils/formatters';

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
```

### **Step 3.2: Layout Components**

**File**: `src/components/layout/Navbar.jsx`

```jsx
import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <nav className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <h1 className="text-2xl font-bold text-blue-600">
              Crime Prediction Dashboard
            </h1>
          </div>
          
          <div className="flex items-center space-x-4">
            <Link 
              to="/" 
              className="px-3 py-2 rounded-md text-sm font-medium text-gray-700 hover:text-blue-600 hover:bg-blue-50"
            >
              Overview
            </Link>
            <Link 
              to="/models" 
              className="px-3 py-2 rounded-md text-sm font-medium text-gray-700 hover:text-blue-600 hover:bg-blue-50"
            >
              Models
            </Link>
            <Link 
              to="/fairness" 
              className="px-3 py-2 rounded-md text-sm font-medium text-gray-700 hover:text-blue-600 hover:bg-blue-50"
            >
              Fairness
            </Link>
            <Link 
              to="/geographic" 
              className="px-3 py-2 rounded-md text-sm font-medium text-gray-700 hover:text-blue-600 hover:bg-blue-50"
            >
              Geographic
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
```

**File**: `src/components/layout/Sidebar.jsx`

```jsx
import React from 'react';

const Sidebar = ({ children }) => {
  return (
    <aside className="w-64 bg-white border-r border-gray-200 h-screen sticky top-0 overflow-y-auto">
      <div className="p-4">
        {children}
      </div>
    </aside>
  );
};

export default Sidebar;
```

---

## 📈 PHASE 4: D3.JS CHART COMPONENTS (Day 3-5)

### **Step 4.1: Model Comparison Bar Chart**

**File**: `src/components/charts/ModelComparisonChart.jsx`

```jsx
import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { modelColors } from '../../utils/colorPalette';

const ModelComparisonChart = ({ data, metric = 'mae' }) => {
  const svgRef = useRef();
  const containerRef = useRef();

  useEffect(() => {
    if (!data || data.length === 0) return;

    // Clear previous chart
    d3.select(svgRef.current).selectAll('*').remove();

    // Dimensions
    const containerWidth = containerRef.current.offsetWidth;
    const margin = { top: 20, right: 30, bottom: 60, left: 60 };
    const width = containerWidth - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    // Create SVG
    const svg = d3.select(svgRef.current)
      .attr('width', containerWidth)
      .attr('height', 400)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const x = d3.scaleBand()
      .domain(data.map(d => d.model))
      .range([0, width])
      .padding(0.2);

    const y = d3.scaleLinear()
      .domain([0, d3.max(data, d => d[metric]) * 1.1])
      .range([height, 0]);

    // X Axis
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x))
      .selectAll('text')
      .attr('transform', 'rotate(-45)')
      .style('text-anchor', 'end')
      .style('font-size', '12px');

    // Y Axis
    svg.append('g')
      .call(d3.axisLeft(y))
      .style('font-size', '12px');

    // Y Axis Label
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left)
      .attr('x', 0 - (height / 2))
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .style('font-size', '14px')
      .style('font-weight', 'bold')
      .text(metric.toUpperCase());

    // Bars
    svg.selectAll('.bar')
      .data(data)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', d => x(d.model))
      .attr('y', height)
      .attr('width', x.bandwidth())
      .attr('height', 0)
      .attr('fill', d => modelColors[d.model] || '#3b82f6')
      .attr('opacity', 0.8)
      .on('mouseover', function(event, d) {
        d3.select(this).attr('opacity', 1);
        
        // Tooltip
        const tooltip = d3.select('body')
          .append('div')
          .attr('class', 'tooltip')
          .style('position', 'absolute')
          .style('background', 'rgba(0, 0, 0, 0.8)')
          .style('color', 'white')
          .style('padding', '8px')
          .style('border-radius', '4px')
          .style('font-size', '12px')
          .style('pointer-events', 'none')
          .style('z-index', '1000')
          .html(`
            <strong>${d.model}</strong><br/>
            ${metric.toUpperCase()}: ${d[metric].toFixed(2)}
          `);

        tooltip
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 20) + 'px');
      })
      .on('mouseout', function() {
        d3.select(this).attr('opacity', 0.8);
        d3.selectAll('.tooltip').remove();
      })
      .transition()
      .duration(800)
      .attr('y', d => y(d[metric]))
      .attr('height', d => height - y(d[metric]));

    // Value labels on top of bars
    svg.selectAll('.label')
      .data(data)
      .enter()
      .append('text')
      .attr('class', 'label')
      .attr('x', d => x(d.model) + x.bandwidth() / 2)
      .attr('y', d => y(d[metric]) - 5)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('font-weight', 'bold')
      .style('fill', '#374151')
      .text(d => d[metric].toFixed(2))
      .style('opacity', 0)
      .transition()
      .delay(800)
      .duration(400)
      .style('opacity', 1);

  }, [data, metric]);

  return (
    <div ref={containerRef} className="w-full">
      <svg ref={svgRef}></svg>
    </div>
  );
};

export default ModelComparisonChart;
```

### **Step 4.2: Fairness Gap Chart**

**File**: `src/components/charts/FairnessGapChart.jsx`

```jsx
import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { groupColors } from '../../utils/colorPalette';

const FairnessGapChart = ({ data }) => {
  const svgRef = useRef();
  const containerRef = useRef();

  useEffect(() => {
    if (!data || data.length === 0) return;

    // Clear previous chart
    d3.select(svgRef.current).selectAll('*').remove();

    // Dimensions
    const containerWidth = containerRef.current.offsetWidth;
    const margin = { top: 20, right: 120, bottom: 60, left: 60 };
    const width = containerWidth - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    // Create SVG
    const svg = d3.select(svgRef.current)
      .attr('width', containerWidth)
      .attr('height', 400)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Get all groups
    const groups = Array.from(new Set(data.flatMap(d => d.map(g => g.group))));

    // Scales
    const x = d3.scaleBand()
      .domain(data.map((_, i) => `Model ${i + 1}`))
      .range([0, width])
      .padding(0.2);

    const y = d3.scaleLinear()
      .domain([0, d3.max(data.flat(), d => d.mae) * 1.1])
      .range([height, 0]);

    const xGroup = d3.scaleBand()
      .domain(groups)
      .range([0, x.bandwidth()])
      .padding(0.05);

    // X Axis
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x))
      .style('font-size', '12px');

    // Y Axis
    svg.append('g')
      .call(d3.axisLeft(y))
      .style('font-size', '12px');

    // Y Axis Label
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left)
      .attr('x', 0 - (height / 2))
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .style('font-size', '14px')
      .style('font-weight', 'bold')
      .text('MAE');

    // Grouped bars
    svg.selectAll('.model-group')
      .data(data)
      .enter()
      .append('g')
      .attr('class', 'model-group')
      .attr('transform', (d, i) => `translate(${x(`Model ${i + 1}`)},0)`)
      .selectAll('rect')
      .data(d => d)
      .enter()
      .append('rect')
      .attr('x', d => xGroup(d.group))
      .attr('y', height)
      .attr('width', xGroup.bandwidth())
      .attr('height', 0)
      .attr('fill', d => groupColors[d.group] || '#3b82f6')
      .attr('opacity', 0.8)
      .on('mouseover', function(event, d) {
        d3.select(this).attr('opacity', 1);
        
        const tooltip = d3.select('body')
          .append('div')
          .attr('class', 'tooltip')
          .style('position', 'absolute')
          .style('background', 'rgba(0, 0, 0, 0.8)')
          .style('color', 'white')
          .style('padding', '8px')
          .style('border-radius', '4px')
          .style('font-size', '12px')
          .style('pointer-events', 'none')
          .style('z-index', '1000')
          .html(`
            <strong>${d.group}</strong><br/>
            MAE: ${d.mae.toFixed(2)}<br/>
            R²: ${d.r2.toFixed(4)}<br/>
            Count: ${d.count}
          `);

        tooltip
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 20) + 'px');
      })
      .on('mouseout', function() {
        d3.select(this).attr('opacity', 0.8);
        d3.selectAll('.tooltip').remove();
      })
      .transition()
      .duration(800)
      .attr('y', d => y(d.mae))
      .attr('height', d => height - y(d.mae));

    // Legend
    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${width + 20}, 0)`);

    groups.forEach((group, i) => {
      const legendRow = legend.append('g')
        .attr('transform', `translate(0, ${i * 25})`);

      legendRow.append('rect')
        .attr('width', 15)
        .attr('height', 15)
        .attr('fill', groupColors[group]);

      legendRow.append('text')
        .attr('x', 20)
        .attr('y', 12)
        .style('font-size', '12px')
        .text(group);
    });

  }, [data]);

  return (
    <div ref={containerRef} className="w-full">
      <svg ref={svgRef}></svg>
    </div>
  );
};

export default FairnessGapChart;
```

### **Step 4.3: Time Series Line Chart**

**File**: `src/components/charts/TimeSeriesChart.jsx`

```jsx
import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const TimeSeriesChart = ({ data, xKey, yKey, title }) => {
  const svgRef = useRef();
  const containerRef = useRef();

  useEffect(() => {
    if (!data || data.length === 0) return;

    // Clear previous chart
    d3.select(svgRef.current).selectAll('*').remove();

    // Dimensions
    const containerWidth = containerRef.current.offsetWidth;
    const margin = { top: 20, right: 30, bottom: 60, left: 60 };
    const width = containerWidth - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    // Create SVG
    const svg = d3.select(svgRef.current)
      .attr('width', containerWidth)
      .attr('height', 400)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const x = d3.scaleLinear()
      .domain(d3.extent(data, d => d[xKey]))
      .range([0, width]);

    const y = d3.scaleLinear()
      .domain([0, d3.max(data, d => d[yKey]) * 1.1])
      .range([height, 0]);

    // Line generator
    const line = d3.line()
      .x(d => x(d[xKey]))
      .y(d => y(d[yKey]))
      .curve(d3.curveMonotoneX);

    // X Axis
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x).ticks(10).tickFormat(d3.format('d')))
      .style('font-size', '12px');

    // Y Axis
    svg.append('g')
      .call(d3.axisLeft(y))
      .style('font-size', '12px');

    // X Axis Label
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height + margin.bottom - 10)
      .style('text-anchor', 'middle')
      .style('font-size', '14px')
      .style('font-weight', 'bold')
      .text(xKey);

    // Y Axis Label
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left)
      .attr('x', 0 - (height / 2))
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .style('font-size', '14px')
      .style('font-weight', 'bold')
      .text(yKey);

    // Line path
    const path = svg.append('path')
      .datum(data)
      .attr('fill', 'none')
      .attr('stroke', '#3b82f6')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Animate line drawing
    const totalLength = path.node().getTotalLength();
    path
      .attr('stroke-dasharray', totalLength + ' ' + totalLength)
      .attr('stroke-dashoffset', totalLength)
      .transition()
      .duration(2000)
      .ease(d3.easeLinear)
      .attr('stroke-dashoffset', 0);

    // Data points
    svg.selectAll('.dot')
      .data(data)
      .enter()
      .append('circle')
      .attr('class', 'dot')
      .attr('cx', d => x(d[xKey]))
      .attr('cy', d => y(d[yKey]))
      .attr('r', 0)
      .attr('fill', '#3b82f6')
      .on('mouseover', function(event, d) {
        d3.select(this).attr('r', 6);
        
        const tooltip = d3.select('body')
          .append('div')
          .attr('class', 'tooltip')
          .style('position', 'absolute')
          .style('background', 'rgba(0, 0, 0, 0.8)')
          .style('color', 'white')
          .style('padding', '8px')
          .style('border-radius', '4px')
          .style('font-size', '12px')
          .style('pointer-events', 'none')
          .style('z-index', '1000')
          .html(`
            ${xKey}: ${d[xKey]}<br/>
            ${yKey}: ${d[yKey].toFixed(2)}
          `);

        tooltip
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 20) + 'px');
      })
      .on('mouseout', function() {
        d3.select(this).attr('r', 4);
        d3.selectAll('.tooltip').remove();
      })
      .transition()
      .delay(2000)
      .duration(400)
      .attr('r', 4);

  }, [data, xKey, yKey]);

  return (
    <div ref={containerRef} className="w-full">
      <svg ref={svgRef}></svg>
    </div>
  );
};

export default TimeSeriesChart;
```

---

## 📄 PHASE 5: PAGES (Day 5-6)

### **Step 5.1: Overview Page**

**File**: `src/pages/Overview.jsx`

```jsx
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
          title="Best Model"
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
```

---

## 🔧 PHASE 6: APP ASSEMBLY (Day 6-7)

### **Step 6.1: Main App Component**

**File**: `src/App.jsx`

```jsx
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
```

### **Step 6.2: Package.json**

**File**: `package.json`

```json
{
  "name": "crime-dashboard",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.0",
    "d3": "^7.8.5",
    "recharts": "^2.10.3",
    "axios": "^1.6.2"
  },
  "devDependencies": {
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "@vitejs/plugin-react": "^4.2.1",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.32",
    "tailwindcss": "^3.3.6",
    "vite": "^5.0.8"
  }
}
```

---

## 🚀 DEPLOYMENT & TESTING (Day 7)

### **Step 7.1: Run Development Server**

```bash
npm run dev
```

Access at: `http://localhost:5173`

### **Step 7.2: Build for Production**

```bash
npm run build
```

### **Step 7.3: Test Checklist**

- [ ] All charts render correctly
- [ ] Data loads from JSON file
- [ ] Navigation between pages works
- [ ] Responsive on mobile/tablet/desktop
- [ ] Tooltips show on hover
- [ ] No console errors
- [ ] Animations smooth

---

## 📋 COMPLETE IMPLEMENTATION CHECKLIST

### **Day 1: Setup**
- [ ] Create React project with Vite
- [ ] Install all dependencies
- [ ] Configure Tailwind CSS
- [ ] Set up project structure
- [ ] Copy data file to public folder

### **Day 2: Utilities & Components**
- [ ] Create data transformers
- [ ] Create color palette
- [ ] Create formatters
- [ ] Build MetricCard component
- [ ] Build Navbar and layout components

### **Day 3-4: Charts**
- [ ] Model comparison bar chart (D3.js)
- [ ] Fairness gap grouped bar chart
- [ ] Time series line chart
- [ ] Test all chart interactivity

### **Day 5-6: Pages**
- [ ] Overview page with metric cards
- [ ] Model comparison page
- [ ] Fairness analysis page
- [ ] Geographic view page

### **Day 7: Polish & Deploy**
- [ ] Test responsiveness
- [ ] Fix any bugs
- [ ] Add loading states
- [ ] Build production version
- [ ] Deploy to hosting (Vercel/Netlify)

---

## 🎯 EXPECTED OUTCOME

**Fully functional dashboard with:**
- ✅ 4 main pages (Overview, Models, Fairness, Geographic)
- ✅ 10+ interactive D3.js charts
- ✅ Real-time data from your JSON file
- ✅ Responsive design (mobile-first)
- ✅ Production-ready code
- ✅ <2s load time

**Timeline**: 7 days (full-time) or 2-3 weeks (part-time)

Ready to start building? Let me know which phase you want to tackle first! 🚀

import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const FairnessAnalysisChart = ({ data }) => {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={data}
        layout="horizontal"
        margin={{
          top: 20,
          right: 30,
          left: 20,
          bottom: 50,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" />
        <YAxis dataKey="model" type="category" width={100} />
        <Tooltip />
        <Legend />
        <Bar dataKey="fairness_gap" name="Fairness Gap" fill="#4caf50" />
      </BarChart>
    </ResponsiveContainer>
  );
};

export default FairnessAnalysisChart;
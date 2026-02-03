import React, { useState, useEffect } from 'react';
import { Container, Typography, Grid, Paper, Box, CircularProgress, Alert } from '@mui/material';
import { BarChart, Bar, PieChart, Pie, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import './App.css';

function App() {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Colors for charts
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

  useEffect(() => {
    // Fetch dashboard data from the results directory
    const fetchData = async () => {
      try {
        const response = await fetch('./results/react_dashboard_data.json');
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();
        
        // Set the FC-MT-LSTM model first in the comparison array
        const fcmtlstmIndex = data.model_comparison.findIndex(item => item.model === 'FC-MT-LSTM');
        if (fcmtlstmIndex !== -1) {
          const fcmtlstmModel = data.model_comparison.splice(fcmtlstmIndex, 1)[0];
          data.model_comparison.unshift(fcmtlstmModel);
        }
        
        setDashboardData(data);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError(err.message);
        setLoading(false);
        
        // If fetching fails, use sample data
        const sampleDashboardData = {
          metadata: {
            generated_at: new Date().toISOString(),
            num_models: 7,
            test_year: 2022
          },
          model_comparison: [
            { model: 'FC-MT-LSTM', mae: 6.52, rmse: 8.23, r2: 0.752, fairness_gap: 0.85, fairness_ratio: 1.15, training_time: 180 },
            { model: 'XGBoost', mae: 18.24, rmse: 24.15, r2: 0.421, fairness_gap: 5.67, fairness_ratio: 2.34, training_time: 45 },
            { model: 'Random Forest', mae: 16.78, rmse: 21.98, r2: 0.543, fairness_gap: 4.21, fairness_ratio: 1.98, training_time: 32 },
            { model: 'CNN-LSTM', mae: 9.34, rmse: 12.76, r2: 0.682, fairness_gap: 2.85, fairness_ratio: 1.76, training_time: 240 },
            { model: 'Transformer', mae: 8.56, rmse: 11.92, r2: 0.713, fairness_gap: 2.12, fairness_ratio: 1.54, training_time: 320 },
            { model: 'SARIMA', mae: 22.34, rmse: 28.76, r2: 0.298, fairness_gap: 7.23, fairness_ratio: 3.12, training_time: 120 },
            { model: 'Prophet', mae: 20.89, rmse: 26.54, r2: 0.342, fairness_gap: 6.78, fairness_ratio: 2.87, training_time: 90 }
          ],
          fairness_breakdown: {
            'FC-MT-LSTM': {
              'SC': { mae: 5.8, rmse: 7.45, r2: 0.78, count: 120 },
              'ST': { mae: 6.2, rmse: 7.98, r2: 0.76, count: 95 },
              'Women': { mae: 6.9, rmse: 8.67, r2: 0.73, count: 210 },
              'Children': { mae: 7.2, rmse: 8.98, r2: 0.72, count: 85 }
            },
            'XGBoost': {
              'SC': { mae: 16.5, rmse: 21.89, r2: 0.45, count: 120 },
              'ST': { mae: 20.2, rmse: 26.34, r2: 0.38, count: 95 },
              'Women': { mae: 17.9, rmse: 23.76, r2: 0.41, count: 210 },
              'Children': { mae: 21.5, rmse: 28.23, r2: 0.35, count: 85 }
            }
          },
          geographic_distribution: [
            { state: 'Maharashtra', total_crimes: 1250, num_districts: 12 },
            { state: 'Uttar Pradesh', total_crimes: 980, num_districts: 18 },
            { state: 'Delhi', total_crimes: 650, num_districts: 1 },
            { state: 'Tamil Nadu', total_crimes: 540, num_districts: 8 },
            { state: 'Karnataka', total_crimes: 480, num_districts: 9 }
          ]
        };
        setDashboardData(sampleDashboardData);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ padding: 3 }}>
        <Alert severity="error">
          Error loading dashboard data: {error}. Using sample data instead.
        </Alert>
        <Container maxWidth="xl">
          <Box sx={{ flexGrow: 1, padding: 3 }}>
            {/* Header */}
            <Box sx={{ textAlign: 'center', marginBottom: 4 }}>
              <Typography variant="h2" component="h1" gutterBottom sx={{ color: '#1a237e' }}>
                🔍 Crime Prediction Dashboard
              </Typography>
              <Typography variant="h5" sx={{ color: '#455a64' }}>
                FC-MT-LSTM: Fairness-Constrained Multi-Task LSTM Model
              </Typography>
            </Box>
            
            {/* Rest of the app with sample data would go here, but since we have loaded sample data into dashboardData, we can continue */}
          </Box>
        </Container>
      </Box>
    );
  }

  // Find FC-MT-LSTM model for metric cards
  const fcmtlstmModel = dashboardData.model_comparison.find(model => model.model === 'FC-MT-LSTM') || dashboardData.model_comparison[0];

  return (
    <Container maxWidth="xl">
      <Box sx={{ flexGrow: 1, padding: 3 }}>
        {/* Header */}
        <Box sx={{ textAlign: 'center', marginBottom: 4 }}>
          <Typography variant="h2" component="h1" gutterBottom sx={{ color: '#1a237e' }}>
            🔍 Crime Prediction Dashboard
          </Typography>
          <Typography variant="h5" sx={{ color: '#455a64' }}>
            FC-MT-LSTM: Fairness-Constrained Multi-Task LSTM Model
          </Typography>
        </Box>

        {/* Metric Cards */}
        <Grid container spacing={3} sx={{ marginBottom: 4 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Paper elevation={3} sx={{ padding: 3, backgroundColor: '#e3f2fd', borderRadius: 2 }}>
              <Typography variant="h6" color="textSecondary">Overall MAE</Typography>
              <Typography variant="h4" color="primary" fontWeight="bold">{fcmtlstmModel ? fcmtlstmModel.mae.toFixed(2) : 'N/A'}</Typography>
              {fcmtlstmModel && dashboardData.model_comparison.length > 1 && (
                <Typography variant="caption" color="success.main">↓ {(100 - (fcmtlstmModel.mae / dashboardData.model_comparison[1]?.mae) * 100).toFixed(0)}% vs {dashboardData.model_comparison[1]?.model || 'Baseline'}</Typography>
              )}
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper elevation={3} sx={{ padding: 3, backgroundColor: '#e8f5e9', borderRadius: 2 }}>
              <Typography variant="h6" color="textSecondary">Fairness Gap</Typography>
              <Typography variant="h4" color="secondary" fontWeight="bold">{fcmtlstmModel ? fcmtlstmModel.fairness_gap.toFixed(2) : 'N/A'}</Typography>
              {fcmtlstmModel && dashboardData.model_comparison.length > 1 && (
                <Typography variant="caption" color="success.main">↓ {(100 - (fcmtlstmModel.fairness_gap / dashboardData.model_comparison[1]?.fairness_gap) * 100).toFixed(0)}% vs {dashboardData.model_comparison[1]?.model || 'Baseline'}</Typography>
              )}
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper elevation={3} sx={{ padding: 3, backgroundColor: '#fff3e0', borderRadius: 2 }}>
              <Typography variant="h6" color="textSecondary">R² Score</Typography>
              <Typography variant="h4" color="warning.main" fontWeight="bold">{fcmtlstmModel ? fcmtlstmModel.r2.toFixed(3) : 'N/A'}</Typography>
              {fcmtlstmModel && dashboardData.model_comparison.length > 2 && (
                <Typography variant="caption" color="success.main">↑ {((fcmtlstmModel.r2 / dashboardData.model_comparison[2]?.r2 - 1) * 100).toFixed(0)}% vs {dashboardData.model_comparison[2]?.model || 'Baseline'}</Typography>
              )}
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper elevation={3} sx={{ padding: 3, backgroundColor: '#f3e5f5', borderRadius: 2 }}>
              <Typography variant="h6" color="textSecondary">Model Status</Typography>
              <Typography variant="h4" color="info.main" fontWeight="bold">Optimized</Typography>
              <Typography variant="caption" color="text.secondary">Balanced Accuracy & Fairness</Typography>
            </Paper>
          </Grid>
        </Grid>

        {/* Charts Section */}
        <Grid container spacing={4} sx={{ marginBottom: 4 }}>
          {/* Model Comparison Chart */}
          <Grid item xs={12} lg={8}>
            <Paper elevation={3} sx={{ padding: 2, height: 400 }}>
              <Typography variant="h6" sx={{ marginBottom: 2, color: '#1a237e' }}>
                Model Performance Comparison (MAE)
              </Typography>
              <ResponsiveContainer width="100%" height="90%">
                <BarChart
                  data={dashboardData.model_comparison}
                  margin={{
                    top: 20,
                    right: 30,
                    left: 20,
                    bottom: 50,
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="model" angle={-45} textAnchor="end" height={60} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="mae" name="Mean Absolute Error" fill="#3f51b5" />
                </BarChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>

          {/* Fairness Analysis Chart */}
          <Grid item xs={12} lg={4}>
            <Paper elevation={3} sx={{ padding: 2, height: 400 }}>
              <Typography variant="h6" sx={{ marginBottom: 2, color: '#1a237e' }}>
                Fairness Gap Comparison
              </Typography>
              <ResponsiveContainer width="100%" height="90%">
                <BarChart
                  data={dashboardData.model_comparison.slice(0, 5)}
                  layout="vertical"
                  margin={{
                    top: 20,
                    right: 30,
                    left: 50,
                    bottom: 5,
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="model" type="category" width={100} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="fairness_gap" name="Fairness Gap" fill="#f50057" />
                </BarChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
        </Grid>

        {/* Per-Group Analysis */}
        <Grid container spacing={4} sx={{ marginBottom: 4 }}>
          {/* Per-Group MAE */}
          <Grid item xs={12} lg={8}>
            <Paper elevation={3} sx={{ padding: 2 }}>
              <Typography variant="h6" sx={{ marginBottom: 2, color: '#1a237e' }}>
                Per-Group Performance (MAE) - FC-MT-LSTM
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                {dashboardData.fairness_breakdown['FC-MT-LSTM'] ? (
                  <BarChart
                    data={[
                      { name: 'SC', sc: dashboardData.fairness_breakdown['FC-MT-LSTM']?.SC?.mae },
                      { name: 'ST', st: dashboardData.fairness_breakdown['FC-MT-LSTM']?.ST?.mae },
                      { name: 'Women', women: dashboardData.fairness_breakdown['FC-MT-LSTM']?.Women?.mae },
                      { name: 'Children', children: dashboardData.fairness_breakdown['FC-MT-LSTM']?.Children?.mae }
                    ]}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    {Object.entries(dashboardData.fairness_breakdown['FC-MT-LSTM'] || {}).map(([group], index) => (
                      <Bar key={group} dataKey={group.toLowerCase()} name={group} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </BarChart>
                ) : (
                  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                    <Typography>No fairness breakdown data available for FC-MT-LSTM</Typography>
                  </Box>
                )}
              </ResponsiveContainer>
            </Paper>
          </Grid>

          {/* Geographic Distribution */}
          <Grid item xs={12} lg={4}>
            <Paper elevation={3} sx={{ padding: 2 }}>
              <Typography variant="h6" sx={{ marginBottom: 2, color: '#1a237e' }}>
                Top 5 States by Crime Volume
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                {dashboardData.geographic_distribution && dashboardData.geographic_distribution.length > 0 ? (
                  <PieChart>
                    <Pie
                      data={dashboardData.geographic_distribution}
                      cx="50%"
                      cy="50%"
                      labelLine={true}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="total_crimes"
                      nameKey="state"
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    >
                      {dashboardData.geographic_distribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => [`${value} crimes`, 'Total Crimes']} />
                  </PieChart>
                ) : (
                  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                    <Typography>No geographic distribution data available</Typography>
                  </Box>
                )}
              </ResponsiveContainer>
            </Paper>
          </Grid>
        </Grid>

        {/* Key Features */}
        <Grid container spacing={3} sx={{ marginBottom: 4 }}>
          <Grid item xs={12}>
            <Typography variant="h5" sx={{ marginBottom: 2, color: '#1a237e' }}>
              Key Features of FC-MT-LSTM
            </Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper elevation={2} sx={{ padding: 3, backgroundColor: '#e3f2fd' }}>
              <Typography variant="h6" color="primary">🔍 Multi-Task Learning</Typography>
              <Typography variant="body2">
                Separate decoders for each protected group (SC, ST, Women, Children)
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper elevation={2} sx={{ padding: 3, backgroundColor: '#e8f5e9' }}>
              <Typography variant="h6" color="secondary">⚖️ Fairness Constraint</Typography>
              <Typography variant="body2">
                Explicit penalty for differences in prediction errors across groups
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper elevation={2} sx={{ padding: 3, backgroundColor: '#fff3e0' }}>
              <Typography variant="h6" color="warning.main">🧠 Deep Architecture</Typography>
              <Typography variant="body2">
                Shared CNN-LSTM encoder with attention mechanism for interpretability
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper elevation={2} sx={{ padding: 3, backgroundColor: '#f3e5f5' }}>
              <Typography variant="h6" color="info.main">📊 Real-time Predictions</Typography>
              <Typography variant="body2">
                Interactive tool for crime rate predictions across regions and demographics
              </Typography>
            </Paper>
          </Grid>
        </Grid>

        {/* Footer */}
        <Box sx={{ textAlign: 'center', marginTop: 4, paddingTop: 2, borderTop: '1px solid #e0e0e0' }}>
          <Typography variant="body2" color="textSecondary">
            FC-MT-LSTM: Fairness-Constrained Multi-Task LSTM for Crime Prediction
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Implemented with React • Recharts • Material UI
          </Typography>
          <Typography variant="caption" color="textSecondary">
            Data generated: {dashboardData.metadata.generated_at}
          </Typography>
        </Box>
      </Box>
    </Container>
  );
}

export default App;
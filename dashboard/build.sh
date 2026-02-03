#!/bin/bash
# Build script for FC-MT-LSTM Dashboard
# Copies results data to public folder and builds the React app

echo "Preparing FC-MT-LSTM Dashboard..."

# Create results directory in public folder if it doesn't exist
mkdir -p public/results

# Copy the dashboard data to the public folder
cp ../results/react_dashboard_data.json public/results/ || echo "Warning: Could not copy results data. Make sure to run export_results_for_react.py first."

# Build the React app
npm run build

echo "Build complete! The dashboard is available in the build/ folder."
echo "To serve locally, you can use: npx serve -s build"
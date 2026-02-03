#!/bin/bash
# Startup script for FC-MT-LSTM Dashboard

echo "Starting FC-MT-LSTM Crime Prediction Dashboard..."
echo ""

# Check if data exists in public/results
if [ ! -f "public/results/react_dashboard_data.json" ]; then
    echo "No dashboard data found in public/results/"
    echo "Running data preparation script..."
    python prepare_data.py
    echo ""
fi

echo "Starting React development server..."
echo "The dashboard will be available at http://localhost:3000"
echo ""

# Start the React development server
npm start
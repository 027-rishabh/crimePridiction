#!/bin/bash
# Complete setup script for FC-MT-LSTM Dashboard

echo "============================================================="
echo "Setting up FC-MT-LSTM Crime Prediction Dashboard"
echo "============================================================="

# Check if node is installed
if ! command -v node &> /dev/null; then
    echo "✗ Node.js is not installed. Please install Node.js first."
    echo "  Visit https://nodejs.org/ to download and install."
    exit 1
else
    echo "✓ Node.js is installed: $(node --version)"
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "✗ npm is not installed. Please install Node.js (which includes npm)."
    exit 1
else
    echo "✓ npm is installed: $(npm --version)"
fi

# Navigate to dashboard directory
cd dashboard || exit

# Install dependencies
echo ""
echo "Installing dashboard dependencies..."
npm install

# Prepare data
echo ""
echo "Preparing dashboard data..."
python prepare_data.py

echo ""
echo "============================================================="
echo "Setup Complete!"
echo "============================================================="
echo ""
echo "To start the dashboard:"
echo "  1. cd dashboard"
echo "  2. ./start.sh  (or npm start)"
echo ""
echo "The dashboard will be available at http://localhost:3000"
echo ""
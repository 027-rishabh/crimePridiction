#!/usr/bin/env python3
"""
Prepare data specifically for the FC-MT-LSTM React dashboard
"""

import json
import os
from pathlib import Path


def prepare_dashboard_data():
    """Prepare dashboard data for React app"""
    print("=" * 70)
    print("PREPARING DATA FOR FC-MT-LSTM REACT DASHBOARD")
    print("=" * 70)

    # Define paths
    input_path = Path("../results/react_dashboard_data.json")
    output_path = Path("public/results/react_dashboard_data.json")
    
    # Create public/results directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load the existing data
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        print(f"✓ Loaded data from {input_path}")
        print(f"  - Number of models: {len(data.get('model_comparison', []))}")
        print(f"  - Generated at: {data.get('metadata', {}).get('generated_at', 'Unknown')}")
    except FileNotFoundError:
        print(f"✗ Error: Could not find {input_path}")
        print("  Run export_results_for_react.py first to generate the required data.")
        return
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        return
    
    # Ensure FC-MT-LSTM is at the top position in model_comparison
    model_comparison = data.get('model_comparison', [])
    fcmtlstm_idx = next((i for i, model in enumerate(model_comparison) if model['model'] == 'FC-MT-LSTM'), -1)
    
    if fcmtlstm_idx != -1:
        # Move FC-MT-LSTM to the front
        fcmtlstm_model = model_comparison.pop(fcmtlstm_idx)
        model_comparison.insert(0, fcmtlstm_model)
        data['model_comparison'] = model_comparison
        print("✓ Ensured FC-MT-LSTM model is prioritized in comparison")
    
    # Save to public directory
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved dashboard-ready data to {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        print(f"✗ Error saving data: {str(e)}")
        return
    
    print("\n" + "=" * 70)
    print("✓ DATA PREPARATION COMPLETE")
    print("=" * 70)
    print("\nTo start the dashboard:")
    print("  1. cd dashboard")
    print("  2. npm start")
    print("\nThe dashboard will be available at http://localhost:3000")


if __name__ == "__main__":
    prepare_dashboard_data()
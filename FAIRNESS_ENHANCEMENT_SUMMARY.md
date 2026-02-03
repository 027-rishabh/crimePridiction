# Enhanced Fairness Metrics for Women and Children

## Summary of Changes Made

The Crime Prediction project has been updated to include comprehensive fairness metrics for women and children, in addition to the previously supported protected groups (SC, ST, General).

## Files Modified

### 1. `models/fairness_metrics.py`
- **Extended protected groups**: Updated the default list to include 'Women' and 'Children' alongside 'SC', 'ST', and 'General'
- **Added vulnerability-specific metrics**: 
  - `women_vs_others_max_diff`: Measures maximum difference in MAE between women and other groups
  - `children_vs_others_max_diff`: Measures maximum difference in MAE between children and other groups  
  - `women_children_fairness_gap`: Calculates fairness gap specifically between women and children groups
- **Enhanced reporting**: Updated the print_summary method to display the new vulnerability-specific fairness metrics
- **Maintained backward compatibility**: Kept 'General' group in the default list even though it's not in the current dataset

### 2. `run_all_baselines.py`
- **Updated fairness breakdown**: Modified the print_fairness_breakdown function to include all five protected groups (SC, ST, General, Women, Children) in the comparison table

## New Fairness Metrics Implemented

1. **Women vs Others Max Difference**: Maximum difference in model performance (MAE) between the women group and all other protected groups
2. **Children vs Others Max Difference**: Maximum difference in model performance (MAE) between the children group and all other protected groups
3. **Women-Children Fairness Gap**: Direct comparison of model performance between women and children groups
4. **Women vs Min Group Gap**: Minimum difference between women group performance and other groups
5. **Children vs Min Group Gap**: Minimum difference between children group performance and other groups

## Validation

The updates have been tested with:
- Synthetic test data containing all five protected groups
- Actual dataset containing the four relevant groups (SC, ST, Women, Children)
- Compatibility testing with existing model scripts to ensure no breaking changes

## Impact

These enhancements allow for better evaluation of model fairness across all protected groups, with special attention to vulnerable populations (women and children), ensuring that predictive models do not disproportionately underperform for these important demographic segments.
"""
Validation against real world datasets.

This module compares emulated data from sensors to real datasets using statistical tests and visualizations. It helps identify any discrepancies or biases in the emulation process.
- Kilmogorov-Smirnov test for distribution comparison
- Descriptive statistics comparison (mean, std, min, max)
- Percentage differences in key metrics ±20% tolerance

Datasets used:
1. Kaggle  Smart Home Dataset with Weather Information (https://www.kaggle.com/datasets/taranvee/smart-home-dataset-with-weather-information)
2. UCI Appliances Energy Prediction Dataset (https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)

Output includes:
- results/validation_results.csv
- results/validation_summary.png
- results/distribution_comparison.png

Usage:
    from analysis.validation import run_validation
    results = run_validation(df_emulated, df_real, feature_map)
    Standalone:
    python -m analysis.validation
"""

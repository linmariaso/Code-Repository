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
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, describe
from config.settings import validation_tolerance, significance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

proc_dir = os.path.join('data', 'processed')
val_dir = os.path.join('data', 'validation')
res_dir = os.path.join('results')
os.makedirs(res_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

def validate_feature(emulated, real, feature_name, tolerance=validation_tolerance):
    """Validate a single feature by comparing emulated and real data.
"""
    emulated = pd.Series(emulated).dropna().values
    real = pd.Series(real).dropna().values

    if len(emulated) == 0 or len(real) == 0:
        logger.warning(f"Feature '{feature_name}' has no data for validation.")
        return None

    emulated_mean, real_mean = emulated.mean(), real.mean()
    emulated_std, real_std = emulated.std(), real.std()
    emulated_min, real_min = emulated.min(), real.min()
    emulated_max, real_max = emulated.max(), real.max()
    emulated_median, real_median = emulated.median(), real.median()

    if real_mean != 0:
        pct_diff = abs(emulated_mean - real_mean) / abs(real_mean) * 100
    else:
        pct_diff = 0.0 if emulated_mean == 0 else 100.0

    within_tolerance = pct_diff <= (tolerance * 100)

    # Kilmogorov-Smirnov test
    ks_stat, ks_p = ks_2samp(emulated, real)
    
    if ks_stat < 0.1:
        ks_result = 'Very Similar distributions'
    elif ks_stat < 0.3:
        ks_result = 'Moderately Similar distributions'
    else:
        ks_result = 'Substantially different distributions'

    results = {
        'feature': feature_name,
        'emulated_n': len(emulated),
        'real_n': len(real),
        'emulated_mean': round(emulated_mean, 2),
        'real_mean': round(real_mean, 2),
        'pct_difference': round(pct_diff, 2),
        'within_tolerance': within_tolerance,
        'emulated_std': round(emulated_std, 2),
        'real_std': round(real_std, 2),
        'emulated_median': round(emulated_median, 2),
        'real_median': round(real_median, 2),
        'emulated_min': round(emulated_min, 2),
        'real_min': round(real_min, 2),
        'emulated_max': round(emulated_max, 2),
        'real_max': round(real_max, 2),
        'ks_statistic': round(ks_stat, 4),
        'ks_p_value': round(ks_p, 6),
        'ks_significant': ks_p < significance,
        'ks_result': ks_result
    }
    tolerance_status = "PASS" if within_tolerance else "FAIL"
    ks_status = "Significant" if ks_p < significance else "Not Significant"
    logger.info(f"Validation for '{feature_name}': {tolerance_status}, KS Test: {ks_status} (stat={ks_stat:.4f}, p={ks_p:.6f})")
    return results

def validate_against_dataset(df_emulated, df_real, feature_map, dataset_name = 'Real Dataset'):
    logger.info(f"Validating against {dataset_name}")
    logger.info(f"Emulated data shape: {df_emulated.shape}, Real data shape: {df_real.shape}")

    results = []

    for emulation_col, real_col in feature_map.items():
        if emulation_col not in df_emulated.columns:
            logger.warning(f"Emulated column {emulation_col} has no data")
            continue
        if real_col not in df_real.columns:
            logger.warning(f"Real column {real_col} has no data")
            continue

        result = validate_feature(df_emulated[emulation_col], df_real[real_col], feature_name=f"{emulation_col} vs {real_col}")

        if result:
            result['dataset'] = dataset_name
            result['emulated_column'] = emulation_col
            result['real_column'] = real_col
            results.append(result)
    
    if results:
        passed = sum(1 for r in results if r['within_tolerance'])
        total = len(results)
        logger.info(f"\n {dataset_name}: {passed}/{total} features within ±{validation_tolerance*100:.0f}%")
    return results

def run_validation(df_emulated, df_real, feature_map, dataset_name='Real Dataset'):
    logger.info(f"Starting validation against dataset: {dataset_name}")
    logger.info(f"Emulated dataset shape: {df_emulated.shape}, Real dataset shape: {df_real.shape}")

    validation_results = []
    for emulation_col, real_col in feature_map.items():
        if emulation_col not in df_emulated.columns:
            logger.warning(f"Emulated feature '{emulation_col}' is missing in the emulated dataset.")
            continue
        if real_col not in df_real.columns:
            logger.warning(f"Real feature '{real_col}' is missing in the real dataset.")
            continue

        result = validate_feature(df_emulated[emulation_col], df_real[real_col], feature_name=f"{emulation_col} vs {real_col}")
        if result:
            result['dataset'] = dataset_name
            result['emulated_column'] = emulation_col
            result['real_column'] = real_col
            validation_results.append(result)
    
    if validation_results:
        passed = sum(1 for r in validation_results if r['within_tolerance'])
        total = len(validation_results)
        logger.info(f"Validation completed on {dataset_name}: {passed}/{total} features passed within ±{validation_tolerance*100}% tolerance.") 
    
    return validation_results

def load_uci_appliances(file_path = None):
    if file_path is None:
        file_path = os.path.join(val_dir, 'energydata_complete.csv')
    if not os.path.exists(file_path):
        logger.error(f"UCI Appliances Energy Prediction dataset not found at {file_path}. Please download and place it there.")
        return None, None
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    temp_cols = [col for col in df.columns if col.startswith('T') and col not in ['T_out', 'Tdewpoint']]
    rh_cols = [col for col in df.columns if col.startswith('RH') and col != 'RH_out']

    if temp_cols:
        df['mean_temperature'] = df[temp_cols].mean(axis=1)
    if rh_cols:
        df['mean_humidity'] = df[rh_cols].mean(axis=1)

    df['total_power'] = df['Appliances'] + df.get('lights', 0)
    df['hour'] = df.index.hour

    feature_map = {
        'mean_temperature': 'mean_temperature',
        'mean_humidity': 'mean_humidity',
        'total_power': 'total_power',
        'hour': 'hour'
    }

    logger.info(f"Loaded UCI Appliances Energy Prediction dataset with shape: {df.shape}")
    return df, feature_map

def load_kaggle_smart_home(file_path = None):
    if file_path is None:
        file_path = os.path.join(val_dir, 'HomeC.csv')
    if not os.path.exists(file_path):
        logger.error(f"Kaggle Smart Home Dataset not found at {file_path}. Please download and place it there.")
        return None, None
    df = pd.read_csv(file_path, low_memory=False)
    time_col = None
    for candidate in ['Time', 'time', 'timestamp', 'date']:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df.set_index(time_col, inplace=True)
        df.dropna(how='all', inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    power_cols = [col for col in df.columns if 'use' in col.lower() or 'energy' in col.lower() or 'kw' in col.lower() or 'power' in col.lower()]
    temp_cols = [col for col in df.columns if 'temp' in col.lower() or 'temperature' in col.lower()]
    rh_cols = [col for col in df.columns if 'hum' in col.lower() or 'humidity' in col.lower()]

    feature_map = {}

    if power_cols:
        df['total_power_real'] = df[power_cols].sum(axis=1)
        if df['total_power_real'].mean() < 50:
            df['total_power_real'] *= 1000
        feature_map['total_power'] = 'total_power_real'

    if temp_cols:
        df['mean_temperature_real'] = df[temp_cols].mean(axis=1)
        feature_map['mean_temperature'] = 'mean_temperature_real'

    if rh_cols:
        df['mean_humidity_real'] = df[rh_cols].mean(axis=1)
        feature_map['mean_humidity'] = 'mean_humidity_real'
    if hasattr(df.index, 'hour'):
        df['hour'] = df.index.hour
        feature_map['hour'] = df.index.hour
    
    logger.info(f"Loaded Kaggle Smart Home dataset with shape: {df.shape}")
    return df, feature_map

def plot_distribution_comparison(df_emulated, df_real, feature_map, dataset_name='Real', output_path=res_dir):
    """Histograms comparing emulated and real distributions"""
    n_features = len(feature_map)
    if n_features == 0:
        return
    cols = min(n_features, 3)
    rows = (n_features + cols - 1)// cols
    fig, axes = plt.subplots(rows,cols, figsize = (5 * cols, 4 * rows))
    if n_features == 1:
        axes = np.array([axes])
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for index, (emulated_col, real_col) in enumerate(feature_map.items()):
        if index >= len(axes):
            break
        ax = axes[index]

        emulated_data = df_emulated[emulated_col].dropna()
        real_data = df_real[real_col].dropna()

        all_data = np.concatenate([emulated_data.values, real_data.values])
        bins = np.linspace(np.percentile(all_data, 1), np.percentile(all_data,99), 50)

        ax.hist(emulated_data, bins = bins, alpha = 0.6, color = 'blue', label = 'Emulated', density = True, edgecolor = 'white')
        ax.hist(real_data, bins = bins, alpha = 0.6, color = 'orange', label = 'Real', density = True, edgecolor = 'white')

        ax.set_title(f'{emulated_col}', fontsize = 10)
        ax.set_ylabel('Density')
        ax.legend(fontsize = 7)
        ax.grid(True, alpha = 0.3)

    for index in range(n_features, len(axes)):
        axes[index].set_visible(False)

    plt.suptitle(f'Distribution comparison: Emulated vs {dataset_name}', fontsize = 10, y=1.02)
    plt.tight_layout()
    safe_name = dataset_name.replace(' ',' ').lower()
    plt.savefig(os.path.join(output_path,f'distribution_comparison_{safe_name}.png'))
    plt.close()
    logger.info(f"Saved Distribution Comparison to {output_path}")

def plot_validation_summary(all_results, output_path=res_dir):
    if not all_results:
        return
    df = pd.DataFrame(all_results)

    fig, axes = plt.subplots( 1, 2, figsize = (12, 5))

    ax = axes[0]
    colours = ['red' if w else 'green' for w in df['within_tolerance']]

    y_pos = np.arrange(len(df))
    ax.barh(y_pos, df['pct_difference'], color = colours, alpha = 0.8)
    ax.axvline(validation_tolerance * 100, color = 'red', linestyle = '--', label = f'±{validation_tolerance*100:.0f}% tolerance')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['feature'], fontsize = 8)
    ax.set_xlabel("Mean % difference")
    ax.set_title("Mean difference (Green = PASS, Red = FAIL)")
    ax.legend(fontsize = 8)
    ax.grid(True, axis= 'x', alpha = 0.3)
    ax.invert_yaxis()

    ax = axes[1]
    colours = ['red' if s else 'green' for s in df['ks_significant']]
    ax.barh(y_pos, df['ks_statistic'], color = colours, alpha = 0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['feature'], fontsize = 8)
    ax.set_xlabel("KS Statistic")
    ax.set_title("KS Test (Green = Not Significant, Red = Significantly Different)")
    ax.legend(fontsize = 8)
    ax.grid(True, axis= 'x', alpha = 0.3)
    ax.invert_yaxis()

    plt.suptitle('Validation Summary', fontsize = 10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'validation_summary.png'), dpi = 150)
    plt.close()
    logger.info(f"Saved Validation Summary to {output_path}")

def save_results(all_results, output_path = res_dir):
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_path, 'validation_results.csv'), index = True)
    logger.info(f"Saved Validation results on {output_path}")
    return df

def run_validation(emulated_features_path = None):
    
    logger.info("Validation against Real World Datasets")

    if emulated_features_path is None:
        emulated_features_path = os.path.join(proc_dir, 'features.csv')
    
    if not os.path.exists(emulated_features_path):
        logger.error(f"Emulated Features not found at {emulated_features_path}. Please run preprocessing first.")
        return []
    
    df_emulated = pd.read_csv(emulated_features_path, index_col='timestamp', parse_dates=True)
    logger.info(f"Loaded emulated features with shape: {df_emulated.shape}")

    all_results = []
    datasets_found = 0

    # UCI Dataset
    df_uci, map_uci = load_uci_appliances()
    if df_uci is not None and map_uci:
        datasets_found += 1
        results = validate_against_dataset(df_emulated, df_uci, map_uci, dataset_name='UCI Appliances Energy')
        all_results.extend(results)
        plot_distribution_comparison(df_emulated, df_uci, map_uci, dataset_name='UCI Appliances Energy')

    # Kaggle Dataset
    df_kaggle, map_kaggle = load_kaggle_smart_home()
    if df_kaggle is not None and map_kaggle:
        datasets_found += 1
        results = validate_against_dataset(df_emulated, df_kaggle, map_kaggle, dataset_name='Kaggle Smart Home')
        all_results.extend(results)
        plot_distribution_comparison(df_emulated, df_kaggle, map_kaggle, dataset_name='Kaggle Smart Home')

    if datasets_found == 0:
        logger.warning(f"No real world dataset found in {val_dir}. To run validation download UCI Apliances Energy or Kaggle Smart Home")
        return []
    
    save_results(all_results)
    plot_validation_summary(all_results)

    passed = sum(1 for r in all_results if r['within_tolerance'])
    total = len(all_results)
    logger.info(f"Validation completed on {datasets_found} datasets: {passed}/{total}({passed/max(total,1)*100}) features passed within ±{validation_tolerance*100}% tolerance.") 

    ks_similar = sum(1 for r in all_results if not r['ks_significant'])
    logger.info(f"KS test not significant (similar): {ks_similar}/{total}")

    return all_results

if __name__ == '__main__':
    results = run_validation()

    if results:
        for r in results:
            tolerance = "PASS" if r['within_tolerance'] else "FAIL"
            signif = "YES" if r['ks_significant'] else "NO"
            print(f"Feature {r['feature']}: {r['pct_difference']} % Difference Pass:{tolerance} KS: {r['ks_significant']} Significant?: {signif}")

        print(f"Results saved to {res_dir}")
    else:
        print("No validation results. Download datasets first.")

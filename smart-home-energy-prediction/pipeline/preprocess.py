"""
Loads raw CSV files, preprocesses the data by pivoting to wide format, resampling, and engineering features. The processed datasets are saved for use in model training and evaluation.

Key steps:
1. Load all raw CSV files from the data/raw directory.
2. Parse timestamps and numeric values, handling any malformed records.
3. Resample data to a consistent frequency (e.g., 5 minutes) and pivot to wide format with separate columns for each sensor type and room/appliance.
4. Engineer additional features such as hour of day, day of week, mean temperature, occupancy count, and total power consumption.
5. Save the processed datasets to data/processed for downstream modeling.
Output:
- data/processed/wide_data.csv: The resampled and pivoted dataset in wide format
Usage:
    from pipeline.preprocess import run_preprocessing
    df_features = run_preprocessing()
    Standalone:
    python -m pipeline.preprocess
"""
import os
import glob
import logging
import numpy as np
import pandas as pd
from config.settings import (anomaly_rate, rooms, appliances, r_seed)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
raw_dir = os.path.join('data', 'raw')
proc_dir = os.path.join('data', 'processed')
os.makedirs(proc_dir, exist_ok=True)

def load_raw_data():
    csv_files = glob.glob(os.path.join(raw_dir, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}. Please run the emulation first to generate data.")
    logger.info(f"Found {len(csv_files)} raw data files. Loading...")

    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, parse_dates=['timestamp'])
            df_list.append(df)
            logger.info(f"Loaded {len(df)} records from {file}")
        except Exception as e:
            logger.warning(f"Skipping {file}: {e}")

    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
    merged_df['values'] = pd.to_numeric(merged_df['value'], errors='coerce')

    before_drop = len(merged_df)
    merged_df.dropna(subset=['timestamp', 'values'], inplace=True)
    dropped = before_drop - len(merged_df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} records due to missing timestamps or non-numeric values.")
    else:
        logger.info("No records dropped during preprocessing.")

    merged_df.sort_values('timestamp', inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    logger.info(f"Total records after preprocessing: {len(merged_df)}\nTime range: {merged_df['timestamp'].min()} to {merged_df['timestamp'].max()}")
    return merged_df

def wide_format(df, resample_freq='5min'):
    logger.info(f"Resampling data to {resample_freq} frequency and pivoting to wide format")
    env_mask = df['sensor_type'].isin(['temperature', 'humidity', 'occupancy'])
    #Separate environmental and appliance data
    df_env = df[env_mask].copy()
    df_appl = df[~env_mask].copy()

    df_env['column'] = df_env['sensor_type'] + '_' + df_env['room']
    env_pivot = df_env.pivot_table(index='timestamp', columns='column', values='values', aggfunc='mean')

    if len(df_appl) > 0:
        df_appl['column'] = 'power_' + df_appl['appliance']
        appl_pivot = df_appl.pivot_table(index='timestamp', columns='column', values='values', aggfunc='mean')
        wide = env_pivot.join(appl_pivot, how='outer')
    else:
        wide = env_pivot

    wide = wide.resample(resample_freq).mean()
    wide= wide.ffill().bfill()
    logger.info(f"Data reshaped to wide format with {wide.shape[0]} records and {wide.shape[1]} columns")
    return wide

def engineer_features(df_wide):
    logger.info("Engineering additional features")
    features = pd.DataFrame(index=df_wide.index)
    features['hour'] = features.index.hour
    features['day_of_week'] = features.index.dayofweek
    features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)

    temp_cols = [col for col in df_wide.columns if col.startswith('temperature_')]
    hum_cols = [col for col in df_wide.columns if col.startswith('humidity_')]
    

    if temp_cols:
        features['mean_temperature'] = df_wide[temp_cols].mean(axis=1)
    else:
        logger.warning("No temperature columns found for feature engineering.")
        features['mean_temperature'] = 0.0
    if hum_cols:
        features['mean_humidity'] = df_wide[hum_cols].mean(axis=1)
    else:
        logger.warning("No humidity columns found for feature engineering.")
        features['mean_humidity'] = 0.0

    occ_cols = [col for col in df_wide.columns if col.startswith('occupancy_')]
    sp_cols = [col for col in df_wide.columns if col.startswith('power_')]

    if occ_cols:
        features['occupancy_count'] = df_wide[occ_cols].sum(axis=1)
    else:
        logger.warning("No occupancy columns found for feature engineering.")
        features['occupancy_count'] = 0
    if sp_cols:
        features['active_appliances'] = (df_wide[sp_cols] > 10).sum(axis=1)
        features['total_power'] = df_wide[sp_cols].sum(axis=1)
    else:
        logger.warning("No appliance power columns found for feature engineering.")
        features['active_appliances'] = 0
        features['total_power'] = 0.0

    features['mean_temperature'] = features['mean_temperature'].round(2)
    features['mean_humidity'] = features['mean_humidity'].round(2)
    features['total_power'] = features['total_power'].round(1)

    logger.info(f"Feature engineering complete. Dataset now has {features.shape[1]} columns and {features.shape[0]} rows.")
    logger.info(f"Total power mean: {features['total_power'].mean():.1f} | Power mean: {features['total_power'].max():.1f} Power min: {features['total_power'].min():.1f}")
    return features

def run_preprocessing(raw_dir=raw_dir, resample_freq='5min', save=True):

    logger.info("Phase 3: Data Preprocessing")

    df_long = load_raw_data(raw_dir)
    df_wide = wide_format(df_long, resample_freq)
    df_features = engineer_features(df_wide)

    if save:
        wide_path = os.path.join(proc_dir, 'wide_data.csv')
        feature_path = os.path.join(proc_dir, 'features_data.csv')
        df_wide.to_csv(wide_path)
        df_features.to_csv(feature_path)
        logger.info(f"Wide format data saved to {wide_path}")
        logger.info(f"Feature engineered data saved to {feature_path}")

    logger.info("Preprocessing complete.")
    return df_features

if __name__ == "__main__":
    df = run_preprocessing()
    print(f"Preprocessed dataset shape: {df.shape}")
    print(f"Columns:\n {df.columns.tolist()}")
    print(f"First five rows:\n{df.head()}")
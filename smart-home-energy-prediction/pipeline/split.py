
import os
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config.settings import test_ratio, r_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
proc_dir = os.path.join('data', 'processed')

def prepare_data(features, test_ratio=test_ratio):
    
    target_col = 'total_power'
    feature_cols = [col for col in features.columns if col != target_col]
    logger.info("Preparing data for modeling")
    logger.info(f"Total records: {len(features)}")
    logger.info(f"Feature columns: {feature_cols}")
    logger.info(f"Target column: {target_col}")

    split_index = int(len(features) * (1 - test_ratio))

    train = features.iloc[:split_index]
    test = features.iloc[split_index:]

    logger.info(f"Training set: {len(train)} records")
    logger.info(f"Testing set: {len(test)} records")
    logger.info(f"Split ratio: {test_ratio:.2%} test, {1 - test_ratio:.2%} train")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_test = scaler.fit_transform(test[feature_cols])
    y_train = train[target_col].values
    y_test = test[target_col].values

    logger.info("\n  Feature scaling (training set):")
    for i, col in enumerate(feature_cols):
        logger.info("    %s: mean=%.2f, std=%.2f",
                     col, scaler.mean_[i], scaler.scale_[i])
        
    logger.info("\n  Target statistics:")
    logger.info("    Train — mean=%.1fW, std=%.1fW, "
                "min=%.1fW, max=%.1fW",
                y_train.mean(), y_train.std(),
                y_train.min(), y_train.max())
    logger.info("    Test  — mean=%.1fW, std=%.1fW, "
                "min=%.1fW, max=%.1fW",
                y_test.mean(), y_test.std(),
                y_test.min(), y_test.max())
    
    return X_train, y_train, X_test, y_test, scaler, feature_cols

def save_split(X_train, y_train, X_test, y_test, scaler, feature_cols, output_dir = proc_dir):

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(output_dir, 'feature_cols.pkl'), 'wb') as f:
        pickle.dump(feature_cols, f)
    logger.info(f"Saved splits and scaler data to {output_dir}")

def load_split(input_dir = proc_dir):
    X_train = np.load(os.path.join(input_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(input_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(input_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(input_dir, 'y_test.npy'))

    with open(os.path.join(input_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(input_dir, 'feature_cols.pkl'), 'rb') as f:
        feature_cols = pickle.load(f)
    logger.info(f"Loaded splits and scaler data from {input_dir}")
    return X_train, y_train, X_test, y_test, scaler, feature_cols

if __name__ == "__main__":
    features_path = os.path.join(proc_dir, 'features_data.csv')
    if not os.path.exists(features_path):
        logger.error(f"Features file not found: {features_path}")
        exit(1)

    df_features = pd.read_csv(features_path, index_col='timestamp', parse_dates=True)
    logger.info(f"Loaded features dataset with {len(df_features)} records and {len(df_features.columns)} columns")
    X_train, y_train, X_test, y_test, scaler, feature_cols = prepare_data(df_features)
    save_split(X_train, y_train, X_test, y_test, scaler, feature_cols)
    print(f"Data preparation complete. Training set: {X_train.shape}, Testing set: {X_test.shape}")
    print(f"Feature columns: {feature_cols}")

"""
Train and tune regression models for energy consumption prediction. This module includes functions to train individual models with optional hyperparameter tuning, train all specified models, and save/load trained models and results. Models include Linear Regression, Random Forest, and MLP Regressor. Training times and CV scores are logged for performance comparison.
Models:
1. Linear Regression (no hyperparameters)
2. Random Forest Regressor (n_estimators, max_depth, min_samples_split)
3. MLP Regressor (hidden_layer_sizes, activation, learning_rate_init)
Each model with the exception of Linear Regression undergoes GridSearchCV for hyperparameter tuning. Results are saved in 'results/trained_models.pkl' and individual model files. 
Usage:
    from models.train import train_all_models, save_models
    results = train_all_models(X_train, y_train)
    Standalone:
    python -m models.train
"""
import os
import time
import pickle
import logging
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from config.settings import folds, r_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

proc_dir = os.path.join('data', 'processed')
res_dir = 'results'
os.makedirs(res_dir, exist_ok=True)

def get_models_and_params():
    return {
        'LinearRegression': {
            'model': LinearRegression(),
            'params': {}
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=r_seed, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 500],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'MLPRegressor': {
            'model': MLPRegressor(random_state=r_seed, max_iter=500, early_stopping=True,validation_fraction=0.1, n_iter_no_change=20),
            'params': {
                'hidden_layer_sizes': [(64,32), (128,64), (128, 64, 32)],
                'activation': ['relu', 'tanh'],
                'learning_rate_init': [0.001, 0.01]
            }
        }
    }

def train_single_model(name, model, params, X_train, y_train):
    logger.info(f"Training {name} with parameters: {params}")
    start_time = time.time()
    if params:
        n_combs = 1
        for param_values in params.values():
            n_combs *= len(param_values)
        total_hits = n_combs * folds
        logger.info(f"Total combinations: {n_combs} | Number of folds: {folds} | Total training runs (with CV): {total_hits}")
        grid = GridSearchCV(model, params, cv=folds, n_jobs=-1, verbose=1, return_score=True)
        grid.fit(X_train, y_train)

        elapsed = time.time() - start_time
        best = grid.best_estimator_
        best_params = grid.best_params_
        best_score = grid.best_score_
        logger.info(f"Best parameters: {best_params} | CV Score: {best_score:.2f} | Training time: {elapsed:.1f}s")
        return{
            'estimator': best,
            'best_params': best_params,
            'cv_score': best_score,
            'cv_results': grid.cv_results_,
            'training_time': elapsed
        }
    else:
        model.fit(X_train, y_train)
        elapsed = time.time() - start_time
        logger.info("No hyperparameters to tune. Training completed.")
        logger.info(f"Training completed in {elapsed:.1f}s")
        return {
            'estimator': model,
            'best_params': None,
            'cv_score': None,
            'cv_results': None,
            'training_time': elapsed
        }

def train_all_models(X_train, y_train):
    logger.info("Starting training of all models")
    logger.info(f"Training data: {X_train.shape[0]} records, {X_train.shape[1]} features")
    models = get_models_and_params()
    results = {}
    total_start_time = time.time()
    for name, info in models.items():
        try:
            result = train_single_model(name, info['model'], info['params'], X_train, y_train)
            results[name] = result
        except Exception as e:
            logger.error(f"Error training {name}: {e}")
    total_time = time.time() - total_start_time
    logger.info(f"All models trained in {total_time:.1f}s")
    for name, res in results.items():
        cv = f"{res['cv_score']:.2f}" if res['cv_score'] is not None else "N/A"
        logger.info(f"{name} | CV Score: {cv} | Training Time: {res['training_time']:.1f}s")
    return results

def save_models(results, output_dir=res_dir):
    full_path = os.path.join(output_dir, 'trained_models.pkl')
    with open(full_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info("Saved all results to %s", full_path)

    for name, res in results.items():
        model_path = os.path.join(output_dir, f"{name}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(res['estimator'], f)
        logger.info(f"Saved {name} model to {model_path}")

def load_models(input_dir=res_dir):
    path = os.path.join(input_dir, 'trained_models.pkl')
    with open(path, 'rb') as f:
        results = pickle.load(f)
    logger.info(f"Loaded {len(results)} results from {path}")
    return results

if __name__ == "__main__":
    from pipeline.split import load_split
    print("Loading data splits")
    X_train, y_train, X_test, y_test, scaler, feature_cols = load_split(proc_dir)
    print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")

    results = train_all_models(X_train, y_train)
    save_models(results)
    print("All models saved. Ready for evaluation.")
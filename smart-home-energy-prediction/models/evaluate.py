"""
Evaluate models and compare statistically
Metrics per model:
- R² Coefficient of Determination
- RMSE Root Mean Squared Error
- MAE Mean Absolute Error

Statistical comparison:
- Paired t-tests on absolute errors of predictions (model A vs model B)
- Check against dissertation targets (R² ≥ 0.7, RMSE < 100 W, MAE < 120 W)

Outputs:
- results/evaluation_metrics.csv
- results/model_comparisons.csv
- results/prediction_vs_actual.png
- results/residual_analysis.png
- results/model_comparison_bars.png

Usage:

- Standalone evaluation:
    python -m models.evaluate
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from config.settings import r2_target, rmse_target, mae_target, significance, permutation_repeats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

proc_dir = os.path.join('data', 'processed')
res_dir = 'results'
os.makedirs(res_dir, exist_ok=True)

def evaluate_model(model, X_test, y_test, model_name = 'Model'):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    r2_diss = r2 >= r2_target
    rmse_diss = rmse <= rmse_target
    mae_diss = mae <= mae_target

    logger.info(f"Evaluation results for {model_name}:")
    logger.info(f"  R²: {r2:.3f} (Target: {r2_target} | {'✓' if r2_diss else '✗'})")
    logger.info(f"  RMSE: {rmse:.1f} W (Target: {rmse_target} W | {'✓' if rmse_diss else '✗'})")
    logger.info(f"  MAE: {mae:.1f} W (Target: {mae_target} W | {'✓' if mae_diss else '✗'})")

    logger.info(f"  R²: {r2:.3f} | RMSE: {rmse:.1f} W | MAE: {mae:.1f} W")
    return {
        'model_name': model_name,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'meets_r2_target': r2_diss,
        'meets_rmse_target': rmse_diss,
        'meets_mae_target': mae_diss,
        'y_pred': y_pred
    }

def compare_models(eval_res, y_test):
    logger.info("\nStatistical comparison (paired t-tests)")

    model_names = list(eval_res.keys())
    comparisons = []

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_a = model_names[i]
            name_b = model_names[j]
            pred_a = eval_res[name_a]['y_pred']
            pred_b = eval_res[name_b]['y_pred']

            abs_errors_a = np.abs(y_test - pred_a)
            abs_errors_b = np.abs(y_test - pred_b)

            t_stat, p_value = stats.ttest_rel(abs_errors_a, abs_errors_b)
            significant = p_value < significance

            better = name_a if abs_errors_a.mean() < abs_errors_b.mean() else name_b

            comp = {
                'Model_A': name_a,
                'Model_B': name_b,
                'MAE_A': abs_errors_a.mean(),
                'MAE_B': abs_errors_b.mean(),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': significant,
                'better_model': better
            }
            comparisons.append(comp)
            sig_string = "Yes" if significant else "No"

            logger.info(f" {name_a} vs {name_b}: t={t_stat:.3f}, p={p_value:.4f} ({sig_string}) | Better: {better})") 
    return comparisons

def plot_predictions_vs_actual(y_test, eval_res, output_dir = res_dir):
    n_models = len(eval_res)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    for ax, (model_name, res) in zip(axes, eval_res.items()):
        y_pred = res['y_pred']
        ax.scatter(y_test, y_pred, alpha=0.3, s=5, label='Predictions', color='steelblue')
        lims = [
            min(y_test.min(), y_pred.min()),
            max(y_test.max(), y_pred.max())
        ]
        ax.plot(lims, lims, 'r--', linewidth = 1, label='Perfect Prediction')
        ax.set_title(f"{model_name} Predictions vs Actual")
        ax.set_xlabel("Actual Power (W)")
        ax.set_ylabel("Predicted Power (W)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_vs_actual.png'))
    plt.close()

def plot_residuals(y_test, eval_res, output_dir = res_dir):
    n_models = len(eval_res)
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = axes.reshape(-1, 1)
        
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
    plt.savefig(os.path.join(output_dir, 'prediction_vs_actual.png'), dpi = 150)
    plt.close()

def plot_residuals(y_test, eval_res, output_dir = res_dir):
    n_models = len(eval_res)
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = axes.reshape(-1, 1)
    
    for col, (name,res) in enumerate(eval_res.items()):
        residuals = y_test - res['y_pred']

        ax = axes[0, col]
        ax.hist(residuals, bins=50, color='salmon', edgecolor='white', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        ax.set_title(f"{name} Residuals")
        ax.set_xlabel("Residual (W)")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

        ax.text(0.95, 0.95, f"Mean: {residuals.mean():.1f} W\nStd: {residuals.std():.1f} W", transform=ax.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axes[1,col]
        ax.scatter(range(len(residuals)), residuals, alpha=0.3, s=3, color='steelblue')
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel("Simple Index")
        ax.setylabel("Residual (W)")
        ax.set_title(f"{name} Residuals over Time")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_analysis.png'), dpi = 150)
    plt.close()
    logger.info(f"Saved residual analysis plots to {output_dir}")

def plot_model_comparison(eval_res, output_dir = res_dir):
    names = list(eval_res.keys())
    r2_scores = [eval_res[n]['r2'] for n in names]
    rmse_scores = [eval_res[n]['rmse'] for n in names]
    mae_scores = [eval_res[n]['mae'] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Model Performance Comparison", fontsize=14)

    x = np.arange(len(names))
    colours = ['steelblue', 'salmon', 'lightgreen', 'orange', 'purple']

    #R2 Scores
    ax = axes[0]
    bars = ax.bar(x, r2_scores, color=colours[:len(names)], alpha=0.8)
    ax.axhline(r2_target, color='red', linestyle='--', label=f"R² Target ({r2_target})")
    ax.set_ylabel("R²")
    ax.set_title("R² Score")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize = 8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars,r2_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.3f}", ha='center', va='bottom', fontsize=8)

    #RMSE Scores
    ax = axes[1]
    bars = ax.bar(x, rmse_scores, color=colours[:len(names)], alpha=0.8)
    ax.axhline(rmse_target, color='red', linestyle='--', label=f"RMSE Target ({rmse_target} W)")
    ax.set_ylabel("RMSE (W)")
    ax.set_title("RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize = 8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rmse_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.1f}", ha='center', va='bottom', fontsize=8)

    #MAE Scores
    ax = axes[2]
    bars = ax.bar(x, mae_scores, color=colours[:len(names)], alpha=0.8)
    ax.axhline(mae_target, color='red', linestyle='--', label=f"MAE Target ({mae_target} W)")
    ax.set_ylabel("MAE (W)")
    ax.set_title("MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize = 8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, mae_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.1f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_bars.png'), dpi = 150)
    plt.close()
    logger.info(f"Saved model comparison bar charts to {output_dir}")

def evaluate_all_models(trained_res, X_test, y_test, save_results = True, output_dir = res_dir):
    logger.info("Starting evaluation of all models")
    logger.info(f"Testing data: {X_test.shape[0]} records, {X_test.shape[1]} features")
    eval_res = {}
    for name, res in trained_res.items():
        metrics = evaluate_model(res['estimator'], X_test, y_test, model_name=name)
        metrics['best_params'] = res.get['best_params']
        metrics['training_time'] = res.get('training_time')
        eval_res[name] = metrics

    comparisons = compare_models(eval_res, y_test)
    best_model = max(eval_res, key=lambda n: eval_res[n]['r2'])
    logger.info(f"\nBest overall model: {best_model} with R²={eval_res[best_model]['r2']:.3f}")

    if save:
        metrics_rows = []
        for name, res in eval_res.items():
            metrics_rows.append({
                'model_name': name,
                'r2': round(res['r2'],4),
                'rmse': round(res['rmse'],2),
                'mae': round(res['mae'],2),
                'meets_r2_target': res['meets_r2_target'],
                'meets_rmse_target': res['meets_rmse_target'],
                'meets_mae_target': res['meets_mae_target'],
                'best_params': str(res['best_params']),
                'training_time': round(res['training_time'],1)
            })
        df_metrics = pd.DataFrame(metrics_rows)
        df_metrics.to_csv(os.path.join(output_dir, 'evaluation_metrics.csv'), index=False)
        logger.info(f"Saved evaluation metrics to {output_dir}/evaluation_metrics.csv")

        df_comparisons = pd.DataFrame(comparisons)
        df_comparisons.to_csv(os.path.join(output_dir, 'model_comparisons.csv'), index=False)
        logger.info(f"Saved model comparisons to {output_dir}/model_comparisons.csv")

    if plot:
        plot_predictions_vs_actual(y_test, eval_res, output_dir)
        plot_residuals(y_test, eval_res, output_dir)
        plot_model_comparison(eval_res, output_dir)

    return eval_res, comparisons

if __name__ == "__main__":
    from pipeline.split import load_split
    from models.train import load_models

    print("Loading test data and trained models")
    X_train, X_test, y_train, y_test, scaler, feature_cols = load_split(proc_dir)
    trained_res = load_models(res_dir)

    print(f"Test data: {X_test.shape[0]} records, {X_test.shape[1]} features")

    eval_res, comparisons = evaluate_all_models(trained_res, X_test, y_test)

    print("\nEvaluation Summary:")
    for name, metrics in eval_res.items():
        print(f"{name}: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}")

    print("\nSignificance levels:")
    for comp in comparisons:
        print(f"{comp['Model_A']} vs {comp['Model_B']}: p-value={comp['p_value']:.4f} | Significant: {'Yes' if comp['significant'] else 'No'} | Better Model: {comp['better_model']}")

    print(f"\nPlots and CSV files saved to {res_dir}. Check evaluation_metrics.csv and model_comparisons.csv for detailed results.")
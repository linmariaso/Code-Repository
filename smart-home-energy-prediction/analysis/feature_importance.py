"""
Permutation importance analysis for the energy prediction model including statistical significance testing.

For each model, shuffle each feature and measure the change in R² score. Repeated N times to get confidence intervals and p-values for each feature's importance. This helps identify which features are truly impactful vs. noise.

Output includes:
- results/permutation_importance.csv
- results/permutation_importance.png
- results/importance_heatmap.png

Usage:

    Standalone:
    python -m analysis.feature_importance
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from scipy import stats
from config.settings import r_seed, permutation_repeats, significance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

proc_dir = os.path.join('data', 'processed')
res_dir = 'results'
os.makedirs(res_dir, exist_ok=True)

def calc_permutation_importance(model, X_test, y_test, feature_names, model_name='Model'):
    logger.info(f"Calculating permutation importance for {model_name} ({permutation_repeats} repeats)")
    result = permutation_importance(model, X_test, y_test, n_repeats=permutation_repeats, random_state=r_seed, scoring='r2', n_jobs=-1)
    
    importance_data = []

    for i, name in enumerate(feature_names):
        
        mean_importance = result.importances_mean[i]
        std_importance = result.importances_std[i]
        if std_importance > 0:
            t_stat = mean_importance / (std_importance/np.sqrt(permutation_repeats))
            p_value = stats.t.sf(t_stat, permutation_repeats-1)  # two-tailed
        else:
            t_stat = 0
            p_value = 1.0 if mean_importance <= 0 else 0.0  
        
        importance_data.append({
            'model': model_name,
            'feature': name,
            'mean_importance': mean_importance,
            'std_importance': std_importance,
            'raw_importances': result.importances[i],
            't_stat': t_stat,
            'p_value': p_value,
            'significant': p_value < significance
        })

    importance_data.sort(key=lambda x: x['mean_importance'], reverse=True)
    for rank, item in enumerate(importance_data, start=1):
        item['rank'] = rank

    for item in importance_data:
        logger.info(f"{item['model']} - {item['feature']}: mean={item['mean_importance']:.4f}, std={item['std_importance']:.4f}, p={item['p_value']:.4e}, significant={item['significant']}")

    return importance_data

def run_permutation_importance(trained_res, X_test, y_test, feature_names):
    logger.info("Running permutation importance")
    all_importance = []

    for name, res in trained_res.items():
        importance = calc_permutation_importance(res['estimator'], X_test, y_test, feature_names, model_name=name)
        all_importance[name] = importance
    return all_importance

def plot_importance_bars(all_importance, output_path= res_dir):
    n_models = len(all_importance)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, importance) in zip(axes, all_importance.items()):
        features = [d['feature'] for d in importance]
        means = [d['mean_importance'] for d in importance]
        stds = [d['std_importance'] for d in importance]
        sigs = [d['significant'] for d in importance]
        colours = ['red' if sig else 'blue' for sig in sigs]

        y_pos = np.arange(len(features))
        ax.barh(y_pos, means, xerr=stds, color=colours, alpha=0.7, edgecolor='white', capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=8)
        ax.set_xlabel('Mean Permutation Importance (R² decrease)')
        ax.set_title(f'{model_name} Feature Importance', fontsize=10)
        ax.axvline(0, color='grey', linestyle='--', linewidth=0.5)
        ax.grid(True, axis='x', alpha=0.3)
        ax.invert_yaxis()  

    plt.suptitle('Permutation Importance of Features with significance', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'permutation_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved permutation importance plot to {output_path}")

def plot_importance_heatmap(all_importance, output_path=res_dir):

    model_names = list(all_importance.keys())
    feature_names = [d['feature'] for d in all_importance[model_names[0]]]

    matrix = np.zeros((len(feature_names),len(model_names)))
    for j, model_name in enumerate(model_names):
        importance_dict = {d['feature']: d['mean_importance'] for d in all_importance[model_name]}
        for i, feature in enumerate(feature_names):
            matrix[i, j] = importance_dict.get(feature, 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap='coolwarm', aspect='auto')

    ax.set_xticks(range(len(model_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(feature_names, fontsize=8)

    for i in range(len(feature_names)):
        for j in range(len(model_names)):
            val = matrix[i, j]
            colour = 'white' if val > (matrix.max() * 0.5) else 'black'
            ax.text(j, i, f"{val:.3f}", ha='center', va='center', color=colour, fontsize=6)

    plt.colorbar(im, ax=ax, shrink=0.8, label='R² decrease')
    ax.set_title('Feature Importance Heatmap', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'importance_heatmap.png'), dpi=150)
    plt.close()
    logger.info(f"Saved importance heatmap to {output_path}")


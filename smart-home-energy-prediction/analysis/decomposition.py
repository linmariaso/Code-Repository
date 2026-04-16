"""
Oh (2022) Feature power interaction decomposition for energy prediction in smart homes.

For each feature, seaparate the importance into:
- Feature power: how much the feature alone contributes to the prediction
- Interaction power: how much the feature contributes through interactions with other features
- Total importance: Feature power + Interaction power

This method works by comparing the model's predictions under three conditions:
1. Original: All features are present (baseline performance)
2. Target feature permuted: Only the target feature is shuffled, breaking its relationship with the target variable (measures feature power)
3. All other features permuted: All features except the target are shuffled, breaking their relationships

The differences in performance between these conditions allow us to decompose the importance of each feature into its direct contribution and its contribution through interactions.

Output includes:
- results/oh_decomposition.csv
- results/oh_decomposition.png
- results/oh_decomposition_stacked.png

Usage:
    from analysis.decomposition import run_oh_decomposition
    decomposition_results = run_oh_decomposition(trained, X_test, y_test, feature_cols)
    
    Standalone:
    python -m analysis.decomposition
"""
import os
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from config.settings import r_seed, permutation_repeats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

proc_dir = os.path.join('data', 'processed')
res_dir = 'results'
os.makedirs(res_dir, exist_ok=True)

def decompose_feature(model, X_test, y_test, feature_index, feature_name, repeats=50):
    rng = np.random.RandomState(r_seed)
    samples, features = X_test.shape

    base_mae = mean_absolute_error(y_test, model.predict(X_test))

    feature_power_scores = []
    interaction_power_scores = []
    total_importance_scores = []

    for rep in range(repeats):
        X_without_fx = X_test.copy()
        perm = rng.permutation(samples)
        X_without_fx[:, feature_index] = X_test[perm, feature_index]
        mae_without_fx = mean_absolute_error(y_test, model.predict(X_without_fx))

        X_only_fx = X_test.copy()
        for j in range(features):
            if j != feature_index:
                perm_j = rng.permutation(samples)
                X_only_fx[:, j] = X_test[perm_j, j]
                mae_only_fx = mean_absolute_error(y_test, model.predict(X_only_fx))

    total_importance = mae_without_fx - base_mae
    feature_power = total_importance_scores
    interaction = 0.0

    feature_power = base_mae - mae_only_fx
    feature_power = abs(feature_power)

    interaction = max(0.0, total_importance - feature_power)

    feature_power_scores.append(feature_power)
    total_importance_scores.append(total_importance)
    interaction_power_scores.append(interaction)

    return {
        'feature': feature_name,
        'feature_index': feature_index,
        'base_mae': base_mae,
        'feature_power_mean': np.mean(feature_power_scores),
        'feature_power_std': np.std(feature_power_scores),
        'interaction_power_mean': np.mean(interaction_power_scores),
        'interaction_power_std': np.std(interaction_power_scores),
        'total_importance_mean': np.mean(total_importance_scores),
        'total_importance_std': np.std(total_importance_scores),
        'feature_power_ratio': np.mean(feature_power_scores) / max(np.mean(total_importance_scores), 1e-10),
    }

def run_oh_decomposition(trained_res, X_test, y_test, feature_names, repeats = None):
    if repeats is None:
        repeats = permutation_repeats
    logger.info(f"Running Oh decomposition with {repeats} repeats,for features: {feature_names}")
    all_decomposition = {}

    for model_name, res in trained_res.items():
        logger.info(f"Model: {model_name}")
        start = time.time()
        decomposition = []
        for index, feature_name in enumerate(feature_names):
            logger.info(f"  Decomposing feature: {feature_name} ({index+1}/{len(feature_names)})")
            result = decompose_feature(res['estimator'], X_test, y_test, feature_index=index, feature_name=feature_name, repeats=repeats)
            result['model'] = model_name
            model_decomposition.append(result)
    model_decomposition.sort(key=lambda x: x['total_importance_mean'], reverse=True)
    elapsed = time.time() - start
    logger.info(f"Completed Oh decomposition for {model_name} in {elapsed:.1f}s")

    for item in model_decomposition:
        logger.info(f"{item['model']} - {item['feature']}: feature_power={item['feature_power_mean']:.4f}, interaction={item['interaction_power_mean']:.4f}, total={item['total_importance_mean']:.4f}, feature_ratio={item['feature_power_ratio']:.2%}")

    decomposition[model_name] = model_decomposition
    
    return decomposition

def plot_decomposition_bars(decomposition, output_path=res_dir):
    n_models = len(decomposition)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    for ax, (model_name, model_decomposition) in zip(axes, decomposition.items()):
        features = [d['feature'] for d in model_decomposition]
        feature_powers = [d['feature_power_mean'] for d in model_decomposition]
        interaction_powers = [d['interaction_power_mean'] for d in model_decomposition]

        y_pos = np.arange(len(features))
        barh = 0.35

        ax.barh(y_pos-barh/2, feature_powers, height=barh, label='Feature Power', color='blue', alpha=0.8)
        ax.barh(y_pos+barh/2, interaction_powers, height=barh, label='Interaction Power', color='orange', alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=8)
        ax.set_xlabel('MAE contribution')
        ax.set_title(f"{model_name} - Oh Decomposition", fontsize=10)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, axis='x', alpha=0.3)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'oh_decomposition.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Oh decomposition bar plot to {output_path}/oh_decomposition.png")

def plot_decomposition_stacked(decomposition, output_path=res_dir):
    model_name = list(decomposition.keys())
    decomposition_list = decomposition[model_name]
    features = [d['feature'] for d in decomposition_list]
    feature_powers = [d['feature_power_mean'] for d in decomposition_list]
    interaction_powers = [d['interaction_power_mean'] for d in decomposition_list]

    y_pos = np.arange(len(features))

    fig, axes = plt.subplots(figsize=( 8, 5))
    ax.barh(y_pos, feature_powers, label='Feature Power (Independent)', color='blue', alpha=0.8)
    ax.barh(y_pos, interaction_powers, left=feature_powers, label='Interaction Power', color='orange', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=8)
    ax.set_xlabel('MAE contribution')
    ax.set_title(f"{model_name} - Oh Decomposition (Stacked)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()

    for i, (fp, ip) in enumerate(zip(feature_powers, interaction_powers)):
        total = fp + ip
        if total > 0:
            pct = fp / total * 100
            ax.text(total+0.001, i, f"{pct:.0f}% independent", color='gray', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'oh_decomposition_stacked.png'), dpi=150)
    plt.close()
    logger.info(f"Saved Oh decomposition stacked bar plot to {output_path}/oh_decomposition_stacked.png")

def save_decomposition_csv(decomposition, output_path=res_dir):
    rows = []
    for model_name, model_decomposition in decomposition.items():
        for item in model_decomposition:
            rows.append({
                'model': item['model'],
                'feature': item['feature'],
                'total_importance_mean': round(item['total_importance_mean'],6),
                'total_importance_std': round(item['total_importance_std'],6),
                'feature_power_mean': round(item['feature_power_mean'],6),
                'feature_power_std': round(item['feature_power_std'],6),
                'interaction_power_mean': round(item['interaction_power_mean'],6),
                'interaction_power_std': round(item['interaction_power_std'],6),
                'feature_power_ratio': round(item['feature_power_ratio'],4),
                'base_mae': round(item['base_mae'],4)
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_path, 'oh_decomposition.csv'), index=False)
    logger.info(f"Saved Oh decomposition data to {output_path}/oh_decomposition.csv")
    return df

if __name__ == "__main__":
    from pipeline.split import load_split
    from models.train import load_models

    logger.info("Loading test data and trained models for Oh decomposition")
    X_train, y_train, X_test, y_test, scaler, feature_cols = load_split(proc_dir)
    trained = load_models(res_dir)

    decomposition = run_oh_decomposition(trained, X_test, y_test, feature_cols)
    save_decomposition_csv(decomposition, res_dir)
    plot_decomposition_bars(decomposition, res_dir)
    plot_decomposition_stacked(decomposition, res_dir)

    print("Oh decomposition summary:")
    for model_name, model_decomposition in decomposition.items():
        logger.info(f"Model: {model_name}")
        for item in model_decomposition:
            logger.info(f"  {item['feature']}: feature_power={item['feature_power_mean']:.4f}, interaction={item['interaction_power_mean']:.4f}, total={item['total_importance_mean']:.4f}, feature_ratio={item['feature_power_ratio']:.2%}")

    print(f"Decomposition completed. Results saved to {res_dir}")
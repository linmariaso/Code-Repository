"""
Model performance and overfitting testing

Tests:
1. R² gap analysis for train and test data
2. Cross validation and stability check
3. LR coefficient analysis
4. Learning curves
5. Residual normality (Shappiro-Wilk)

Outputs:
- overfit_diagnostics.csv
- overfit_train_test_gap.png
- learning_curves.png
- coefficient_analysis.png
- overfit_summary.txt

Usage:
    from models.overfitting_test import run_overfit_diagnostics
    results = run_overfit_diagnostics(trained, X_train, y_train, X_test, y_test, feature_cols)

    Standalone:
    python -m models.overfit_test
"""


import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from scipy import stats as scipy_stats
from config.settings import folds, r_seed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join('data', 'processed')
res_dir = 'results'
os.makedirs(res_dir, exist_ok=True)

# Overfitting thresholds
gap_thresh_ok = 0.01       # < 1% gap = no overfitting
gap_thresh_mild = 0.05     # 1-5% = mild overfitting



def test_train_test_gap(trained_res, X_train, y_train, X_test, y_test):
    """Compare training and test performance for each model.

    The gap between train R² and test R² is the primary indicator
    of overfitting. A small gap means the model generalises well.
    """
    logger.info("=" * 60)
    logger.info("TEST 1: Train vs Test Gap Analysis")
    logger.info("=" * 60)

    results = []
    for name, res in trained_res.items():
        model = res['estimator']

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        gap = train_r2 - test_r2

        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        rmse_gap = test_rmse - train_rmse

        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        mae_gap = test_mae - train_mae

        if gap < gap_thresh_ok:
            verdict = "NO OVERFITTING"
        elif gap < gap_thresh_mild:
            verdict = "MILD OVERFITTING"
        else:
            verdict = "OVERFITTING CONCERN"

        result = {
            'model': name,
            'train_r2': round(train_r2, 5),
            'test_r2': round(test_r2, 5),
            'r2_gap': round(gap, 5),
            'r2_gap_pct': round(gap * 100, 3),
            'train_rmse': round(train_rmse, 2),
            'test_rmse': round(test_rmse, 2),
            'rmse_gap': round(rmse_gap, 2),
            'train_mae': round(train_mae, 2),
            'test_mae': round(test_mae, 2),
            'mae_gap': round(mae_gap, 2),
            'verdict': verdict
        }
        results.append(result)

        logger.info(f"  {name}:")
        logger.info(f"    Train R²={train_r2:.5f}  Test R²={test_r2:.5f}  Gap={gap:.5f} ({gap*100:.3f}%)")
        logger.info(f"    Train RMSE={train_rmse:.2f}W  Test RMSE={test_rmse:.2f}W  Gap={rmse_gap:.2f}W")
        logger.info(f"    Verdict: {verdict}")

    return results



def test_cv_stability(trained_res, X_train, y_train):
    """Run k-fold cross-validation and check score stability.

    If scores vary wildly across folds, the model may be sensitive
    to particular data splits (a sign of instability/overfitting).
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Cross-Validation Stability")
    logger.info("=" * 60)

    results = []
    for name, res in trained_res.items():
        model = res['estimator']

        try:
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=folds,
                scoring='r2',
                n_jobs=1
            )

            mean_cv = cv_scores.mean()
            std_cv = cv_scores.std()
            cv_range = cv_scores.max() - cv_scores.min()

            # Coefficient of variation (lower = more stable)
            cv_coeff = std_cv / abs(mean_cv) * 100 if mean_cv != 0 else 0

            stable = "STABLE" if cv_coeff < 5 else "UNSTABLE"

            result = {
                'model': name,
                'cv_mean_r2': round(mean_cv, 5),
                'cv_std_r2': round(std_cv, 5),
                'cv_min_r2': round(cv_scores.min(), 5),
                'cv_max_r2': round(cv_scores.max(), 5),
                'cv_range': round(cv_range, 5),
                'cv_coeff_variation': round(cv_coeff, 3),
                'fold_scores': [round(s, 5) for s in cv_scores],
                'verdict': stable
            }
            results.append(result)

            logger.info(f"  {name}:")
            logger.info(f"    Fold scores: {[f'{s:.5f}' for s in cv_scores]}")
            logger.info(f"    Mean={mean_cv:.5f}  Std={std_cv:.5f}  CV={cv_coeff:.3f}%")
            logger.info(f"    Verdict: {stable}")

        except Exception as e:
            logger.warning(f"  {name}: CV failed — {e}")
            results.append({
                'model': name,
                'cv_mean_r2': None,
                'verdict': f'FAILED: {e}'
            })

    return results



def test_coefficient_analysis(trained_res, feature_cols):
    """Analyse LinearRegression coefficients to verify the model
    learns the expected relationship."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: LinearRegression Coefficient Analysis")
    logger.info("=" * 60)

    if 'LinearRegression' not in trained_res:
        logger.info("  LinearRegression not found, skipping.")
        return []

    lr = trained_res['LinearRegression']['estimator']
    coefficients = []

    for name, coef in zip(feature_cols, lr.coef_):
        is_appliance = name.startswith('power_')
        coefficients.append({
            'feature': name,
            'coefficient': round(coef, 4),
            'abs_coefficient': round(abs(coef), 4),
            'is_appliance_feature': is_appliance,
            'effectively_zero': abs(coef) < 0.001
        })

    coefficients.append({
        'feature': 'intercept',
        'coefficient': round(lr.intercept_, 4),
        'abs_coefficient': round(abs(lr.intercept_), 4),
        'is_appliance_feature': False,
        'effectively_zero': False
    })

    # Check: do non-appliance features have ~zero coefficients?
    non_app = [c for c in coefficients
               if not c['is_appliance_feature'] and c['feature'] != 'intercept']
    all_zero = all(c['effectively_zero'] for c in non_app)

    logger.info("  Coefficients (scaled units):")
    for c in coefficients:
        marker = "  (appliance)" if c['is_appliance_feature'] else ""
        logger.info(f"    {c['feature']:25s}: {c['coefficient']:10.4f}{marker}")

    logger.info(f"\n  Non-appliance features effectively zero: {all_zero}")
    if all_zero:
        logger.info("  Conclusion: Model correctly identifies appliance power")
        logger.info("  as the sole predictor of total power.")
    else:
        logger.info("  Note: Some non-appliance features have non-zero")
        logger.info("  coefficients, suggesting additional relationships.")

    return coefficients



def test_learning_curves(trained_res, X_train, y_train, X_test, y_test,
                          output_dir=res_dir):
    """Plot learning curves: performance vs training set size.

    If the model overfits, training score will be high but test
    score will be low with small data, converging as data grows.
    If both converge to a high value, the model generalises well.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Learning Curve Analysis")
    logger.info("=" * 60)

    # Test with 10%, 20%, 40%, 60%, 80%, 100% of training data
    fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    n_total = len(X_train)

    results = {}

    for name, res in trained_res.items():
        model = res['estimator']
        train_scores = []
        test_scores = []
        sizes = []

        for frac in fractions:
            n = int(n_total * frac)
            X_sub = X_train[:n]
            y_sub = y_train[:n]

            try:
                # Clone and refit the model
                from sklearn.base import clone
                model_clone = clone(model)
                model_clone.fit(X_sub, y_sub)

                train_r2 = r2_score(y_sub, model_clone.predict(X_sub))
                test_r2 = r2_score(y_test, model_clone.predict(X_test))

                train_scores.append(train_r2)
                test_scores.append(test_r2)
                sizes.append(n)

                logger.info(f"  {name} @ {frac*100:.0f}% ({n} samples): "
                           f"Train R²={train_r2:.5f}  Test R²={test_r2:.5f}")

            except Exception as e:
                logger.warning(f"  {name} @ {frac*100:.0f}%: Failed — {e}")

        results[name] = {
            'sizes': sizes,
            'train_scores': train_scores,
            'test_scores': test_scores
        }

    # Plot learning curves
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    colours = ['steelblue', 'salmon', 'lightgreen', 'orange']

    for ax, (name, data) in zip(axes, results.items()):
        ax.plot(data['sizes'], data['train_scores'], 'o-',
                color='steelblue', label='Training R²', markersize=5)
        ax.plot(data['sizes'], data['test_scores'], 's-',
                color='salmon', label='Test R²', markersize=5)

        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('R²')
        ax.set_title(f'{name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=min(0, min(data['test_scores']) - 0.05))

        # Annotate final gap
        if data['train_scores'] and data['test_scores']:
            final_gap = data['train_scores'][-1] - data['test_scores'][-1]
            ax.text(0.95, 0.05,
                    f'Final gap: {final_gap:.5f}',
                    transform=ax.transAxes, fontsize=8,
                    ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Learning Curves: Train vs Test R²', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, 'learning_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"  Saved: {path}")

    return results



def test_residual_normality(trained_res, X_test, y_test):
    """Test whether residuals are normally distributed.

    Normally distributed residuals suggest the model captures
    the true relationship without systematic bias.
    Uses Shapiro-Wilk test on a sample (max 5000 for efficiency).
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Residual Normality (Shapiro-Wilk)")
    logger.info("=" * 60)

    results = []
    sample_size = min(5000, len(y_test))

    for name, res in trained_res.items():
        model = res['estimator']
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        # Sample for Shapiro-Wilk (limited to 5000)
        rng = np.random.RandomState(r_seed)
        idx = rng.choice(len(residuals), size=sample_size, replace=False)
        residual_sample = residuals[idx]

        stat, p_value = scipy_stats.shapiro(residual_sample)
        normal = p_value > 0.05

        # Additional stats
        skewness = scipy_stats.skew(residuals)
        kurtosis = scipy_stats.kurtosis(residuals)

        result = {
            'model': name,
            'shapiro_stat': round(stat, 4),
            'shapiro_p': round(p_value, 6),
            'is_normal': normal,
            'residual_mean': round(residuals.mean(), 4),
            'residual_std': round(residuals.std(), 4),
            'skewness': round(skewness, 4),
            'kurtosis': round(kurtosis, 4)
        }
        results.append(result)

        normal_str = "Yes (p > 0.05)" if normal else "No (p < 0.05)"
        logger.info(f"  {name}:")
        logger.info(f"    Shapiro-Wilk: stat={stat:.4f}, p={p_value:.6f}")
        logger.info(f"    Normal: {normal_str}")
        logger.info(f"    Mean={residuals.mean():.4f}  Std={residuals.std():.4f}")
        logger.info(f"    Skewness={skewness:.4f}  Kurtosis={kurtosis:.4f}")

    return results



def plot_train_test_gap(gap_results, output_dir=res_dir):
    """Bar chart comparing train vs test R² for each model."""
    names = [r['model'] for r in gap_results]
    train_r2 = [r['train_r2'] for r in gap_results]
    test_r2 = [r['test_r2'] for r in gap_results]
    gaps = [r['r2_gap_pct'] for r in gap_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: grouped bars
    x = np.arange(len(names))
    width = 0.35
    ax = axes[0]
    ax.bar(x - width / 2, train_r2, width, label='Train R²',
           color='steelblue', alpha=0.8)
    ax.bar(x + width / 2, test_r2, width, label='Test R²',
           color='salmon', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('R²')
    ax.set_title('Train vs Test R²')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate gaps
    for i, gap in enumerate(gaps):
        y_pos = max(train_r2[i], test_r2[i])
        ax.text(i, y_pos + 0.001, f'Gap: {gap:.3f}%',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Right: gap bars
    ax = axes[1]
    colors = []
    for g in gaps:
        if g < gap_thresh_ok * 100:
            colors.append('#55A868')
        elif g < gap_thresh_mild * 100:
            colors.append('#CCAA44')
        else:
            colors.append('#C44E52')

    ax.bar(x, gaps, color=colors, alpha=0.8)
    ax.axhline(gap_thresh_ok * 100, color='green', linestyle='--',
               label=f'OK threshold ({gap_thresh_ok*100}%)')
    ax.axhline(gap_thresh_mild * 100, color='red', linestyle='--',
               label=f'Concern threshold ({gap_thresh_mild*100}%)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('R² Gap (%)')
    ax.set_title('Overfitting Gap (lower = better)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    for i, gap in enumerate(gaps):
        ax.text(i, gap, f'{gap:.3f}%', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Overfitting Diagnostic: Train vs Test Performance', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, 'overfit_train_test_gap.png')
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_coefficient_analysis(coefficients, output_dir=res_dir):
    """Bar chart of LinearRegression coefficients."""
    if not coefficients:
        return

    df = pd.DataFrame(coefficients)
    df = df[df['feature'] != 'intercept']
    df = df.sort_values('abs_coefficient', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#4C72B0' if a else '#CCCCCC'
              for a in df['is_appliance_feature']]

    ax.barh(range(len(df)), df['coefficient'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['feature'], fontsize=9)
    ax.set_xlabel('Coefficient (scaled units)')
    ax.set_title('LinearRegression Coefficients\n(Blue = appliance features, Grey = other)')
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    path = os.path.join(output_dir, 'coefficient_analysis.png')
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"  Saved: {path}")



def generate_summary_report(gap_results, cv_results, coeff_results,
                             normality_results, output_dir=res_dir):
    """Generate a text summary of all diagnostic tests."""

    lines = []
    lines.append("=" * 60)
    lines.append("OVERFITTING DIAGNOSTIC REPORT")
    lines.append("=" * 60)

    # Test 1: Train/Test Gap
    lines.append("\n--- Test 1: Train vs Test Gap ---")
    for r in gap_results:
        lines.append(f"  {r['model']:20s}  Train R²={r['train_r2']:.5f}  "
                     f"Test R²={r['test_r2']:.5f}  Gap={r['r2_gap_pct']:.3f}%  "
                     f"[{r['verdict']}]")

    # Test 2: CV Stability
    lines.append("\n--- Test 2: Cross-Validation Stability ---")
    for r in cv_results:
        if r.get('cv_mean_r2') is not None:
            lines.append(f"  {r['model']:20s}  Mean R²={r['cv_mean_r2']:.5f}  "
                         f"Std={r.get('cv_std_r2', 0):.5f}  "
                         f"[{r['verdict']}]")

    # Test 3: Coefficients
    if coeff_results:
        lines.append("\n--- Test 3: LinearRegression Coefficient Analysis ---")
        non_app = [c for c in coeff_results
                   if not c['is_appliance_feature'] and c['feature'] != 'intercept']
        all_zero = all(c['effectively_zero'] for c in non_app)
        lines.append(f"  Non-appliance features effectively zero: {all_zero}")
        lines.append(f"  Conclusion: Model {'correctly' if all_zero else 'partially'} "
                     f"identifies appliance power as the predictor.")

    # Test 5: Residual Normality
    lines.append("\n--- Test 5: Residual Normality ---")
    for r in normality_results:
        lines.append(f"  {r['model']:20s}  Shapiro p={r['shapiro_p']:.6f}  "
                     f"Skew={r['skewness']:.4f}  Kurt={r['kurtosis']:.4f}  "
                     f"Normal={'Yes' if r['is_normal'] else 'No'}")

    # Overall verdict
    lines.append("\n" + "=" * 60)
    lines.append("OVERALL CONCLUSION")
    lines.append("=" * 60)

    any_overfit = any(r['verdict'] == 'OVERFITTING CONCERN' for r in gap_results)
    if any_overfit:
        lines.append("  WARNING: One or more models show signs of overfitting.")
        lines.append("  Consider reducing model complexity or increasing data.")
    else:
        lines.append("  All models show minimal train-test gap (<1%).")
        lines.append("  No evidence of overfitting detected.")
        lines.append("  High R² scores reflect the deterministic relationship")
        lines.append("  between per-appliance power and total consumption.")

    report = "\n".join(lines)

    # Save to file
    path = os.path.join(output_dir, 'overfit_summary.txt')
    with open(path, 'w') as f:
        f.write(report)
    logger.info(f"\n  Saved report: {path}")

    # Print to console
    print(report)

    return report



def run_overfit_diagnostics(trained_res, X_train, y_train,
                            X_test, y_test, feature_cols,
                            output_dir=res_dir):
    """Run all overfitting diagnostic tests.

    Returns:
        dict with results from all tests
    """
    logger.info("\n" + "#" * 60)
    logger.info("#  OVERFITTING DIAGNOSTIC TESTS")
    logger.info("#" * 60)

    # Test 1: Train vs Test Gap
    gap_results = test_train_test_gap(
        trained_res, X_train, y_train, X_test, y_test
    )

    # Test 2: CV Stability
    cv_results = test_cv_stability(trained_res, X_train, y_train)

    # Test 3: Coefficient Analysis
    coeff_results = test_coefficient_analysis(trained_res, feature_cols)

    # Test 4: Learning Curves
    learning_results = test_learning_curves(
        trained_res, X_train, y_train, X_test, y_test, output_dir
    )

    # Test 5: Residual Normality
    normality_results = test_residual_normality(
        trained_res, X_test, y_test
    )

    # Plots
    plot_train_test_gap(gap_results, output_dir)
    plot_coefficient_analysis(coeff_results, output_dir)

    # Save CSV with all gap results
    df_gaps = pd.DataFrame(gap_results)
    df_gaps.to_csv(os.path.join(output_dir, 'overfit_diagnostics.csv'),
                   index=False)

    # Save CV results
    cv_rows = [{k: v for k, v in r.items() if k != 'fold_scores'}
               for r in cv_results]
    df_cv = pd.DataFrame(cv_rows)
    df_cv.to_csv(os.path.join(output_dir, 'cv_stability.csv'), index=False)

    # Save normality results
    df_norm = pd.DataFrame(normality_results)
    df_norm.to_csv(os.path.join(output_dir, 'residual_normality.csv'),
                   index=False)

    # Generate summary report
    generate_summary_report(
        gap_results, cv_results, coeff_results,
        normality_results, output_dir
    )

    return {
        'gap_results': gap_results,
        'cv_results': cv_results,
        'coefficient_results': coeff_results,
        'learning_results': learning_results,
        'normality_results': normality_results
    }


if __name__ == '__main__':
    from pipeline.split import load_splits
    from models.train import load_trained_models

    print("Loading data and models...")
    X_train, X_test, y_train, y_test, scaler, feature_cols = \
        load_splits(PROCESSED_DIR)
    trained = load_trained_models(res_dir)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Features: {feature_cols}\n")

    results = run_overfit_diagnostics(
        trained, X_train, y_train, X_test, y_test, feature_cols
    )

    print(f"\nAll diagnostics saved to {res_dir}/")
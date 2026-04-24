"""
Runs all the scripts contained for Smart Home Energy Prediction from sensor emulation to model evaluation.

Usage:
- Pipeline with MQTT: python -m run_all.py
- Pipeline without MQTT: python -m run_all.py --skip-mqtt
- Run specific phases: python -m run_all.py --start-from 3 or python -m run_all.py --only 3

Prerrequisites:
- Python with all dependencies installed
- Mosquitto unless its skipped
"""
import os
import sys
import time
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def phase_banner(phase_number, title):
    print("="*100)
    print(f"PHASE {phase_number}: {title}")
    print("="*100)

def phase1_generate():
    phase_banner(1,"Sensor data emulation")
    from emulation.generate_dataset import generate_dataset
    from emulation.sensor_tests import test_summary_stats, test_time_patterns, test_weekday_weekend, test_all_appliances, test_correlations, test_full_timeseries
    from config.settings import days_to_generate

    output_dir = os.path.join('data', 'raw')
    os.makedirs(output_dir, exist_ok=True)

    print("Generating Dataset with emulated data")
    df = generate_dataset()
    dataset_name = 'test_dataset.csv'
    output_path = os.path.join(output_dir, dataset_name)
    df.to_csv(output_path)
    print(f"Dataset {dataset_name} saved to {output_dir}")

    completeness = test_summary_stats(df)
    test_time_patterns(df)
    test_weekday_weekend(df)
    passed, total = test_all_appliances(df)
    test_correlations(df)
    test_full_timeseries(df)

    print("PHASE 1 COMPLETE")
    print(f"Dataset shape: {df.shape} collected in {days_to_generate} days")
    print(f"Completeness: {completeness:.1f}%")
    print(f"Physical consistency: {passed}/{total} checks passed")
    print(f"Data saved in {output_path}")
    print("="*100)

    return df

def phase2_mqtt():
    phase_banner(2, "MQTT Infrastructure")
    from emulation.mqtt_publisher import create_mqtt_client, run_emulation
    from config.settings import mqtt_broker, mqtt_port
    import subprocess
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3)

    try:
        sock.connect(mqtt_broker,mqtt_port)
        sock.close()
        print(f"MQTT reachable at {mqtt_broker}:{mqtt_port}")
    except (ConnectionRefusedError, socket.timeout, OSError):
        print(f"ERROR: MQTT cannot be reached at {mqtt_broker}:{mqtt_port}")
        print("Start Mosquitto first")
        sys.exit(1)

    print("Launching subscriber")
    subscriber_proc = subprocess.Popen([sys.executable, '-m', 'pipeline.subscriber'], stdout= subprocess.PIPE, stderr= subprocess.STDOUT)
    time.sleep(2)

    if subscriber_proc.poll() is not None:
        print("ERROR: Subscriber could not start.")
        sys.exit(1)
    print(f"Subscriber running PID: {subscriber_proc.pid}")

    try:
        print("Starting publisher for simulated mode")
        client = create_mqtt_client()
        client.connect(mqtt_broker,mqtt_port, keepalive=60)
        client.loop_start()

        run_emulation(client, mode='simulated')

        print("Waiting for last messages...")
        time.sleep(5)

        client.loop_stop()
        client.disconnect()
    finally:
        print(f"Stopping subscriber PID: {subscriber_proc.pid}")
        subscriber_proc.terminate()
        subscriber_proc.wait(timeout=10)

    print("PASE 2 COMPLETED")
    print("Raw CSV data saved to data/raw")
    print("="*100)

def phase3_pipeline():
    phase_banner(3, "Data pipeline")

    from pipeline.preprocess import run_preprocessing
    from pipeline.split import prepare_data, save_split

    print("Running preprocessing pipeline")
    df_features = run_preprocessing()

    print(f"Feature matrix: {df_features.shape}")
    print(f"Columns {list(df_features.columns)}")

    print("Preparing train/test splits")
    X_train, y_train, X_test, y_test, scaler, feature_cols = prepare_data(df_features)

    save_split(X_train, y_train, X_test, y_test, scaler, feature_cols)

    print("PHASE 3 COMPLETE")
    print(F"X train shape: {X_train.shape} X Test shape: {X_test.shape}")
    print(f"Features: {feature_cols}")
    print("Data saved to data/processed")
    print("="*100)

def phase4_ml():
    phase_banner(4,"Machine Learning training and evaluation")

    from pipeline.split import load_split
    from models.train import train_all_models, save_models
    from models.evaluate import evaluate_all_models

    proc_dir = os.path.join('data','processed')
    res_dir = 'results'

    print("Loading train/test splits")
    X_train, y_train, X_test, y_test, scaler, feature_cols = load_split(proc_dir)

    print(f"X train shape: {X_train.shape} X test shape: {X_test.shape}")
    
    trained = train_all_models(X_train, y_train)
    save_models(trained)

    eval_result, comparisons = evaluate_all_models(trained, X_test, y_test)

    print("PHASE 4 COMPLETE")
    print(f"\n {'Model':<20} {'R²':>8} {'RMSE':>10} {'MAE':>10}")
    for name, result in eval_result.items():
        print(f"{name:<20} {result['r2']:>8.4f} {result['rmse']:>9.2f}W {result['mae']:>9.2F}W")

    signif = [col for col in comparisons if col['significant']]

    if signif:
        print(f"Significant differences (p<0.05):")
        for col in signif:
            print(f"{col['Model_A']} vs {col['Model_B']}: p={col['p_value']:.4f} → {col['better_model']}")
    best = max(eval_result, key = lambda n:eval_result[n]['r2'])
    print(f"\n Best model: {best} (R² = {eval_result[best]['r2']:.4f})")
    print(f"Results saved to {res_dir}")
    print("="*100)

def phase5_analysis():
    phase_banner(5,"Feature importance and decomposition analysis")

    from pipeline.split import load_split
    from models.train import load_models
    from analysis.feature_importance import run_permutation_importance, save_importance_data, plot_importance_bars, plot_importance_heatmap
    from analysis.decomposition import run_oh_decomposition, save_decomposition_csv, plot_decomposition_bars, plot_decomposition_stacked

    proc_dir = os.path.join('data','processed')
    res_dir = 'results'

    X_train, y_train, X_test, y_test, scaler, feature_cols = load_split(proc_dir)
    trained = load_models(res_dir)

    print("Permutation Importance Analysis")
    all_importance = run_permutation_importance(trained, X_test, y_test, feature_cols)
    save_importance_data(all_importance)
    plot_importance_bars(all_importance)
    plot_importance_heatmap(all_importance)

    print("Oh's Decomposition Analysis")
    all_decomposition = run_oh_decomposition(trained, X_test, y_test, feature_cols)
    save_decomposition_csv(all_decomposition)
    plot_decomposition_bars(all_decomposition)
    plot_decomposition_stacked(all_decomposition)

    print("PHASE 5 COMPLETE")
    for model_name, importance_list in all_importance.items():
        top = importance_list[0]
        print(f"{model_name}: top feature = {top['feature']} (importance = {top['mean_importance']:.4f})")
    print(f"\n Results saved to {res_dir}")
    print("="*100)

def phase6_validation():
    phase_banner(6,"Validation against real world datasets")

    from analysis.validation import run_validation
    res_dir = 'results'

    results = run_validation()

    print("PHASE 6 COMPLETE")

    if results:
        passed = sum(1 for res in results if res['within_tolerance'])
        total = len(results)
        print(f"Features compared: {total}")
        print(f"Within tolerance: {passed}/{total}")
        print(f"Results saved in {res_dir}")
    else:
        print("No real world datasets found to compare")
        print("Download data sets to data/validation and re run")

    print("="*100)

def main():
    parser = argparse.ArgumentParser( description="Smart Home Energy Prediction - Full pipeline")
    parser.add_argument('--skip-mqtt', action='store_true', help='Skip Phase 2 MQTT, Using Phase 1 Only')
    parser.add_argument('--only', type=int, choices=[1,2,3,4,5,6], default=None, help='Run a single phase')
    parser.add_argument('--start-from', type=int, choices=[1,2,3,4,5,6], default=None, help='Run from a specific phase')

    args = parser.parse_args()
    total_start = time.time()

    if args.only:
        phases = [args.only]
    elif args.start_from:
        phases = list(range(args.start_from,7))
    elif args.skip_mqtt:
        phases = [1,3,4,5,6]
    else:
        phases = [1,2,3,4,5,6]

    print("Smart Home Energy prediction pipeline")
    print(f"Phases to run: {phases}")

    if 1 in phases:
        phase1_generate()
    if 2 in phases:
        phase2_mqtt
    if 3 in phases:
        phase3_pipeline()
    if 4 in phases:
        phase4_ml()
    if 5 in phases:
        phase5_analysis()
    if 6 in phases:
        phase6_validation()

    total_time = time.time() - total_start

    print("Pipeline completed")
    print(f"Total time: {total_time:.1f} seconds / {total_time/60:.1f} minutes")
    print("Outputs:")
    print("  data/raw/            — Raw sensor data (CSV)")
    print("  data/processed/      — Features, train/test splits")
    print("  results/             — Trained models, metrics, plots")

if __name__ == '__main__':
    main()
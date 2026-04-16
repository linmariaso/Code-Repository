"""
Run emulation for smart home energy prediction. This script can generate a synthetic dataset or run a real-time MQTT emulation, optionally launching a subscriber process to consume the published messages. It also includes comprehensive testing of the generated dataset to ensure quality and consistency.
Usage:
    # Run publisher only (subscriber running separately):
    python -m emulation.run_emulation

    # Run both publisher and subscriber together:
    python -m emulation.run_emulation --with-subscriber

    # Real-time mode (publishes at actual sensor intervals):
    python -m emulation.run_emulation --realtime
"""
import argparse
import subprocess
import sys
import time
import logging
from emulation.generate_dataset import generate_dataset
from emulation.mqtt_publisher import create_mqtt_client, run_emulation
from emulation.sensor_tests import test_summary_stats, test_time_patterns, test_weekday_weekend, test_correlations, test_full_timeseries, test_all_appliances
from config.settings import output_dir, days_to_generate, mqtt_broker, mqtt_port

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def run_generate_mode(days_to_generate, output_dir):
    
    logger.info(f"Generating dataset for {days_to_generate} days")
    df = generate_dataset()
    df.to_csv(f'{output_dir}/test_dataset.csv')
    logger.info(f"Dataset saved to: {output_dir}/test_dataset.csv")

    # Run tests
    completeness = test_summary_stats(df)
    test_time_patterns(df)
    test_weekday_weekend(df)
    passed, total = test_all_appliances(df)
    test_correlations(df)
    test_full_timeseries(df)

    # Final summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"  Dataset: {len(df)} records over {days_to_generate} days")
    print(f"  Completeness: {completeness:.1f}%")
    print(f"  Physical consistency: {passed}/{total} checks passed")
    print(f"  Plots saved to: {output_dir}/")
    print("=" * 60)
    return df

def check_broker():
    """Check if MQTT broker is running."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((mqtt_broker, mqtt_port))
        s.close()
        logger.info(f"MQTT broker is running at {mqtt_broker}:{mqtt_port}")
        return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        logger.error(f"MQTT broker is not running at {mqtt_broker}:{mqtt_port}\nPlease start the broker and try again.")
        return False
    
def run_mqtt_mode(with_subscriber=False, realtime=False):
    if not check_broker():
        sys.exit(1)
    sub_process = None

    try:
        if with_subscriber:
            logger.info("Launching subscriber process...")
            sub_process = subprocess.Popen([sys.executable, '-m','pipeline.subscriber'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            time.sleep(2)  # Give subscriber time to start
            if sub_process.poll() is not None:
                logger.error("Subscriber process failed to start. Output:\n%s", sub_process.stdout.read().decode())
                sys.exit(1)
            logger.info("Subscriber process running PID: %d", sub_process.pid)

        mode = "real-time" if realtime else "simulated"
        logger.info(f"Generating dataset in {mode} mode...")

        client = create_mqtt_client()
        client.connect(mqtt_broker, mqtt_port, keepalive=60)
        client.loop_start()
        run_emulation(client, mode=mode)
        
        logger.info("Waiting for messages to be processed")
        time.sleep(5)  # Wait for subscriber to process messages

        client.loop_stop()
        client.disconnect()
        logger.info("Emulation completed and MQTT client disconnected.")
    except KeyboardInterrupt:
        logger.info("Emulation interrupted by user.")
    finally:
        if sub_process and sub_process.poll() is None:
            logger.info("Stopping subscriber process PID: %d", sub_process.pid)
            sub_process.terminate()
            try:
                sub_process.wait(timeout=5)
                logger.info("Subscriber process terminated.")
            except subprocess.TimeoutExpired:
                logger.warning("Subscriber process did not terminate in time. Killing it.")
                sub_process.kill()

def main():
    parser = argparse.ArgumentParser(description="Smart Home Energy Prediction — Emulation Runner.")
    parser.add_argument('--mode', choices=['generate', 'mqtt'], default='generate', help="Choose 'generate' to create a dataset or 'mqtt' to run the MQTT emulation.")
    parser.add_argument('--with-subscriber', action='store_true', help="Launch subscriber in background process.")
    parser.add_argument('--realtime', action='store_true', help="Publish at real sensor intervals.")
    args = parser.parse_args()

    if args.mode == 'generate':
        run_generate_mode(days_to_generate, output_dir)
    elif args.mode == 'mqtt':
        run_mqtt_mode(with_subscriber=args.with_subscriber, realtime=args.realtime)

if __name__ == "__main__":
    main()
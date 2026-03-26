
import json
import csv
import os
import signal
import sys
import time
import logging
import threading
from collections import defaultdict
from datetime import datetime
import paho.mqtt.client as mqtt
from config.settings import mqtt_broker, mqtt_port, mqtt_topic_prefix

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
RAW_DATA_DIR = os.path.join('data', 'raw')
STATS_INTERVAL = 30  # Log statistics every N seconds
CSV_HEADERS = ['timestamp', 'room', 'sensor_type', 'appliance', 'value']


class SensorSubscriber:
    """Manages MQTT subscription and writes sensor data to CSV."""

    def __init__(self, broker=mqtt_broker, port=mqtt_port):
        self.broker = broker
        self.port = port
        self.client = None
        self.running = False

        # Thread-safe file handles: {date_str: (file_handle, csv_writer)}
        self._file_handles = {}
        self._write_lock = threading.Lock()

        # Statistics
        self.stats = defaultdict(int)
        self._stats_lock = threading.Lock()
        self._start_time = None

        # Ensure output directory exists
        os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # --- CSV File Management ---

    def _get_writer(self, date_str):
        """Return or create a CSV writer for the given date.

        Each day gets its own file: data/raw/2025-01-06.csv
        Headers are written only when the file is first created.
        """
        if date_str not in self._file_handles:
            filepath = os.path.join(RAW_DATA_DIR, f"{date_str}.csv")
            file_exists = os.path.exists(filepath)
            fh = open(filepath, 'a', newline='', buffering=1)  # line-buffered
            writer = csv.writer(fh)
            if not file_exists:
                writer.writerow(CSV_HEADERS)
                logger.info("Created new daily file: %s", filepath)
            self._file_handles[date_str] = (fh, writer)
        return self._file_handles[date_str][1]

    def _close_all_files(self):
        """Flush and close all open file handles."""
        with self._write_lock:
            for date_str, (fh, _) in self._file_handles.items():
                fh.flush()
                fh.close()
                logger.info("Closed file for %s", date_str)
            self._file_handles.clear()

    # --- MQTT Callbacks ---

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to broker at %s:%s", self.broker, self.port)
            # Subscribe to all topics under the prefix
            topic = f"{mqtt_topic_prefix}/#"
            client.subscribe(topic, qos=1)
            logger.info("Subscribed to: %s", topic)
        else:
            logger.error("Connection failed with code %d", rc)

    def _on_disconnect(self, client, userdata, rc):
        if rc != 0:
            logger.warning("Unexpected disconnect (code %d). Will auto-reconnect.", rc)

    def _on_message(self, client, userdata, msg):
        """Handle every incoming MQTT message."""
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("Bad payload on %s: %s", msg.topic, e)
            with self._stats_lock:
                self.stats['errors'] += 1
            return

        # Handle status messages (publisher online/offline/complete)
        if msg.topic == f"{mqtt_topic_prefix}/status":
            status = payload.get('status', 'unknown')
            logger.info("Publisher status: %s", status)
            with self._stats_lock:
                self.stats['status_messages'] += 1
            if status == 'complete':
                logger.info("Publisher signalled completion. "
                            "Total messages from publisher: %s",
                            payload.get('total_messages', '?'))
            return

        # Extract fields from payload
        timestamp = payload.get('timestamp', '')
        room = payload.get('room', '')
        sensor_type = payload.get('sensor_type', '')
        appliance = payload.get('appliance', '')  # Empty for non-plug sensors
        value = payload.get('value', '')

        # Derive the date for file routing
        date_str = timestamp[:10] if len(timestamp) >= 10 else 'unknown'

        # Write to CSV (thread-safe)
        with self._write_lock:
            writer = self._get_writer(date_str)
            writer.writerow([timestamp, room, sensor_type, appliance, value])

        # Update statistics
        with self._stats_lock:
            self.stats['total_messages'] += 1
            self.stats[f'sensor_{sensor_type}'] += 1
            self.stats[f'room_{room}'] += 1

    # --- Statistics ---

    def _log_stats(self):
        """Periodically log subscription statistics."""
        while self.running:
            time.sleep(STATS_INTERVAL)
            if not self.running:
                break
            with self._stats_lock:
                total = self.stats['total_messages']
                errors = self.stats['errors']
                elapsed = time.time() - self._start_time
                rate = total / elapsed if elapsed > 0 else 0
                logger.info(
                    "Stats | Messages: %d | Errors: %d | Rate: %.1f msg/s | Uptime: %.0fs",
                    total, errors, rate, elapsed
                )

    # --- Lifecycle ---

    def start(self):
        """Connect to the broker and start listening."""
        self.client = mqtt.Client(
            client_id="smart_home_subscriber", clean_session=True
        )
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        # Enable automatic reconnection
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)

        try:
            self.client.connect(self.broker, self.port, keepalive=60)
        except ConnectionRefusedError:
            logger.error(
                "Cannot connect to %s:%s. Is Mosquitto running?",
                self.broker, self.port
            )
            sys.exit(1)

        self.running = True
        self._start_time = time.time()

        # Start stats logging in a background thread
        stats_thread = threading.Thread(target=self._log_stats, daemon=True)
        stats_thread.start()

        logger.info("Subscriber is running. Press Ctrl+C to stop.")
        self.client.loop_forever()

    def stop(self):
        """Graceful shutdown."""
        logger.info("Shutting down subscriber...")
        self.running = False

        # Log final statistics
        with self._stats_lock:
            logger.info("=== Final Statistics ===")
            for key, val in sorted(self.stats.items()):
                logger.info("  %s: %d", key, val)

        self._close_all_files()

        if self.client:
            self.client.disconnect()
            logger.info("Disconnected from broker.")


# --- Entry Point ---

def main():
    subscriber = SensorSubscriber()

    # Handle graceful shutdown on SIGINT (Ctrl+C) and SIGTERM
    def shutdown_handler(signum, frame):
        subscriber.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    subscriber.start()


if __name__ == '__main__':
    main()
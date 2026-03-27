
import json
import time
import logging
import datetime
import paho.mqtt.client as mqtt
from config.settings import (
    mqtt_broker, mqtt_port, mqtt_topic_prefix,
    rooms, appliances,
    temp_interval, hum_interval,
    occ_interval, sp_interval,
    days_to_generate, r_seed
)
from emulation.sensors.temperature import generate_temperature
from emulation.sensors.humidity import generate_humidity
from emulation.sensors.occupancy import generate_occupancy
from emulation.sensors.smart_plug import generate_power

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# --- Connection Callbacks ---

def on_connect(client, userdata, flags, rc):
    """Callback when the client connects to the broker."""
    if rc == 0:
        logger.info("Connected to MQTT broker at %s:%s", mqtt_broker, mqtt_port)
    else:
        logger.error("Connection failed with code %d", rc)

def on_disconnect(client, userdata, rc):
    """Callback when the client disconnects."""
    if rc != 0:
        logger.warning("Unexpected disconnection (code %d). Attempting reconnect...", rc)

def on_publish(client, userdata, mid):
    """Callback when a message is successfully published."""
    userdata['msg_count'] += 1


# --- Publisher Logic ---

def create_mqtt_client():
    """Create and configure an MQTT client with callbacks."""
    client = mqtt.Client(client_id="smart_home_publisher", clean_session=True)
    client.user_data_set({'msg_count': 0})
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_publish = on_publish

    # Optional: set a Last Will & Testament so the subscriber knows
    # if the publisher goes offline unexpectedly
    client.will_set(
        f"{mqtt_topic_prefix}/status",
        payload=json.dumps({"status": "offline", "timestamp": ""}),
        qos=1,
        retain=True
    )
    return client


def publish_reading(client, room, sensor_type, value, timestamp_str):
    """Publish a single sensor reading to the MQTT broker.

    Topic format: home/{room}/{sensor_type}
    Payload: JSON with timestamp, room, sensor_type, value
    """
    topic = f"{mqtt_topic_prefix}/{room}/{sensor_type}"
    payload = json.dumps({
        'timestamp': timestamp_str,
        'room': room,
        'sensor_type': sensor_type,
        'value': value
    })
    result = client.publish(topic, payload, qos=1)
    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        logger.warning("Failed to publish to %s (rc=%d)", topic, result.rc)
    return result


def publish_appliance_reading(client, room, appliance, value, timestamp_str):
    """Publish a single appliance power reading.

    Topic format: home/{room}/appliance/{appliance_name}
    """
    topic = f"{mqtt_topic_prefix}/{room}/appliance/{appliance}"
    payload = json.dumps({
        'timestamp': timestamp_str,
        'room': room,
        'sensor_type': 'smart_plug',
        'appliance': appliance,
        'value': round(value, 2)
    })
    result = client.publish(topic, payload, qos=1)
    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        logger.warning("Failed to publish to %s (rc=%d)", topic, result.rc)
    return result


def run_emulation(client, mode='simulated'):
    """
    Main emulation loop.

    mode='simulated' — Steps through 2 years of data as fast as
        possible, publishing one reading per simulated interval.
        Use this for your dissertation (fast, deterministic).

    mode='realtime' — Publishes at actual sensor intervals.
        Use this to demonstrate live MQTT behaviour.
    """
    import numpy as np
    np.random.seed(r_seed)

    # Announce that the publisher is online
    client.publish(
        f"{mqtt_topic_prefix}/status",
        json.dumps({"status": "online",
                     "timestamp": datetime.datetime.now().isoformat()}),
        qos=1, retain=True
    )

    # We use the smallest interval (SMART_PLUG_INTERVAL = 5s) as the
    # simulation tick, and only fire other sensors when their interval
    # has elapsed.
    tick_seconds = sp_interval
    total_seconds = days_to_generate * 24 * 3600
    total_ticks = total_seconds // tick_seconds

    start_date = datetime.datetime(2023, 1, 1, 0, 0, 0)

    logger.info("Starting emulation: %d days, %d ticks (tick=%ds)",
                days_to_generate, total_ticks, tick_seconds)

    for tick in range(total_ticks):
        elapsed = tick * tick_seconds
        current_time = start_date + datetime.timedelta(seconds=elapsed)
        ts_str = current_time.strftime('%Y-%m-%dT%H:%M:%S')
        hour = current_time.hour
        day_of_week = current_time.weekday()
        is_weekend = day_of_week >= 5

        for room in rooms:
            # --- Temperature (every TEMP_INTERVAL seconds) ---
            if elapsed % temp_interval == 0:
                temp = generate_temperature(room, hour, day_of_week)
                publish_reading(client, room, 'temperature', temp, ts_str)

            # --- Humidity (every HUMIDITY_INTERVAL seconds) ---
            if elapsed % hum_interval == 0:
                # Need current temperature for correlation
                temp_for_humidity = generate_temperature(room, hour, day_of_week)
                hum = generate_humidity(room, hour, temp_for_humidity)
                publish_reading(client, room, 'humidity', hum, ts_str)

            # --- Occupancy (every OCCUPANCY_INTERVAL seconds) ---
            if elapsed % occ_interval == 0:
                occ = generate_occupancy(room, hour, is_weekend)
                publish_reading(client, room, 'occupancy', occ, ts_str)

            # --- Smart Plugs (every tick = SMART_PLUG_INTERVAL) ---
            # Determine if *any* room is occupied (for appliances
            # that depend on someone being home)
            is_anyone_home = any(
                generate_occupancy(r, hour, is_weekend) for r in rooms
            )

            for appliance in appliances:
                power = generate_power(
                    appliance, hour, is_anyone_home
                )
                publish_appliance_reading(
                    client, room, appliance, power, ts_str
                )

        # --- Progress logging ---
        if tick % 10000 == 0 and tick > 0:
            pct = (tick / total_ticks) * 100
            msg_count = client._userdata['msg_count']
            logger.info("Progress: %.1f%% | Messages published: %d", pct, msg_count)

        # In real-time mode, sleep between ticks
        if mode == 'realtime':
            time.sleep(tick_seconds)

    # Final stats
    msg_count = client._userdata['msg_count']
    logger.info("Emulation complete. Total messages published: %d", msg_count)

    # Announce completion
    client.publish(
        f"{mqtt_topic_prefix}/status",
        json.dumps({"status": "complete",
                     "total_messages": msg_count,
                     "timestamp": datetime.datetime.now().isoformat()}),
        qos=1, retain=True
    )


# --- Entry Point ---

if __name__ == '__main__':
    client = create_mqtt_client()
    try:
        client.connect(mqtt_broker, mqtt_port, keepalive=60)
        client.loop_start()  # Background thread for network I/O
        run_emulation(client, mode='simulated')
    except ConnectionRefusedError:
        logger.error(
            "Could not connect to broker at %s:%s. "
            "Is Mosquitto running? Start it with: mosquitto -v",
            mqtt_broker, mqtt_port
        )
    except KeyboardInterrupt:
        logger.info("Emulation interrupted by user.")
    finally:
        client.loop_stop()
        client.disconnect()
        logger.info("Publisher shut down cleanly.")
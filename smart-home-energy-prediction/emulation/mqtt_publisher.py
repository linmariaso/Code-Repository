import json
import time
import paho.mqtt.client as mqtt

from config.settings import mqtt_broker, mqtt_port, mqtt_topic_prefix

def publish_mqtt(client, room, sensor_type, value):
    topic = f"{mqtt_topic_prefix}/{room}/{sensor_type}"
    payload = json.dumps({
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'room': room,
        'sensor_type': sensor_type,
        'value': value
    })
    client.publish(topic, payload)
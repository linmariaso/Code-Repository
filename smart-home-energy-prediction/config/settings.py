# Configuration File
from datetime import datetime

#Output directory
output_dir = 'results/'
# Emulation Parameters
rooms = ['bedroom','living_room','bathroom','kitchen','studio']
days_to_generate = 730
sample_freq = 5 #minutes
start_time = datetime(2023,1,1,0,0)
intervals = days_to_generate *24 * (60//sample_freq)
anomaly_rate = 0.20
anomaly_rate_active = 0.10
anomaly_rate_inactive = 0.015
r_seed = 42

# Sensor intervals (Seconds)
temp_interval = 30
hum_interval = 30
occ_interval = 10
sp_interval = 5

# Anomalies
an_rate = 0.02

# MQTT
mqtt_broker = 'localhost'
mqtt_port = 1883
mqtt_topic_prefix = 'home'

# Appliances
appliances = ['washing_machine','refrigerator','television','microwave','kettle','computer','dishwasher','lighting']

# Machine Learning settings
test_ratio = 0.2
folds = 5

# Evaluation metrics
r2_target = 0.70
rmse_target = 100
mae_target = 70
significance = 0.05
permutation_repeats = 50

# Validation
validation_tolerance = 0.20
"""
Generate synthetic dataset for smart home energy prediction, including occupancy, temperature, humidity, and appliance power consumption. The dataset is generated based on realistic patterns and correlations between variables. The generated dataset is saved as a CSV file and undergoes a series of tests to ensure its quality and realism.
The dataset includes:
- Timestamp, hour, day of week, and weekend indicator
- Temperature and humidity for each room
- Occupancy status for each room
- Power consumption for each appliance
- Derived features like mean temperature, mean humidity, occupancy count, active appliances, and total power
Output:
- A CSV file containing the synthetic dataset
Usage:
    from emulation.generate_dataset import generate_dataset
    Standalone:
    python -m emulation.generate_dataset
"""
import pandas as pd
import numpy as np
from datetime import timedelta

from matplotlib.pylab import record
from emulation.sensors.temperature import generate_temperature
from emulation.sensors.humidity import generate_humidity
from emulation.sensors.occupancy import generate_occupancy
from emulation.sensors.smart_plug import generate_power, apply_anomalies
from emulation.sensor_tests import test_summary_stats, test_time_patterns, test_weekday_weekend, test_correlations, test_full_timeseries, test_all_appliances
from config.settings import r_seed, intervals, output_dir, rooms, appliances, days_to_generate
from config.settings import sample_freq, start_time, anomaly_rate_active, anomaly_rate_inactive

def generate_dataset():
    records = []

    for i in range (intervals):
       t_s = start_time + timedelta(minutes=i*sample_freq)
       hour = t_s.hour
       day_of_week = t_s.weekday()
       is_weekend = day_of_week >= 5
       occupancy = {r: generate_occupancy(r, hour, is_weekend) for r in rooms}
       home_occupation = any(occupancy.values())
       temperatures = {}
       humidities = {}
       for room in rooms:
        t = generate_temperature(room, hour, day_of_week)
        temperatures[room] = t
        humidities[room] = generate_humidity(room, hour, t)


       powers = {}
       for appliance in appliances:
         p = generate_power(appliance, hour, home_occupation, is_weekend) # Call generate_power for each appliance
         p = apply_anomalies(appliance, p) # Apply anomalies
         powers[appliance] = round(p,1)

       record = {
         'timestamp': t_s,
         'hour': hour,
         'day_of_week': day_of_week,
         'is_weekend': int(is_weekend)
         }
       for room in rooms:
        record[f'temperature_{room}'] = temperatures[room]
        record[f'humidity_{room}'] = humidities[room]
        record[f'occupancy_{room}'] = occupancy[room]

       for appliance in appliances:
         record[f'power_{appliance}'] = powers[appliance]

       record['mean_temperature'] = np.mean(list(temperatures.values()))
       record['mean_humidity'] = np.mean(list(humidities.values()))
       record['occupancy_count'] = sum(occupancy.values())
       record['active_appliances'] = sum(1 for p in powers.values() if p > 10)
       record['total_power'] = round(sum(powers.values()),1)
       records.append(record)
       
    df = pd.DataFrame(records)
    df.set_index('timestamp', inplace=True)
    return df
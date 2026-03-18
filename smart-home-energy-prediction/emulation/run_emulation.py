#Generate occupancy, temperature, humidity and appliances

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import timedelta, datetime

output_dir = 'results/sensor-tests'
r_seed= 42
rooms = ['bedroom','living_room','bathroom','kitchen','studio']
appliances = ['washing_machine','refrigerator','television','microwave','kettle','computer','dishwasher','lighting']
days_to_generate = 730
sample_freq = 5 #minutes
start_time = datetime(2023,1,1,0,0)
intervals = days_to_generate *24 * (60//sample_freq)
anomaly_rate = 0.20
anomaly_rate_active = 0.10
anomaly_rate_inactive = 0.015
records = []
np.random.seed(r_seed)
os.makedirs(output_dir, exist_ok=True)

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
    t = generate_temp(room, hour, day_of_week)
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
  df.to_csv('emulation.csv')

  completeness = test_summary_stats(df)
  test_time_patterns(df)
  test_weekday_weekend(df)
  passed, total = test_physical_consistency(df)
  test_correlations(df)
  test_full_timeseries(df)
  test_all_appliances(df)

  # Final summary
  print("\n" + "=" * 60)
  print("TEST SUMMARY")
  print("=" * 60)
  print(f"  Dataset: {len(df)} records over {days_to_generate} days")
  print(f"  Completeness: {completeness:.1f}%")
  print(f"  Physical consistency: {passed}/{total} checks passed")
  print(f"  Plots saved")
  print("=" * 60)
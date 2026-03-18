import numpy as np

appliance_power = {
    'washing_machine':  {'base': 0, 'active': 500, 'always_on': False},
    'refrigerator':     {'base': 50, 'cycle': 120, 'always_on': True},
    'television':       {'base': 5, 'active': 100, 'always_on': False},
    'microwave':        {'base': 0, 'active': 800, 'always_on': False},
    'kettle':           {'base': 0, 'active': 2200, 'always_on': False},
    'computer':         {'base': 5, 'active': 150, 'always_on': False},
    'dishwasher':       {'base': 0, 'active': 1200, 'always_on': False},
    'lighting':         {'base': 0, 'active': 60, 'always_on': False}
}
appliance_schedules = {
    'washing_machine':  {
        'default': 0.0,
        'weekday': { 9: 0.03, 10: 0.05, 11: 0.03, 14: 0.02},
        'weekend': { 9: 0.06, 10: 0.08, 11: 0.06, 14: 0.04}
    },
    'television':       {
        'default': 0.0,
        'weekday': { 12: 0.05, 13: 0.05, 18: 0.15, 19: 0.40, 20: 0.50, 21: 0.50, 22: 0.30, 23: 0.10},
        'weekend': { 10: 0.10, 11: 0.15, 12: 0.15, 13: 0.10, 14: 0.10, 15: 0.10, 18: 0.20, 19: 0.50, 20: 0.60, 21: 0.55, 22: 0.40, 23: 0.15}
},
    'microwave':        {
        'default': 0.0,
        'weekday': { 7: 0.04, 13: 0.12, 18: 0.08, 19: 0.04},
        'weekend': { 9: 0.04, 13: 0.10, 14: 0.06, 18: 0.08, 19: 0.04}
        },
    'kettle':           {
        'default': 0.0,
        'weekday': { 8: 0.06, 13: 0.06, 16: 0.06, 18: 0.04, 20: 0.03},
        'weekend': { 8: 0.06, 9:0.10, 11: 0.04, 13: 0.06, 15: 0.06, 18:0.06, 20: 0.04}
    },
    'computer':         {
        'default': 0.0,
        'weekday': { 9: 0.60, 10: 0.70, 11: 0.70, 13: 0.70, 14: 0.70, 15: 0.60, 16: 0.50, 20: 0.10, 21: 0.10},
        'weekend': { 10: 0.10, 11: 0.10, 14: 0.10, 15:0.10, 20: 0.15, 21: 0.15}
    },
    'dishwasher':{
        'default': 0.0,
        'weekday': { 20: 0.06, 21: 0.04},
        'weekend': { 13:0.03, 20: 0.06, 21: 0.04}
    },
    'lighting':         {
        'default': 0.0,
        'weekday': { 6: 0.20, 7: 0.40, 8:0.20, 17: 0.30, 18: 0.50, 19: 0.60, 20: 0.60, 21: 0.50, 22: 0.30},
        'weekend': { 8: 0.15, 9: 0.20, 17: 0.20, 18: 0.50, 19: 0.60, 20: 0.60, 21: 0.55, 22: 0.30, 23: 0.15}
    }
}
appliance_max = {
    'washing_machine':2500,
    'refrigerator':300,
    'television':250,
    'microwave':1500,
    'kettle':2400,
    'computer':500,
    'dishwasher':2500,
    'lighting':200
}
phantom_loads = {
        'washing_machine': 0,
        'refrigerator': 0,
        'television': 15,
        'microwave': 0,
        'kettle': 0,
        'computer': 20,
        'dishwasher': 0,
        'lighting': 10
    }
short_appliance = {'microwave': 0.2, 'kettle': 0.3}

def activation_probability(appliance, hour, weekend):
    schedule = appliance_schedules.get(appliance, {})
    default = schedule.get('default', 0.0)

    if weekend and 'weekend' in schedule:
        prob = schedule['weekend']
    else:
      prob = schedule.get('weekday',{})

    return prob.get(hour, default)

def generate_power (appliance, hour, occupancy, weekend=False, noise=5.0):
    behaviour = appliance_power[appliance]
    max_power = appliance_max.get(appliance,3000)

    if behaviour['always_on']:
        spike_power = behaviour.get('cycle', behaviour['base'])
        power = spike_power if np.random.rand() < 0.3 else behaviour['base']
        power += np.random.normal(0, noise)
        return np.clip(power, 0, max_power)

    if not occupancy:
        return float(behaviour['base'])  # Minimal power when off

    activation_prob = activation_probability(appliance, hour, weekend)

    if np.random.rand() < activation_prob:
      power = behaviour['active'] + np.random.normal(0, noise)

      if appliance in short_appliance:
        power *= short_appliance[appliance]
      return np.clip(power, 0, max_power)

    return float(behaviour['base'])  # Minimal power when not active

# Anomalies function, made a separate top-level function
def apply_anomalies(appliance, power):
  
  max_power = appliance_max.get(appliance,3000)
  if power > 10:
    if np.random.rand() < anomaly_rate_active:
      power *= np.random.uniform(1.5,3.0)
      return min(power, max_power)
  else:
    phantom = phantom_loads.get(appliance,0)
    if phantom > 0 and np.random.rand() < anomaly_rate_inactive:
      return (phantom + np.random.normal(0, 3))
  
  return power
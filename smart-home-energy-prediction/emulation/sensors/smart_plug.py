import numpy as np

appliance_power = {
    'washing_machine': {'base': 0, 'active': 500, 'always_on': False},
    'refrigerator': {'base': 50, 'cycle': 120, 'always_on': True},
    'television': {'base': 5, 'active': 100, 'always_on': False},
    'microwave': {'base': 0, 'active': 800, 'always_on': False},
    'kettle': {'base': 0, 'active': 2200, 'always_on': False},
    'computer': {'base': 5, 'active': 150, 'always_on': False},
    'dishwasher': {'base': 0, 'active': 1200, 'always_on': False},
    'lighting': {'base': 0, 'active': 60, 'always_on': False}
}

def activation_appliance(appliance, hour):
    prob = {
        'washing_machine': {9: 0.3, 10: 0.4, 11: 0.2, 14: 0.2, 18: 0.5, 19: 0.4, 20: 0.3},
        'television': {18: 0.3, 19: 0.7, 20: 0.8, 21: 0.8, 22: 0.5},
        'microwave': {7: 0.2, 8: 0.3, 13: 0.4, 18: 0.3},
        'kettle': {7: 0.6, 8: 0.5, 13: 0.3, 16: 0.3, 18: 0.5},
        'computer': {9:0.7, 10:0.8, 11:0.8, 12:0.8, 15:0.8, 16:0.7, 17:0.6, 18:0.6},
        'dishwasher': {7:0.3, 8:0.4, 20:0.4, 21:0.3},
        'lighting': {6:0.3, 7:0.5, 17:0.3, 18:0.6, 19:0.8, 20:0.8, 21:0.7, 22:0.6}
    }
    return prob.get(appliance, {}).get(hour, 0.05)

def generate_power (appliance, hour, occupancy, noise=5.0):
    behaviour = appliance_power[appliance]
    
    if behaviour['always_on']:
        base = behaviour['base']
        if np.random.rand() < 0.3:  # Simulate occasional power spikes
            base = behaviour['cycle']
        return max(0, base + np.random.normal(0, noise))
    
    if not occupancy:
        return behaviour['base']  # Minimal power when off
    
    activation = activation_appliance(appliance, hour)
    if np.random.rand() < activation:
        return max(0, behaviour['active'] + np.random.normal(0, noise))
    return behaviour['base']  # Minimal power when not active
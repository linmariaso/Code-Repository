"""
Occupancy sensor emulation based on time of day and day of week for every room.
This module generates realistic occupancy patterns for different rooms in a smart home. It incorporates:
- Room-specific occupancy probabilities based on typical human behavior
- Time-of-day patterns (e.g., higher occupancy in living room evenings, bedroom at night)
- Weekday vs weekend differences (e.g., more daytime occupancy on weekends)
Usage:
    from emulation.sensors.occupancy import generate_occupancy
    occupancy = generate_occupancy(room, hour, is_weekend)
    Standalone:
    python -m emulation.sensors.occupancy
"""
import numpy as np

prob_weekend_occupancy = {
    'bedroom':[0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.8,0.5,0.2,0.1,0.0,0.0,0.0,0.1,0.1,0.0,0.0,0.0,0.1,0.2,0.4,0.7,0.9],
    'living_room':[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.2,0.4,0.5,0.5,0.4,0.5,0.5,0.4,0.5,0.6,0.8,0.9,0.8,0.7,0.4,0.1],
    'bathroom':[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.3,0.4,0.2,0.1,0.0,0.0,0.1,0.1,0.1,0.1,0.2,0.3,0.3,0.3,0.2,0.1],
    'kitchen':[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.3,0.5,0.3,0.2,0.5,0.3,0.1,0.1,0.2,0.4,0.6,0.3,0.2,0.1,0.0,0.0],
    'studio':[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.1,0.1,0.0,0.1,0.1,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
}

prob_weekday_occupancy = {
    'bedroom':[0.9,0.9,0.9,0.9,0.9,0.9,0.7,0.3,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.2,0.4,0.7,0.9],
    'living_room':[0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.3,0.2,0.1,0.1,0.1,0.2,0.1,0.1,0.1,0.2,0.4,0.7,0.8,0.9,0.8,0.5,0.2],
    'bathroom':[0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.5,0.3,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.1,0.2,0.3,0.4,0.2,0.1],
    'kitchen':[0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.6,0.4,0.1,0.0,0.1,0.5,0.2,0.0,0.0,0.1,0.3,0.7,0.4,0.2,0.1,0.0,0.0],
    'studio':[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.7,0.8,0.8,0.5,0.8,0.8,0.8,0.7,0.3,0.1,0.0,0.0,0.0,0.0,0.0]
}

def generate_occupancy(room, hour, is_weekend):
    if is_weekend:
        prob_occupancy = prob_weekend_occupancy
    else:
        prob_occupancy = prob_weekday_occupancy

    prob = prob_occupancy[room][hour]
    
    return 1 if np.random.rand() < prob else 0
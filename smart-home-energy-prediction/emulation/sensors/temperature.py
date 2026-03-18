import numpy as np
from config import settings

np.random.seed(settings.random_seed)

# Baselines by room

room_temperature= {
    'bedroom':      {'base':18.0,'amplitude':1.5}, 
    'living_room':  {'base':20.0,'amplitude':2.0},
    'bathroom':     {'base':22.0,'amplitude':2.5},
    'kitchen':      {'base':21.0,'amplitude':3.0},
    'studio':       {'base':20.5,'amplitude':2.0},
}

def generate_temp(room, hour,day_of_week,noise=0.5):
    behaviour = room_temperature[room]
    # Coolest 5am, warmest 3pm
    day = behaviour['amplitude']*np.sin((hour-5)*np.pi/12)

    #Kitchen behaviour during meals 7-9, 13-14, 18-20
    k_boost = 0
    if room == 'kitchen' and hour in [7, 8, 13, 18, 19]:
        k_boost = np.random.uniform(1.0, 3.0)

    # Bathroom behaviour mornings evenings
    b_boost = 0
    if room == 'bathroom' and hour in [7, 8, 21, 22]:
        b_boost = np.random.uniform(1.0, 2.0)

    temp = behaviour['base']+ day + k_boost + b_boost
    temp += np.random.normal(0,noise)

    return(round(temp,1))


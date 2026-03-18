import numpy as np

room_humidity = {
    'bedroom':      {'base':40, 'amplitude':5}, 
    'living_room':  {'base':45, 'amplitude':5},
    'bathroom':     {'base':55, 'amplitude':15},
    'kitchen':      {'base':50, 'amplitude':10},
    'studio':       {'base':42, 'amplitude':5},
}

def generate_humidity(room, hour, temperature, noise=2.0):
    behaviour = room_humidity(room)
    
    # Temperature-Humidity correlation: higher humidity when warmer
    t_h = -0.5*(temperature-20)

    #Kitchen behaviour during meals 7-9, 13-14, 18-20
    h_boost = 0
    if room == 'kitchen' and hour in [7, 8, 13, 18, 19]:
        h_boost = np.random.uniform(3, 8)

    # Bathroom behaviour mornings evenings
    if room == 'bathroom' and hour in [7, 8, 21, 22]:
        h_boost = np.random.uniform(5, 15)

    humidity = behaviour['base']+ t_h + h_boost
    humidity += np.random.normal(0,noise)
    humidity = np.clip(humidity, 20, 95)  # Realistic humidity range

    return(round(humidity,1))
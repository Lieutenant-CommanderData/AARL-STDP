import numpy as np
from random import randint

''' 
Models for Emulating Ia Neurons
'''

BASERATE = 82 # Hz, Base Spike Frequency
BASERATE_CURRENT = 1.09 # Amount of current to achieve ~82Hz

# Equation relating input current to output spike frequency. Modeled in "Plot_CurrentFrequencyRelation" code
def freq2cur(freq):
    T_fit = 0.00499132 # From equation fit to experimental data for neuron firing rate based on current
    x = -1 / (T_fit * freq)
    x = np.clip(x, -700, 700)
    return 1 / (-1*np.exp(x) + 1)

''' Emulates type Ia feedback converting length/velocity data into current to be sent to sensory neuron '''
def vel2cur(length, velocity, current_time):
    percent_noise = 0.001
    # Convert to mm for the equation
    length = length * 1000 
    velocity = velocity * 1000
    current = np.zeros(len(length))

    global BASERATE_CURRENT_NOISY

    if current_time % 10 == 0:
        BASERATE_CURRENT_NOISY = np.zeros(len(length))
        for neuron in range(len(length)):
            BASERATE_CURRENT_NOISY[neuron] = randint(int((BASERATE_CURRENT * (1 - percent_noise)) * 10000), int((BASERATE_CURRENT * (1 + percent_noise)) * 10000)) / 10000

    for neuron in range(len(length)):
        if abs(velocity[neuron]) >= 1:
            # Calculate the current to be sent to the neuron
            # Maintain signs for negatives raised to a power
            velocity_sign = np.sign(velocity[neuron])
            ''' Prochazka's Equation. Desired spiking frequency '''
            Ia_Spike_Freq = (velocity_sign * 4.3 * abs(velocity[neuron])**0.6) + (2 * length[neuron]) + BASERATE
            # print('Ia Spike Frequency: ' + str(Ia_Spike_Freq))
            
            # Calculate the current that alligns to the desired spike frequency
            current[neuron] = freq2cur(Ia_Spike_Freq)
            if current[neuron] < (BASERATE_CURRENT * (1 - 0.025)):
                current[neuron] = BASERATE_CURRENT_NOISY[neuron]
        else:
            current[neuron] = BASERATE_CURRENT_NOISY[neuron]

    return current

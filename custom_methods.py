''' Standard Imports '''
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

''' SNS Imports '''
# from NetworkGeneration import STDPNetworkGenerator
# from NetworkGeneration_MN import MNNetworkGenerator
from sns_toolbox.renderer import render
from sns_toolbox.neurons import SpikingNeuron
from sns_toolbox.connections import SpikingSynapse
from sns_toolbox.networks import Network
from sns_toolbox.renderer import render

''' MuJoCo Imports '''
import mujoco as mj
import mediapy as media

''' Other Misc Imports '''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from random import Random, randint

''' Make the plots pretty and slow '''
mpl.rcParams['figure.dpi'] = 600 # 600 is pretty high definition

# Base spiking rate of 80Hz (Current corresponding to the spiking rate)
BASERATE = 82 # Hz
BASERATE_CURRENT = 1.09 # Old method. Amount of current to achieve ~82Hz

# Takes spike data string, returns frequency of spikes
def getFreq(data, dt):
    loc = np.where(data == 1)[0]
    freq = []
    for i in range(len(loc)-1):
        freq.append(1 / ((loc[i+1] - loc[i]) * (dt * 0.001)))
    # The first element needs to be removed because this is inner spike, so more spikes than frequencies recorded
    loc = loc[1:] * dt
    return freq, loc


def manualActivation(t, num_motors, active):
    interneuron_i = np.zeros(shape=[len(t), num_motors])

    if active & 0b0001:
        interneuron_i[:, 0] += 1
    if active & 0b0010:
        interneuron_i[:, 1] += 1
    if active & 0b0100:
        interneuron_i[:, 2] += 1
    if active & 0b1000:
        interneuron_i[:, 3] += 1
    return interneuron_i

def manualActivation2(t, split_ms):
    i_IN1 = np.zeros([len(t)])
    i_IN2 = np.zeros([len(t)])
    i_IN3 = np.zeros([len(t)])
    i_IN4 = np.zeros([len(t)])

    active = 1

    for i in range(len(t)):
        if active == 0:
            i_IN1[i] += 1
            i_IN2[i] += 1

        if active == 1:
            i_IN3[i] += 1
            i_IN4[i] += 1

        if t[i] % split_ms == 0:
            active = not active

    return i_IN1, i_IN2, i_IN3, i_IN4

def manualActivation3(t, on_time, off_time):
    i_IN1 = np.zeros([len(t)])
    i_IN2 = np.zeros([len(t)])
    i_IN3 = np.zeros([len(t)])
    i_IN4 = np.zeros([len(t)])

    # Converting from ms on and off time to timestep on and off time
    on_time = on_time * 10
    off_time = off_time * 10

    active = 0
    on_counter = 0
    off_counter = 0

    ts1 = 0
    ts2 = 0
    ts3 = 0
    ts4 = 0

    for i in range(len(t)):
        if active == 0:
            i_IN1[i] += 1
            i_IN2[i] += 1
            ts1 += 1
        if active == 1:
            ts2 += 1
        if active == 2:
            i_IN3[i] += 1
            i_IN4[i] += 1
            ts3 += 1
        if active == 3:
            ts4 += 1

        if active == 0 or active == 2:
            on_counter += 1
        if active == 1 or active == 3:
            off_counter += 1

        if on_counter == on_time:
            on_counter = 0
            active += 1
        if off_counter == off_time:
            off_counter = 0
            active += 1

        if active >= 4:
            active = 0

    return i_IN1, i_IN2, i_IN3, i_IN4


def randActivation(t, act_freq):
    i_IN1 = np.zeros([len(t)])
    i_IN2 = np.zeros([len(t)])
    i_IN3 = np.zeros([len(t)])
    i_IN4 = np.zeros([len(t)])

    act = 0b00

    for i in range(len(t)):
        if i % act_freq == 0:
            if act & 0b10:
                i_IN1[i-act_freq:i] += 1
            if act & 0b10:
                i_IN2[i-act_freq:i] += 1
            if act & 0b01:
                i_IN3[i-act_freq:i] += 1
            if act & 0b01:
                i_IN4[i-act_freq:i] += 1

            act = randint(0, 3)

    return i_IN1, i_IN2, i_IN3, i_IN4


''' Need to update such that there is a dynamic number of motor neurons '''
def randActivation2(t, pair_freq, indi_freq, num_motors):
    i_IN1 = np.zeros([len(t)])
    i_IN2 = np.zeros([len(t)])
    i_IN3 = np.zeros([len(t)])
    i_IN4 = np.zeros([len(t)])

    # Initilize at 0b00
    pair_act = 0b00
    indi_act1 = 0b00
    indi_act2 = 0b00

    for i in range(len(t)):
        if pair_act & 0b01:
            if indi_act1 & 0b01:
                i_IN1[i] += 1
            if indi_act1 & 0b10:
                i_IN2[i] += 1

        if pair_act & 0b10:
            if indi_act2 & 0b01:
                i_IN3[i] += 1
            if indi_act2 & 0b10:
                i_IN4[i] += 1

        if i % pair_freq == 0:
            pair_act = randint(0, 3)

        if i % indi_freq == 0:
            indi_act1 = randint(0, 3)
            indi_act2 = randint(0, 3)

    return i_IN1, i_IN2, i_IN3, i_IN4

def randActivation3(t, pair_freq, indi_freq, num_motors):
    interneuron_i = np.zeros(shape=[len(t), num_motors])

    # Inilize at 0b00
    pair_act = 0b00
    indi_act1 = 0b00
    indi_act2 = 0b00

    for i in range(len(t)):
        if pair_act & 0b01:
            if indi_act1 & 0b01:
                interneuron_i[i, 0] += 1
            if indi_act1 & 0b10:
                interneuron_i[i, 1] += 1

        if pair_act & 0b10:
            if indi_act2 & 0b01:
                interneuron_i[i, 2] += 1
            if indi_act2 & 0b10:
                interneuron_i[i, 3] += 1

        if i % pair_freq == 0:
            pair_act = randint(0, 3)

        if i % indi_freq == 0:
            indi_act1 = randint(0, 3)
            indi_act2 = randint(0, 3)

    return interneuron_i

def randConnections(pre_num, post_num, seed=None, g_min=1.50, g_max=5.00):
    if seed is not None:
        rng = Random(x=seed)
    else:
        rng = Random()
    '''
    For Conductances
    2.99 is a barely spikes the post when paired.
    g_max is for solo spiking.
    '''
    g_min = int(g_min * 100) # Convert to int
    g_max = int(g_max * 100) # Convert to int

    # Empty matrix for conductivity parameters
    matrix = np.zeros([pre_num + post_num, pre_num + post_num])

    for pre in range(0, pre_num):
        for post in range(pre_num, pre_num + post_num):
            matrix[post, pre] = (rng.randint(g_min, g_max) / 100)

    return matrix

''' Produces 8 random conductances for the 8 connections between the INs and the POSTs'''
def randConnectionCond(seed='AUGGGH', g_min=2.99, g_max=5.98):
    rng = Random(x=seed)
    '''
    For Conductances
    2.99 is a barely spikes the post when paired.
    g_max is for solo spiking.
    '''
    g_min = int(g_min * 100) # Convert to int
    g_max = int(g_max * 100) # Convert to int

    # Generating two arrays for the 2 connection arrays needed for g_update
    g_POST1 = []
    g_POST2 = []

    for i in range(4):
        g_POST1.append(rng.randint(g_min, g_max) / 100) # Get rand int, convert back to float
        g_POST2.append(rng.randint(g_min, g_max) / 100)
    for i in range(2):
        g_POST1.append(0.0)
        g_POST2.append(0.0)

    return g_POST1, g_POST2

''' Adds noise to current signals '''
def NoisyAmps(smooth, percent_noise=float):
    percent_noise = percent_noise / 100
    noisy = np.copy(smooth)
    time, motor = smooth.shape
    for mn in range(motor):
        for i in range(time):
            if smooth[i, mn] > 1.0:
                noisy[i, mn] = randint(int((smooth[i, mn] * (1 - percent_noise)) * 10000), int((smooth[i, mn] * (1 + percent_noise)) * 10000)) / 10000
    return noisy

# Equation relating input current to output spike frequency. Modeled in "Plot_CurrentFrequencyRelation" code
def freq2cur(freq):
    T_fit = 0.00499132 # From equation fit to experimental data for neuron firing rate based on current
    return 1 / (-1*np.exp(-1 / (T_fit * freq)) + 1)

''' Takes a scalar velocity feedback from MuJoCo and converts to scalar current value '''
# Takes the 0 to 1.0 velocity readings and returns 0 to 1.1 nA
def vel2curX(velocity):
    if velocity < 0:
        return 1.000
    else:
        return (1.000 + 0.1*velocity)

''' Emulates type Ia feedback converting length/velocity data into current to be sent to sensory neuron '''
def vel2curX(length, velocity):
    # Convert to mm for the equation
    length = length * 1000
    velocity = velocity * 1000
    current = np.zeros(len(length))
    for neuron in range(len(length)):
        if velocity[neuron] >= 0:
            # Calculate the current to be sent to the neuron
            # current[neuron] = BASERATE + 0.05*velocity[neuron] + 0.005 * length[neuron]
            # Prochazka's Equation. Desired spiking frequency
            # Ia_Spike_Freq = (4.3 * velocity[neuron]**0.6) + (2 * length[neuron]) + BASERATE
            Ia_Spike_Freq = (4.3 * velocity[neuron]**0.6) + BASERATE
            # Calculate the current that alligns to the desired spike frequency
            current[neuron] = freq2cur(Ia_Spike_Freq)

        # Ensure it remains above the baserate
        if current[neuron] < BASERATE_CURRENT:
            current[neuron] = BASERATE_CURRENT
    return current


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
            # Prochazka's Equation. Desired spiking frequency
            Ia_Spike_Freq = (velocity_sign * 4.3 * abs(velocity[neuron])**0.6) + (2 * length[neuron]) + BASERATE
            # print('Ia Spike Frequency: ' + str(Ia_Spike_Freq))

            # Calculate the current that alligns to the desired spike frequency
            current[neuron] = freq2cur(Ia_Spike_Freq)
            if current[neuron] < (BASERATE_CURRENT * (1 - 0.025)):
                current[neuron] = BASERATE_CURRENT_NOISY[neuron]
        else:
            current[neuron] = BASERATE_CURRENT_NOISY[neuron]
        # Ensure it remains above the baserate
        # if current[neuron] <= BASERATE_CURRENT:
        #     # current[neuron] = BASERATE_CURRENT
        #     current[neuron] = BASERATE_CURRENT_NOISY[neuron]
    return current

''' Implements the STDP window. Send your time difference, get some LTP / LTD '''
def weightUpdate(dt_array):
    A_pos = 0.1 # Max potentiation level
    A_neg = 0.25 # Max Depression value
    tou_pos = 10.0 # ms. Decay time constant for potentiation
    tou_neg = 11.0 # ms. Decay time constant for depression
    weight_change = []

    for dt in dt_array:
        if dt == None:
            weight_change.append(0.0)
        else:
            # If dt is positive, that means that the postsynaptic neuron spiked before the presynaptic. DEPRESSION
            if dt > 0:
                weight_change.append(-A_neg * np.exp(-dt / tou_neg))
            # If dt is negative, that means the presynaptic neuron spiked before the postsynaptic. POTENTIATION
            if dt <= 0:
                weight_change.append(A_pos * np.exp(dt / tou_pos))
    return weight_change

''' Network Generation Code '''
def MNNetworkGenerator(neuron_type, motor_neurons):
    # Create base network
    net = Network()

    # Create name array for motor neurons
    mn_names = []
    for i in range(1, motor_neurons + 1):
        mn_names.append('Motor Neuron ' + str(i))

    # Colors for the network
    mn_color = 'seagreen'
    input_color = 'palevioletred'
    output_color = 'cornflowerblue'

    # Add neurons to the network
    for mn in mn_names:
        net.add_neuron(neuron_type, name=mn, color=mn_color)
        net.add_input(dest=mn, name=('IN ' + mn), color=input_color)
        net.add_output(source=mn, spiking=True, name=(mn + ' Spike'), color=output_color)

    return net

''' Network Creation Code '''
def STDPNetworkGenerator(presynaptic_neurons=None, postsynaptic_neurons=None, presynaptic_type=None, postsynaptic_type=None, synapse_type=None):
    # Check to ensure values passed in are actually filled
    if presynaptic_neurons == None:
        print('No presynaptic neurons defined!')
        return None
    if postsynaptic_neurons == None:
        print('No postsynaptic neurons defined!')
        return None
    # If values not given, generate here
    if presynaptic_type == None:
        presynaptic_type = SpikingNeuron()
    if postsynaptic_type == None:
        postsynaptic_type = SpikingNeuron()
    if synapse_type == None:
        synapse_type = SpikingSynapse()

    # Create base network
    net = Network()

    # Create name array for presynaptic neurons
    pre_names = []
    for i in range(1, presynaptic_neurons + 1):
        pre_names.append('PRE' + str(i))

    # Create name array for postsynaptc neurons
    post_names = []
    for i in range(1, postsynaptic_neurons + 1):
        post_names.append('POST' + str(i))

    # Colors for the network
    pre_color = 'seagreen'
    post_color = 'bisque'
    input_color = 'palevioletred'
    output_color = 'cornflowerblue'

    # Add presynaptic and postsynaptic neurons to the network
    for presynaptic in pre_names:
        net.add_neuron(presynaptic_type, name=presynaptic, color=pre_color)
        net.add_input(dest=presynaptic, name=('IN ' + presynaptic), color=input_color)
    for postsynaptic in post_names:
        net.add_neuron(postsynaptic_type, name=postsynaptic, color=post_color)
        net.add_output(source=postsynaptic, name=(postsynaptic + ' Spike'), spiking=True, color=output_color)

    # Connect all neurons
    for presynaptic in pre_names:
        for postsynaptic in post_names:
            net.add_connection(connection_type=synapse_type, source=presynaptic, destination=postsynaptic)

    return net

def save_video(frames, fps, length):
    # Cut down to desired length
    video_max_frames = (length * fps) - 1

    # Output file name
    output_name = 'results/LegSimulationVideo' + str(length) + '.mp4'
    # Write frames to video
    media.write_video(output_name, images=frames[0:video_max_frames], fps=fps)

from sns_toolbox.neurons import SpikingNeuron
from sns_toolbox.connections import SpikingSynapse
from .NetworkGeneration import STDPNetworkGenerator
from .NetworkGeneration_MN import MNNetworkGenerator
from random import Random
import numpy as np

# Define STDP Neurons
pre_neuron = SpikingNeuron(
    threshold_proportionality_constant=0.0,
    threshold_initial_value=1.0
)
post_neuron = SpikingNeuron(
    threshold_proportionality_constant=0.0,
    threshold_initial_value=1.0
)
# Define STDP Synapse
synapse = SpikingSynapse(
    reversal_potential=2.0,
    max_conductance=3.0
)

# Define MN Neuron
motor_neuron = SpikingNeuron(
    threshold_proportionality_constant=0.0,
    threshold_initial_value=1.0
)

''' Functions for Networks '''
# Create matrix of random conductance values. Only populates the section consisting of pre and postsynaptic neurons
def randConnections(pre_num, post_num, seed=None, g_min=1.50, g_max=4.00):
    if seed is not None:
        rng = Random(x=seed)
    else:
        rng = Random()
        
    g_min = int(g_min * 100) # Convert to int
    g_max = int(g_max * 100) # Convert to int

    # Empty matrix for conductivity parameters
    matrix = np.zeros([pre_num + post_num, pre_num + post_num])

    for pre in range(0, pre_num):
        for post in range(pre_num, pre_num + post_num):
            matrix[post, pre] = (rng.randint(g_min, g_max) / 100)
    
    return matrix

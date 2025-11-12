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

''' Custom imports '''
import custom_methods

''' MuJoCo Imports '''
import mujoco as mj
import mediapy as media

''' Other Misc Imports '''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from random import Random, randint

''' Make the plots pretty and slooooooow '''
mpl.rcParams['figure.dpi'] = 600 # 600 is pretty high definition

''' SNS Parameters '''
sns_dt = 0.1 # ms NOTE: This aligns to the XML file. You cannot change this without changing the definition in the xml file!
sns_tmax = 20000 # ms
sns_t = np.arange(0, sns_tmax, sns_dt)

''' MuJoCo Parameters '''
mj_dt = sns_dt / 1000 # Converts to seconds. essentially 0.0001 s
mj_tmax = sns_tmax / 1000 # Converts to seconds. essentially 10.000 s
mj_t = np.arange(0, mj_tmax, mj_dt)

framerate = 60 # fps for movie capture

''' Load MuJoCo File '''
# Load model
mjmodel = mj.MjModel.from_xml_path('resources/MJ_Leg_Smaller.xml')
mjdata = mj.MjData(mjmodel)
# Create MuJoCo renderer
renderer = mj.Renderer(mjmodel)

# Camera options for the fun of it
mj_camera = ['fixed', 'Angle2', 'Angle3', 'Angle4', 'Angle5', 'Angle6']
active_camera = 5

# Show image of XML file for reference
mj.mj_forward(mjmodel, mjdata)
renderer.update_scene(mjdata, camera=mj_camera[0])
image = renderer.render()
# media.show_image(renderer.render())
media.write_image('results/combo_creation_noisy.png', image)
renderer.close() # Needed to prevent crashing?

# Define pre- and postsynaptic neurons
pre_neuron = SpikingNeuron(
    threshold_proportionality_constant=0.0,
    threshold_initial_value=1.0
)
post_neuron = SpikingNeuron(
    threshold_proportionality_constant=0.0,
    threshold_initial_value=1.0
)
# Define Synapse
synapse = SpikingSynapse(
    reversal_potential=2.0,
    max_conductance=3.0
)

# Parameters for the number of neurons
STDP_PRE_NUM = 4
STDP_POST_NUM = 10

# Send to function to generate the network
net = custom_methods.STDPNetworkGenerator(presynaptic_type=pre_neuron, postsynaptic_type=post_neuron, presynaptic_neurons=STDP_PRE_NUM, postsynaptic_neurons=STDP_POST_NUM, synapse_type=synapse)

# Network learning variables
# LTP
ltp_a = 0.010 #0.016
ltp_t = 0.5
# LTD
ltd_a = 0.012
ltd_t = 2.5
max_condutance = 4.0

# Display the network
render(net, view=False, filename='results/network', img_format='png')

# Define spiking motor neuron
motor_neuron = SpikingNeuron(
    threshold_proportionality_constant=0.0,
    threshold_initial_value=1.0
)

# Parameter for number of motor neurons (should like up with number of presynaptic neurons)
MOTOR_NEURON_NUM = STDP_PRE_NUM

# Send to function to generate the network as desired
net_motor = custom_methods.MNNetworkGenerator(neuron_type=motor_neuron, motor_neurons=MOTOR_NEURON_NUM)

# Display the network
render(net_motor, view=False, filename='results/network_motor', img_format='png')

# Compile learning network
sns_stdp_network = net.compile(dt=sns_dt, backend='numpy', debug=False,
                STDP_PRE=STDP_PRE_NUM, STDP_POST=STDP_POST_NUM,
                STDP_LTP_A=ltp_a, STDP_LTP_T=ltp_t, STDP_LTD_A=ltd_a, STDP_LTD_T=ltd_t,
                MAX_CONDUCTIVITY=max_condutance)


# Compile motor neuron network
sns_mn_network = net_motor.compile(dt=sns_dt, backend='numpy_standard', debug=False)


RANDOMIZED_CONDUCTIVITY = custom_methods.randConnections(STDP_PRE_NUM, STDP_POST_NUM, g_max=max_condutance)

# Print statements if you want to check
print(sns_stdp_network.__dict__.get('g_max_spike'))
# print(sns_stdp_network.g_increment)
print(sns_stdp_network.g_max_spike[STDP_PRE_NUM:, 0:STDP_PRE_NUM])
initial_conductance = np.copy(RANDOMIZED_CONDUCTIVITY[STDP_PRE_NUM:, 0:STDP_PRE_NUM])
# print(sns_stdp_network.__dict__.get('pre_spike_diff'))
# print(sns_stdp_network.__dict__.get('post_spike_diff'))

# Number of muscles in MuJoCo model
MJ_MUSCLE_NUM = MOTOR_NEURON_NUM

''' Motor Neuron '''
# Activation sent to SNS
mn_activation_current = np.zeros(shape=(len(sns_t), net_motor.get_num_inputs_actual()))

# Motor neuron output from SNS
mn_data = np.zeros(shape=(len(sns_t), net_motor.get_num_outputs_actual()))

''' Sensor feedback from MuJoCo '''
mj_length_data = np.zeros(shape=[len(mj_t), MJ_MUSCLE_NUM])
mj_velocity_data = np.zeros(shape=[len(mj_t), MJ_MUSCLE_NUM])

# Resting length recording
mj_length_resting = mjdata.sensordata[0:MJ_MUSCLE_NUM].copy()

''' STDP Network Parameters '''
# Current to Inject into Ia Feedback Neurons == INTO SNS
stdp_activation_current = np.zeros(shape=(len(sns_t), net.get_num_inputs_actual()))

# Learning network output data collection == OUTPUT of SNS
stdp_data = np.zeros([len(sns_t), net.get_num_outputs_actual()])

''' Assigning injected current to motor neurons '''
mn_activation = custom_methods.randActivation3(sns_t, 10000, 2000, num_motors=MJ_MUSCLE_NUM)
# mn_activation = manualActivation(sns_t, num_motors=MJ_MUSCLE_NUM, active=0b0011)

activation_level = 1.02 # From Paper: Hoffer Cat Hindlimb Motoneurons

mn_activation_current[:, 0] = mn_activation[:, 0] * activation_level
mn_activation_current[:, 1] = mn_activation[:, 1] * activation_level
mn_activation_current[:, 2] = mn_activation[:, 2] * activation_level
mn_activation_current[:, 3] = mn_activation[:, 3] * activation_level

mn_activation_current = custom_methods.NoisyAmps(mn_activation_current, 5)

MAX_MUSCLE_POWER = 25

# Tracking changes in conductance over the simulation
g_track = np.zeros(shape=[len(sns_t), STDP_POST_NUM, STDP_PRE_NUM])

# Percent noise for Ia afferent neurons
IA_BASELINE_NOISE = 0.01


# Set up variable to capture frames
frames = []
# Reset Simulation
mj.mj_resetData(mjmodel, mjdata)
# Restart Renderer
renderer = mj.Renderer(mjmodel)


for i in range(len(sns_t)):

    ''' Motor Neuron SNS '''
    mn_data[i,:] = sns_mn_network(mn_activation_current[i,:])

    ''' Spiking Motor Neuron === Muscle Activation '''
    if sum(sns_mn_network.__dict__.get('spikes')) != 0:
        mn_spike = sns_mn_network.__dict__.get('spikes')

        # If a spike occured for a specific motor, activate it for a timestep
        for muscle in range(MJ_MUSCLE_NUM):
            if mn_spike[muscle] != 0:
                mjdata.act[muscle] = MAX_MUSCLE_POWER
            else:
                mjdata.act[muscle] = 0.0
    else:
        mjdata.act[:] = 0.0

    ''' Advance MuJoCo Simulation '''
    mj.mj_step(mjmodel, mjdata)

    # Capture frame data if it corresponds to framerate demands
    if len(frames) < mjdata.time*framerate:
        if len(frames) % 120 == 0:
            active_camera += 1
            if active_camera >= len(mj_camera):
                active_camera = 0
        renderer.update_scene(mjdata, camera=mj_camera[0])
        pixels = renderer.render().copy()
        frames.append(pixels)


    ''' Record MuJoCo Sensor Outputs '''
    mj_length_data[i] = mjdata.sensordata[0:MJ_MUSCLE_NUM]
    mj_velocity_data[i] = mjdata.sensordata[MJ_MUSCLE_NUM:]

    # Subtract resting length to get displacement. Not sure if this is the right way?
    mj_length_data[i] = mj_length_data[i] - mj_length_resting

    # Convert Ia feedback (length & velocity) to current input into neuron
    stdp_activation_current[i] = custom_methods.vel2cur(length=mj_length_data[i], velocity=mj_velocity_data[i], current_time=sns_t[i])

    ''' STDP SNS '''
    # At the first call, update the conductance matrix. Afterwards, do not
    if i == 0:
        stdp_data[i, :] = sns_stdp_network(stdp_activation_current[i, :], current_time=sns_t[i], dt=sns_dt, g_update=RANDOMIZED_CONDUCTIVITY)
    else:
        stdp_data[i, :] = sns_stdp_network(stdp_activation_current[i, :], current_time=sns_t[i], dt=sns_dt)

    # Record conductance values to plot
    g_track[i] = sns_stdp_network.g_max_spike[STDP_PRE_NUM:, 0:STDP_PRE_NUM]

# Fix data orientation for better plotting
mn_data = mn_data.transpose()
mj_length_data = mj_length_data.transpose()
mj_velocity_data = mj_velocity_data.transpose()
stdp_activation_current = stdp_activation_current.transpose()
stdp_data = stdp_data.transpose()

# Optional save video
custom_methods.save_video(frames=frames, fps=framerate, length=20)

import numpy as np
import mujoco as mj
from .muscleActivation import MuscleAct
from .calculations import NoisyAmps
from .networks import motor_neuron, pre_neuron, post_neuron, synapse, randConnections
from .NetworkGeneration import STDPNetworkGenerator
from .NetworkGeneration_MN import MNNetworkGenerator

''' Constants '''

''' Time Parameters '''
# SNS
sns_dt = 0.1 # ms NOTE: This aligns to the XML file. You cannot change this without changing the definition in the xml file!
sns_tmax = 5000 # ms
sns_t = np.arange(0, sns_tmax, sns_dt)

# MuJoCo
mj_dt = sns_dt / 1000 # Converts to seconds. essentially 0.0001 s
mj_tmax = sns_tmax / 1000 # Converts to seconds. essentially 10.000 s
mj_t = np.arange(0, mj_tmax, mj_dt)

framerate = 60 # fps for movie capture

''' STDP Window Parameters '''
# LTP
ltp_a = 0.010
ltp_t = 0.5
# LTD
ltd_a = 0.012
ltd_t = 2.5

max_condutance = 4.0

''' STDP Network Parameters '''
# Parameters for the number of neurons
STDP_PRE_NUM = 4
STDP_POST_NUM = 2
MOTOR_NEURON_NUM = STDP_PRE_NUM

# Current to Inject into Ia Feedback Neurons == INTO SNS
stdp_activation_current = np.zeros(shape=(len(sns_t), STDP_PRE_NUM))

# Learning network output data collection == OUTPUT of SNS
stdp_data = np.zeros([len(sns_t), STDP_POST_NUM + 2])

# Build STDP network
net_stdp = STDPNetworkGenerator(presynaptic_type=pre_neuron, postsynaptic_type=post_neuron, 
                               presynaptic_neurons=STDP_PRE_NUM, postsynaptic_neurons=STDP_POST_NUM, 
                               synapse_type=synapse)

# Compile learning network
sns_stdp_network = net_stdp.compile(dt=sns_dt, backend='numpy', debug=False,
                STDP_PRE=STDP_PRE_NUM, STDP_POST=STDP_POST_NUM,
                STDP_LTP_A=ltp_a, STDP_LTP_T=ltp_t, STDP_LTD_A=ltd_a, STDP_LTD_T=ltd_t,
                MAX_CONDUCTIVITY=max_condutance)

print(sns_stdp_network.g_max_spike)
print(sns_stdp_network.del_e)

RANDOMIZED_CONDUCTIVITY = randConnections(STDP_PRE_NUM, STDP_POST_NUM, g_max=max_condutance, inhibitoryConnections=True)
RANDOMIZED_CONDUCTIVITY[4] = [2.3, 2.3, 2.4, 2.4, 0, 1]
RANDOMIZED_CONDUCTIVITY[5] = [2.3, 2.3, 2.4, 2.6, 1, 0]
print(RANDOMIZED_CONDUCTIVITY)
initial_conductance = np.copy(RANDOMIZED_CONDUCTIVITY[STDP_PRE_NUM:, 0:STDP_PRE_NUM])


''' Motor Neuron Network Parameters '''
MJ_MUSCLE_NUM = MOTOR_NEURON_NUM

# Activation sent to SNS
mn_activation_current = np.zeros(shape=(len(sns_t), MOTOR_NEURON_NUM))

# Motor neuron output from SNS 
mn_data = np.zeros(shape=(len(sns_t), MJ_MUSCLE_NUM))

# Build motor neuorn network
net_motor = MNNetworkGenerator(neuron_type=motor_neuron, motor_neurons=MOTOR_NEURON_NUM)

# Compile motor neuron network
sns_mn_network = net_motor.compile(dt=sns_dt, backend='numpy_standard', debug=False)

''' MuJoCo Parameters '''
# Sensor feedback
mj_length_data = np.zeros(shape=[len(mj_t), MJ_MUSCLE_NUM])
mj_velocity_data = np.zeros(shape=[len(mj_t), MJ_MUSCLE_NUM])

# Load model
mjmodel = mj.MjModel.from_xml_path('./projectCode/MJ_Leg_Smaller.xml')
mjdata = mj.MjData(mjmodel)
frames = []

# Resting length recording
mj.mj_forward(mjmodel, mjdata)
mj_length_resting = mjdata.sensordata[0:MJ_MUSCLE_NUM].copy()

''' Assigning injected current to motor neurons '''
mn_activation = MuscleAct().randActivation(sns_t, 10000, 2500, num_motors=MJ_MUSCLE_NUM)
# mn_activation = MuscleAct().manualActivation(sns_t, num_motors=MJ_MUSCLE_NUM, active=0b0011)

activation_level = 1.02 # From Paper: Hoffer Cat Hindlimb Motoneurons

mn_activation_current[:, 0] = mn_activation[:, 0] * activation_level
mn_activation_current[:, 1] = mn_activation[:, 1] * activation_level
mn_activation_current[:, 2] = mn_activation[:, 2] * activation_level
mn_activation_current[:, 3] = mn_activation[:, 3] * activation_level

# mn_activation_current[0:100000, 0] = 1.02
# mn_activation_current[0:100000, 1] = 1.02
# mn_activation_current[0:100000, 2] = 0
# mn_activation_current[0:100000, 3] = 0

mn_activation_current = NoisyAmps(mn_activation_current, 5)

MAX_MUSCLE_POWER = 25

# Tracking changes in conductance over the simulation
g_track = np.zeros(shape=[len(sns_t), STDP_POST_NUM, STDP_PRE_NUM])

# Percent noise for Ia afferent neurons
IA_BASELINE_NOISE = 0.01
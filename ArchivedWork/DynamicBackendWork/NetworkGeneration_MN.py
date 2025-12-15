''' SNS Imports '''
from sns_toolbox.neurons import SpikingNeuron
from sns_toolbox.networks import Network

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
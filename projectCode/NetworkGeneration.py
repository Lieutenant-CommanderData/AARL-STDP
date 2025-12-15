
''' SNS Imports '''
from sns_toolbox.neurons import SpikingNeuron
from sns_toolbox.connections import SpikingSynapse
from sns_toolbox.networks import Network
from sns_toolbox.renderer import render

''' Network Creation Code '''
def STDPNetworkGenerator(presynaptic_neurons=None, postsynaptic_neurons=None, presynaptic_type=None, postsynaptic_type=None, synapse_type=None, renderNet=True):
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

        if True:
            net.add_output(source=postsynaptic, name=(postsynaptic + ' Voltage'), spiking=False, color=output_color)

    # Connect all neurons
    for presynaptic in pre_names:
        for postsynaptic in post_names:
            net.add_connection(connection_type=synapse_type, source=presynaptic, destination=postsynaptic)

    ''' New addition
    Mutually inhibitory connections between TWO interneurons
    '''
    if True:
        inhibitorySynapse = SpikingSynapse(reversal_potential=-2.0, max_conductance=1.0)
        net.add_connection(connection_type=inhibitorySynapse, source=post_names[0], destination=post_names[1])
        net.add_connection(connection_type=inhibitorySynapse, source=post_names[1], destination=post_names[0])

    # Save the .png of the network
    if renderNet == True:
        render(net, view=False, save=True, filename='./results/STDPNetwork')

    return net
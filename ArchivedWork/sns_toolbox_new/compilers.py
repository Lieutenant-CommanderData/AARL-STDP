from sns_toolbox.backends import SNS_Numpy, SNS_Numpy_standard
from sns_toolbox.neurons import SpikingNeuron, NonSpikingNeuronWithGatedChannels

import numpy as np
import sys
import warnings

def __compile_numpy__(network, dt=0.01, debug=False, STDP_PRE=None, STDP_POST=None, STDP_LTP_A=None, STDP_LTP_T=None, STDP_LTD_A=None, STDP_LTD_T=None, MAX_CONDUCTIVITY=None) -> SNS_Numpy:
    if debug:
        print('-------------------------------------------------------------------------------------------------------')
        print('COMPILING NETWORK USING NUMPY:')
        print('-------------------------------------------------------------------------------------------------------')

    """
    --------------------------------------------------------------------------------------------------------------------
    Get net parameters
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('-----------------------------')
        print('Getting network parameters...')
        print('-----------------------------')
    spiking = network.params['spiking']
    delay = network.params['delay']
    electrical = network.params['electrical']
    electrical_rectified = network.params['electricalRectified']
    gated = network.params['gated']
    num_channels = network.params['numChannels']
    name = network.params['name']
    num_populations = network.get_num_populations()
    num_neurons = network.get_num_neurons()
    num_connections = network.get_num_connections()
    num_inputs = network.get_num_inputs()
    num_outputs = network.get_num_outputs()
    # R = network.params['R']
    if debug:
        print('Spiking:')
        print(spiking)
        print('Spiking Propagation Delay:')
        print(delay)
        print('Electrical Synapses:')
        print(electrical)
        print('Rectified Electrical Synapses:')
        print(electrical_rectified)
        print('Number of Populations:')
        print(num_populations)
        print('Number of Neurons:')
        print(num_neurons)
        print('Number of Connections')
        print(num_connections)
        print('Number of Inputs:')
        print(num_inputs)
        print('Number of Outputs:')
        print(num_outputs)
        # print('Network Voltage Range (mV):')
        # print(R)

    """
    --------------------------------------------------------------------------------------------------------------------
    Initialize vectors and matrices
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------------------------')
        print('Initializing vectors and matrices')
        print('---------------------------------')
    V = np.zeros(num_neurons)
    V_last = np.zeros(num_neurons)
    V_0 = np.zeros(num_neurons)
    V_rest = np.zeros(num_neurons)
    c_m = np.zeros(num_neurons)
    g_m = np.zeros(num_neurons)
    i_b = np.zeros(num_neurons)
    if spiking:
        spikes = np.zeros(num_neurons)

        # Mark added these
        spike_time = np.zeros(num_neurons)
        POST1_spike_diff = []
        POST2_spike_diff = []
        POST1_counter = 100
        POST2_counter = 100
        IN1_spike_diff = []
        IN2_spike_diff = []
        IN3_spike_diff = []
        IN4_spike_diff = []

        # V2 Additions
        # Defining of initial values
        # Matrix of spike time differences. A column shows a Pre's relative spike time to the Post's
        PRE_spike_diff = np.ones(shape=[STDP_PRE, STDP_POST]) * 100 # Buffer added so more recent spike time is not 0.0ms
        POST_spike_diff = np.ones(shape=[STDP_POST, STDP_PRE]) * 100 # Buffer added
        POST_counter = np.ones(STDP_POST) * 51 # Add buffer so that neurons can spike when the code begins
        STDP_LTD_A = STDP_LTD_A # STDP Parameter
        STDP_LTP_A = STDP_LTP_A # STDP Parameter
        STDP_LTP_T = STDP_LTP_T # STDP Parameter
        STDP_LTD_T = STDP_LTD_T # STDP Parameter
        MAX_CONDUCTIVITY = MAX_CONDUCTIVITY # STDP Parameter

        theta_0 = np.zeros(num_neurons)
        theta = np.zeros(num_neurons)
        theta_last = np.zeros(num_neurons)
        m = np.zeros(num_neurons)
        tau_theta = np.zeros(num_neurons)
        theta_leak = np.zeros(num_neurons)
        theta_increment = np.zeros(num_neurons)
        theta_floor = np.zeros(num_neurons)
        V_reset = np.zeros(num_neurons)

    g_max_non = np.zeros([num_neurons, num_neurons])
    del_e = np.zeros([num_neurons, num_neurons])
    e_lo = np.zeros([num_neurons, num_neurons])
    e_hi = np.ones([num_neurons, num_neurons])
    if spiking:
        g_max_spike = np.zeros([num_neurons, num_neurons])
        g_spike = np.zeros([num_neurons, num_neurons])
        tau_syn = np.zeros([num_neurons, num_neurons]) + 1
        g_increment = np.zeros([num_neurons, num_neurons])
        if delay:
            spike_delays = np.zeros([num_neurons, num_neurons])
            spike_rows = []
            spike_cols = []
            buffer_steps = []
            buffer_nrns = []
            delayed_spikes = np.zeros([num_neurons, num_neurons])
    if electrical:
        g_electrical = np.zeros([num_neurons, num_neurons])
    if electrical_rectified:
        g_rectified = np.zeros([num_neurons, num_neurons])
    if gated:
        # Channel params
        g_ion = np.zeros([num_channels, num_neurons])
        e_ion = np.zeros([num_channels, num_neurons])
        # A gate params
        pow_a = np.zeros([num_channels, num_neurons])
        slope_a = np.zeros([num_channels, num_neurons])
        k_a = np.zeros([num_channels, num_neurons]) + 1
        e_a = np.zeros([num_channels, num_neurons])
        # B gate params
        pow_b = np.zeros([num_channels, num_neurons])
        slope_b = np.zeros([num_channels, num_neurons])
        k_b = np.zeros([num_channels, num_neurons]) + 1
        e_b = np.zeros([num_channels, num_neurons])
        tau_max_b = np.zeros([num_channels, num_neurons]) + 1
        # C gate params
        pow_c = np.zeros([num_channels, num_neurons])
        slope_c = np.zeros([num_channels, num_neurons])
        k_c = np.zeros([num_channels, num_neurons]) + 1
        e_c = np.zeros([num_channels, num_neurons])
        tau_max_c = np.zeros([num_channels, num_neurons]) + 1

        b_gate = np.zeros([num_channels, num_neurons])
        b_gate_last = np.zeros([num_channels, num_neurons])
        b_gate_0 = np.zeros([num_channels, num_neurons])
        c_gate = np.zeros([num_channels, num_neurons])
        c_gate_last = np.zeros([num_channels, num_neurons])
        c_gate_0 = np.zeros([num_channels, num_neurons])

    pops_and_nrns = []
    index = 0
    for pop in range(len(network.populations)):
        num_neurons_in_pop = network.populations[pop]['number']  # find the number of neurons in the population
        pops_and_nrns.append([])
        for num in range(num_neurons_in_pop):
            pops_and_nrns[pop].append(index)
            index += 1

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Neurons
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------')
        print('Setting neurons')
        print('---------------')
    index = 0
    for pop in range(len(network.populations)):
        num_neurons_in_pop = network.populations[pop]['number']  # find the number of neurons in the population
        initial_value = network.populations[pop]['initial_value']
        for num in range(num_neurons_in_pop):  # for each neuron, copy the parameters over
            c_m[index] = network.populations[pop]['type'].params['membrane_capacitance']
            g_m[index] = network.populations[pop]['type'].params['membrane_conductance']
            i_b[index] = network.populations[pop]['type'].params['bias']
            V_rest[index] = network.populations[pop]['type'].params['resting_potential']
            if hasattr(initial_value, '__iter__'):
                V_last[index] = initial_value[num]
            elif initial_value is None:
                V_last[index] = V_rest[index]
            else:
                V_last[index] = initial_value
            if spiking:
                if isinstance(network.populations[pop]['type'],
                              SpikingNeuron):  # if the neuron is spiking, copy more
                    theta_0[index] = network.populations[pop]['type'].params['threshold_initial_value']
                    m[index] = network.populations[pop]['type'].params['threshold_proportionality_constant']
                    tau_theta[index] = network.populations[pop]['type'].params['threshold_time_constant']
                    theta_leak[index] = network.populations[pop]['type'].params['threshold_leak_rate']
                    theta_increment[index] = network.populations[pop]['type'].params['threshold_increment']
                    theta_floor[index] = network.populations[pop]['type'].params['threshold_floor']
                    V_reset[index] = network.populations[pop]['type'].params['reset_potential']

                else:  # otherwise, set to the special values for NonSpiking
                    theta_0[index] = sys.float_info.max
                    m[index] = 0
                    tau_theta[index] = 1
                    theta_leak[index] = 0
                    theta_increment[index] = 0
                    theta_floor[index] = -sys.float_info.max
                    V_reset[index] = V_rest[index]
            if gated:
                if isinstance(network.populations[pop]['type'], NonSpikingNeuronWithGatedChannels):
                    # Channel params
                    g_ion[:, index] = network.populations[pop]['type'].params['Gion']
                    e_ion[:, index] = network.populations[pop]['type'].params['Eion']
                    # A gate params
                    pow_a[:, index] = network.populations[pop]['type'].params['paramsA']['pow']
                    slope_a[:, index] = network.populations[pop]['type'].params['paramsA']['slope']
                    k_a[:, index] = network.populations[pop]['type'].params['paramsA']['k']
                    e_a[:, index] = network.populations[pop]['type'].params['paramsA']['reversal']
                    # B gate params
                    pow_b[:, index] = network.populations[pop]['type'].params['paramsB']['pow']
                    slope_b[:, index] = network.populations[pop]['type'].params['paramsB']['slope']
                    k_b[:, index] = network.populations[pop]['type'].params['paramsB']['k']
                    e_b[:, index] = network.populations[pop]['type'].params['paramsB']['reversal']
                    tau_max_b[:, index] = network.populations[pop]['type'].params['paramsB']['TauMax']
                    # C gate params
                    pow_c[:, index] = network.populations[pop]['type'].params['paramsC']['pow']
                    slope_c[:, index] = network.populations[pop]['type'].params['paramsC']['slope']
                    k_c[:, index] = network.populations[pop]['type'].params['paramsC']['k']
                    e_c[:, index] = network.populations[pop]['type'].params['paramsC']['reversal']
                    tau_max_c[:, index] = network.populations[pop]['type'].params['paramsC']['TauMax']

                    b_gate_last[:, index] = 1 / (1 + k_b[:, index] * np.exp(
                        slope_b[:, index] * (V_last[index] - e_b[:, index])))
                    c_gate_last[:, index] = 1 / (1 + k_c[:, index] * np.exp(
                        slope_c[:, index] * (V_last[index] - e_c[:, index])))
            index += 1
    V = np.copy(V_last)
    V_0 = np.copy(V_last)
    if spiking:
        theta = np.copy(theta_0)
        theta_last = np.copy(theta_0)
    if gated:
        b_gate = np.copy(b_gate_last)
        b_gate_0 = np.copy(b_gate_last)
        c_gate = np.copy(c_gate_last)
        c_gate_0 = np.copy(c_gate_last)

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Inputs
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('--------------')
        print('Setting Inputs')
        print('--------------')
    input_connectivity = np.zeros(
        [num_neurons, network.get_num_inputs_actual()])  # initialize connectivity matrix
    index = 0
    for inp in range(network.get_num_inputs()):  # iterate over the connections in the network
        size = network.inputs[inp]['size']
        dest_pop = network.inputs[inp]['destination']  # get the destination
        if size == 1:
            for dest in pops_and_nrns[dest_pop]:
                input_connectivity[dest][index] = 1.0  # set the weight in the correct source and destination
            index += 1
        else:
            for dest in pops_and_nrns[dest_pop]:
                input_connectivity[dest][index] = 1.0
                index += 1

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Connections
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('-------------------')
        print('Setting connections')
        print('-------------------')
    for syn in range(len(network.connections)):
        source_pop = network.connections[syn]['source']
        dest_pop = network.connections[syn]['destination']
        g_max = network.connections[syn]['params']['max_conductance']
        del_e_val = None
        e_lo_val = None
        e_hi_val = None
        if network.connections[syn]['params']['electrical'] is False:  # electrical connection
            del_e_val = network.connections[syn]['params']['reversal_potential']

        if network.connections[syn]['params']['matrix']:  # pattern and matrix connections
            pop_size_source = len(pops_and_nrns[source_pop])
            pop_size_dest = len(pops_and_nrns[dest_pop])
            source_index = pops_and_nrns[source_pop][0]
            dest_index = pops_and_nrns[dest_pop][0]
            if network.connections[syn]['params']['spiking']:
                tau_s = network.connections[syn]['params']['synapticTimeConstant']
                g_inc = network.connections[syn]['params']['conductance_increment']
                g_max_spike[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = g_max
                del_e[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = del_e_val
                tau_syn[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = tau_s
                g_increment[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = g_inc
                if delay:
                    delay_val = network.connections[syn]['params']['synapticTransmissionDelay']
                    spike_delays[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = delay_val

                    for source in pops_and_nrns[source_pop]:
                        for dest in pops_and_nrns[dest_pop]:
                            buffer_nrns.append(source)
                            buffer_steps.append(delay)
                            spike_rows.append(dest)
                            spike_cols.append(source)
            else:
                e_lo_val = network.connections[syn]['params']['e_lo']
                e_hi_val = network.connections[syn]['params']['e_hi']
                g_max_non[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = g_max
                del_e[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = del_e_val
                e_lo[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = e_lo_val
                e_hi[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = e_hi_val
        elif network.connections[syn]['params']['electrical']:  # electrical connection
            for source in pops_and_nrns[source_pop]:
                for dest in pops_and_nrns[dest_pop]:
                    if network.connections[syn]['params']['rectified']:  # rectified
                        g_rectified[dest][source] = g_max / len(pops_and_nrns[source_pop])
                    else:
                        g_electrical[dest][source] = g_max / len(pops_and_nrns[source_pop])
                        g_electrical[source][dest] = g_max / len(pops_and_nrns[source_pop])
        else:  # chemical connection
            if network.connections[syn]['params']['spiking']:  # spiking chemical synapse
                tau_s = network.connections[syn]['params']['synapticTimeConstant']
                g_inc = network.connections[syn]['params']['conductance_increment']
                if delay:
                    delay_val = network.connections[syn]['params']['synapticTransmissionDelay']
                for source in pops_and_nrns[source_pop]:
                    for dest in pops_and_nrns[dest_pop]:
                        g_max_spike[dest][source] = g_max / len(pops_and_nrns[source_pop])
                        g_increment[dest][source] = g_inc
                        del_e[dest][source] = del_e_val
                        tau_syn[dest][source] = tau_s
                        if delay:
                            spike_delays[dest][source] = delay_val
                            buffer_nrns.append(source)
                            buffer_steps.append(delay_val)
                            spike_rows.append(dest)
                            spike_cols.append(source)
            else:  # nonspiking chemical synapse
                e_lo_val = network.connections[syn]['params']['e_lo']
                e_hi_val = network.connections[syn]['params']['e_hi']
                for source in pops_and_nrns[source_pop]:
                    for dest in pops_and_nrns[dest_pop]:
                        g_max_non[dest][source] = g_max / len(pops_and_nrns[source_pop])
                        del_e[dest][source] = del_e_val
                        e_lo[dest][source] = e_lo_val
                        e_hi[dest][source] = e_hi_val

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate Time Factors
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('------------------------')
        print('Calculating Time Factors')
        print('------------------------')
    time_factor_membrane = dt / (c_m / g_m)
    if spiking:
        time_factor_threshold = dt / tau_theta
        time_factor_synapse = dt / tau_syn

    """
    --------------------------------------------------------------------------------------------------------------------
    Initialize Propagation Delay
    --------------------------------------------------------------------------------------------------------------------
    """
    if delay:
        if debug:
            print('------------------------------')
            print('Initializing Propagation Delay')
            print('------------------------------')
        buffer_length = int(np.max(spike_delays) + 1)
        spike_buffer = np.zeros([buffer_length, num_neurons])

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Outputs
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------')
        print('Setting Outputs')
        print('---------------')
    output_nodes = []
    index = 0
    for out in range(len(network.outputs)):
        source_pop = network.outputs[out]['source']
        num_source_neurons = network.populations[source_pop]['number']
        output_nodes.append([])
        for num in range(num_source_neurons):
            output_nodes[out].append(index)
            index += 1
    num_outputs = index

    output_voltage_connectivity = np.zeros(
        [num_outputs, num_neurons])  # initialize connectivity matrix
    if spiking:
        output_spike_connectivity = np.copy(output_voltage_connectivity)
    outputs = np.zeros(num_outputs)
    for out in range(len(network.outputs)):  # iterate over the connections in the network
        source_pop = network.outputs[out]['source']  # get the source
        for i in range(len(pops_and_nrns[source_pop])):
            if network.outputs[out]['spiking']:
                output_spike_connectivity[output_nodes[out][i]][
                    pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination
            else:
                output_voltage_connectivity[output_nodes[out][i]][
                    pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination

    """
    --------------------------------------------------------------------------------------------------------------------
    Arrange states and parameters into dictionary
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------------')
        print('Writing to Dictionary')
        print('---------------------')
    params = {'dt': dt,
              'name': name,
              'spiking': spiking,
              'delay': delay,
              'elec': electrical,
              'rect': electrical_rectified,
              'gated': gated,
              'numChannels': num_channels,
              'v': V,
              'vLast': V_last,
              'vRest': V_rest,
              'v0': V_0,
              'cM': c_m,
              'gM': g_m,
              'iB': i_b,
              'gMaxNon': g_max_non,
              'delE': del_e,
              'eLo': e_lo,
              'eHi': e_hi,
              'timeFactorMembrane': time_factor_membrane,
              'inputConn': input_connectivity,
              'numPop': num_populations,
              'numNeurons': num_neurons,
              'numConn': num_connections,
              'numInputs': num_inputs,
              'numOutputs': num_outputs,
              # 'r': R,
              'outConnVolt': output_voltage_connectivity}
    if spiking:
        params['spikes'] = spikes

        # Mark added these
        params['spikeTime'] = spike_time
        params['post1_spike_diff'] = POST1_spike_diff
        params['post2_spike_diff'] = POST2_spike_diff
        params['post1_counter'] = POST1_counter
        params['post2_counter'] = POST2_counter
        params['in1_spike_diff'] = IN1_spike_diff
        params['in2_spike_diff'] = IN2_spike_diff
        params['in3_spike_diff'] = IN3_spike_diff
        params['in4_spike_diff'] = IN4_spike_diff

        # Dictionary Definition
        # V2 Additions
        params['pre_spike_diff'] = PRE_spike_diff
        params['post_spike_diff'] = POST_spike_diff
        params['post_counter'] = POST_counter
        params['num_pre'] = STDP_PRE
        params['num_post'] = STDP_POST
        params['ltp_a'] = STDP_LTP_A
        params['ltp_t'] = STDP_LTP_T
        params['ltd_a'] = STDP_LTD_A
        params['ltd_t'] = STDP_LTD_T
        params['max_conductance'] = MAX_CONDUCTIVITY

        params['theta0'] = theta_0
        params['theta'] = theta
        params['thetaLast'] = theta_last
        params['m'] = m
        params['tauTheta'] = tau_theta
        params['gMaxSpike'] = g_max_spike
        params['gSpike'] = g_spike
        params['tauSyn'] = tau_syn
        params['timeFactorThreshold'] = time_factor_threshold
        params['timeFactorSynapse'] = time_factor_synapse
        params['outConnSpike'] = output_spike_connectivity
        params['thetaLeak'] = theta_leak
        params['thetaIncrement'] = theta_increment
        params['thetaFloor'] = theta_floor
        params['vReset'] = V_reset
        params['gIncrement'] = g_increment
    if delay:
        params['spikeDelays'] = spike_delays
        params['spikeRows'] = spike_rows
        params['spikeCols'] = spike_cols
        params['bufferSteps'] = buffer_steps
        params['bufferNrns'] = buffer_nrns
        params['delayedSpikes'] = delayed_spikes
        params['spikeBuffer'] = spike_buffer
    if electrical:
        params['gElectrical'] = g_electrical
    if electrical_rectified:
        params['gRectified'] = g_rectified
    if gated:
        params['gIon'] = g_ion
        params['eIon'] = e_ion
        params['powA'] = pow_a
        params['slopeA'] = slope_a
        params['kA'] = k_a
        params['eA'] = e_a
        params['powB'] = pow_b
        params['slopeB'] = slope_b
        params['kB'] = k_b
        params['eB'] = e_b
        params['tauMaxB'] = tau_max_b
        params['powC'] = pow_c
        params['slopeC'] = slope_c
        params['kC'] = k_c
        params['eC'] = e_c
        params['tauMaxC'] = tau_max_c
        params['bGate'] = b_gate
        params['bGateLast'] = b_gate_last
        params['bGate0'] = b_gate_0
        params['cGate'] = c_gate
        params['cGateLast'] = c_gate_last
        params['cGate0'] = c_gate_0

    """
    --------------------------------------------------------------------------------------------------------------------
    Passing params to backend object
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('-------------------------------------------------')
        print('Passing states and parameters to SNS_Numpy object')
        print('-------------------------------------------------')
    model = SNS_Numpy(params)

    """
    --------------------------------------------------------------------------------------------------------------------
    Final print
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('----------------------------')
        print('Final states and parameters:')
        print('----------------------------')
        print('Input Connectivity:')
        print(input_connectivity)
        print('g_max_non:')
        print(g_max_non)
        if spiking:
            print('GmaxSpike:')
            print(g_max_spike)
            print('Theta Increment:')
            print(g_increment)
        print('del_e:')
        print(del_e)
        print('e_lo:')
        print(e_lo)
        print('e_hi:')
        print(e_hi)
        if electrical:
            print('Gelectrical:')
            print(g_electrical)
        if electrical_rectified:
            print('GelectricalRectified:')
            print(g_rectified)
        print('Output Voltage Connectivity')
        print(output_voltage_connectivity)
        if spiking:
            print('Output Spike Connectivity:')
            print(output_spike_connectivity)
        print('v:')
        print(V)
        print('v_last:')
        print(V_last)
        print('v_rest:')
        print(V_rest)
        if spiking:
            print('theta_0:')
            print(theta_0)
            print('ThetaLast:')
            print(theta_last)
            print('Theta:')
            print(theta)
            print('ThetaLeak:')
            print(theta_leak)
            print('ThetaIncrement:')
            print(theta_increment)
            print('ThetaFloor:')
            print(theta_floor)
            print('v_reset:')
            print(V_reset)
        if gated:
            print('Number of Channels:')
            print(num_channels)
            print('Ionic Conductance:')
            print(g_ion)
            print('Ionic Reversal Potentials:')
            print(e_ion)
            print('A Gate Parameters:')
            print('Power:')
            print(pow_a)
            print('Slope:')
            print(slope_a)
            print('K:')
            print(k_a)
            print('Reversal Potential:')
            print(e_a)
            print('B Gate Parameters:')
            print('Power:')
            print(pow_b)
            print('Slope:')
            print(slope_b)
            print('K:')
            print(k_b)
            print('Reversal Potential:')
            print(e_b)
            print('Tau Max:')
            print(tau_max_b)
            print('B:')
            print(b_gate)
            print('B_last:')
            print(b_gate_last)
            print('C Gate Parameters:')
            print('Power:')
            print(pow_c)
            print('Slope:')
            print(slope_c)
            print('K:')
            print(k_c)
            print('Reversal Potential:')
            print(e_c)
            print('Tau Max:')
            print(tau_max_c)
            print('B:')
            print(c_gate)
            print('B_last:')
            print(c_gate_last)

    return model


def __compile_numpy_standard__(network, dt=0.01, debug=False) -> SNS_Numpy_standard:
    if debug:
        print('-------------------------------------------------------------------------------------------------------')
        print('COMPILING NETWORK USING NUMPY:')
        print('-------------------------------------------------------------------------------------------------------')

    """
    --------------------------------------------------------------------------------------------------------------------
    Get net parameters
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('-----------------------------')
        print('Getting network parameters...')
        print('-----------------------------')
    spiking = network.params['spiking']
    delay = network.params['delay']
    electrical = network.params['electrical']
    electrical_rectified = network.params['electricalRectified']
    gated = network.params['gated']
    num_channels = network.params['numChannels']
    name = network.params['name']
    num_populations = network.get_num_populations()
    num_neurons = network.get_num_neurons()
    num_connections = network.get_num_connections()
    num_inputs = network.get_num_inputs()
    num_outputs = network.get_num_outputs()
    # R = network.params['R']
    if debug:
        print('Spiking:')
        print(spiking)
        print('Spiking Propagation Delay:')
        print(delay)
        print('Electrical Synapses:')
        print(electrical)
        print('Rectified Electrical Synapses:')
        print(electrical_rectified)
        print('Number of Populations:')
        print(num_populations)
        print('Number of Neurons:')
        print(num_neurons)
        print('Number of Connections')
        print(num_connections)
        print('Number of Inputs:')
        print(num_inputs)
        print('Number of Outputs:')
        print(num_outputs)
        # print('Network Voltage Range (mV):')
        # print(R)

    """
    --------------------------------------------------------------------------------------------------------------------
    Initialize vectors and matrices
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------------------------')
        print('Initializing vectors and matrices')
        print('---------------------------------')
    V = np.zeros(num_neurons)
    V_last = np.zeros(num_neurons)
    V_0 = np.zeros(num_neurons)
    V_rest = np.zeros(num_neurons)
    c_m = np.zeros(num_neurons)
    g_m = np.zeros(num_neurons)
    i_b = np.zeros(num_neurons)
    if spiking:
        spikes = np.zeros(num_neurons)
        theta_0 = np.zeros(num_neurons)
        theta = np.zeros(num_neurons)
        theta_last = np.zeros(num_neurons)
        m = np.zeros(num_neurons)
        tau_theta = np.zeros(num_neurons)
        theta_leak = np.zeros(num_neurons)
        theta_increment = np.zeros(num_neurons)
        theta_floor = np.zeros(num_neurons)
        V_reset = np.zeros(num_neurons)

    g_max_non = np.zeros([num_neurons, num_neurons])
    del_e = np.zeros([num_neurons, num_neurons])
    e_lo = np.zeros([num_neurons, num_neurons])
    e_hi = np.ones([num_neurons, num_neurons])
    if spiking:
        g_max_spike = np.zeros([num_neurons, num_neurons])
        g_spike = np.zeros([num_neurons, num_neurons])
        tau_syn = np.zeros([num_neurons, num_neurons]) + 1
        g_increment = np.zeros([num_neurons, num_neurons])
        if delay:
            spike_delays = np.zeros([num_neurons, num_neurons])
            spike_rows = []
            spike_cols = []
            buffer_steps = []
            buffer_nrns = []
            delayed_spikes = np.zeros([num_neurons, num_neurons])
    if electrical:
        g_electrical = np.zeros([num_neurons, num_neurons])
    if electrical_rectified:
        g_rectified = np.zeros([num_neurons, num_neurons])
    if gated:
        # Channel params
        g_ion = np.zeros([num_channels, num_neurons])
        e_ion = np.zeros([num_channels, num_neurons])
        # A gate params
        pow_a = np.zeros([num_channels, num_neurons])
        slope_a = np.zeros([num_channels, num_neurons])
        k_a = np.zeros([num_channels, num_neurons]) + 1
        e_a = np.zeros([num_channels, num_neurons])
        # B gate params
        pow_b = np.zeros([num_channels, num_neurons])
        slope_b = np.zeros([num_channels, num_neurons])
        k_b = np.zeros([num_channels, num_neurons]) + 1
        e_b = np.zeros([num_channels, num_neurons])
        tau_max_b = np.zeros([num_channels, num_neurons]) + 1
        # C gate params
        pow_c = np.zeros([num_channels, num_neurons])
        slope_c = np.zeros([num_channels, num_neurons])
        k_c = np.zeros([num_channels, num_neurons]) + 1
        e_c = np.zeros([num_channels, num_neurons])
        tau_max_c = np.zeros([num_channels, num_neurons]) + 1

        b_gate = np.zeros([num_channels, num_neurons])
        b_gate_last = np.zeros([num_channels, num_neurons])
        b_gate_0 = np.zeros([num_channels, num_neurons])
        c_gate = np.zeros([num_channels, num_neurons])
        c_gate_last = np.zeros([num_channels, num_neurons])
        c_gate_0 = np.zeros([num_channels, num_neurons])

    pops_and_nrns = []
    index = 0
    for pop in range(len(network.populations)):
        num_neurons_in_pop = network.populations[pop]['number']  # find the number of neurons in the population
        pops_and_nrns.append([])
        for num in range(num_neurons_in_pop):
            pops_and_nrns[pop].append(index)
            index += 1

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Neurons
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------')
        print('Setting neurons')
        print('---------------')
    index = 0
    for pop in range(len(network.populations)):
        num_neurons_in_pop = network.populations[pop]['number']  # find the number of neurons in the population
        initial_value = network.populations[pop]['initial_value']
        for num in range(num_neurons_in_pop):  # for each neuron, copy the parameters over
            c_m[index] = network.populations[pop]['type'].params['membrane_capacitance']
            g_m[index] = network.populations[pop]['type'].params['membrane_conductance']
            i_b[index] = network.populations[pop]['type'].params['bias']
            V_rest[index] = network.populations[pop]['type'].params['resting_potential']
            if hasattr(initial_value, '__iter__'):
                V_last[index] = initial_value[num]
            elif initial_value is None:
                V_last[index] = V_rest[index]
            else:
                V_last[index] = initial_value
            if spiking:
                if isinstance(network.populations[pop]['type'],
                              SpikingNeuron):  # if the neuron is spiking, copy more
                    theta_0[index] = network.populations[pop]['type'].params['threshold_initial_value']
                    m[index] = network.populations[pop]['type'].params['threshold_proportionality_constant']
                    tau_theta[index] = network.populations[pop]['type'].params['threshold_time_constant']
                    theta_leak[index] = network.populations[pop]['type'].params['threshold_leak_rate']
                    theta_increment[index] = network.populations[pop]['type'].params['threshold_increment']
                    theta_floor[index] = network.populations[pop]['type'].params['threshold_floor']
                    V_reset[index] = network.populations[pop]['type'].params['reset_potential']

                else:  # otherwise, set to the special values for NonSpiking
                    theta_0[index] = sys.float_info.max
                    m[index] = 0
                    tau_theta[index] = 1
                    theta_leak[index] = 0
                    theta_increment[index] = 0
                    theta_floor[index] = -sys.float_info.max
                    V_reset[index] = V_rest[index]
            if gated:
                if isinstance(network.populations[pop]['type'], NonSpikingNeuronWithGatedChannels):
                    # Channel params
                    g_ion[:, index] = network.populations[pop]['type'].params['Gion']
                    e_ion[:, index] = network.populations[pop]['type'].params['Eion']
                    # A gate params
                    pow_a[:, index] = network.populations[pop]['type'].params['paramsA']['pow']
                    slope_a[:, index] = network.populations[pop]['type'].params['paramsA']['slope']
                    k_a[:, index] = network.populations[pop]['type'].params['paramsA']['k']
                    e_a[:, index] = network.populations[pop]['type'].params['paramsA']['reversal']
                    # B gate params
                    pow_b[:, index] = network.populations[pop]['type'].params['paramsB']['pow']
                    slope_b[:, index] = network.populations[pop]['type'].params['paramsB']['slope']
                    k_b[:, index] = network.populations[pop]['type'].params['paramsB']['k']
                    e_b[:, index] = network.populations[pop]['type'].params['paramsB']['reversal']
                    tau_max_b[:, index] = network.populations[pop]['type'].params['paramsB']['TauMax']
                    # C gate params
                    pow_c[:, index] = network.populations[pop]['type'].params['paramsC']['pow']
                    slope_c[:, index] = network.populations[pop]['type'].params['paramsC']['slope']
                    k_c[:, index] = network.populations[pop]['type'].params['paramsC']['k']
                    e_c[:, index] = network.populations[pop]['type'].params['paramsC']['reversal']
                    tau_max_c[:, index] = network.populations[pop]['type'].params['paramsC']['TauMax']

                    b_gate_last[:, index] = 1 / (1 + k_b[:, index] * np.exp(
                        slope_b[:, index] * (V_last[index] - e_b[:, index])))
                    c_gate_last[:, index] = 1 / (1 + k_c[:, index] * np.exp(
                        slope_c[:, index] * (V_last[index] - e_c[:, index])))
            index += 1
    V = np.copy(V_last)
    V_0 = np.copy(V_last)
    if spiking:
        theta = np.copy(theta_0)
        theta_last = np.copy(theta_0)
    if gated:
        b_gate = np.copy(b_gate_last)
        b_gate_0 = np.copy(b_gate_last)
        c_gate = np.copy(c_gate_last)
        c_gate_0 = np.copy(c_gate_last)

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Inputs
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('--------------')
        print('Setting Inputs')
        print('--------------')
    input_connectivity = np.zeros(
        [num_neurons, network.get_num_inputs_actual()])  # initialize connectivity matrix
    index = 0
    for inp in range(network.get_num_inputs()):  # iterate over the connections in the network
        size = network.inputs[inp]['size']
        dest_pop = network.inputs[inp]['destination']  # get the destination
        if size == 1:
            for dest in pops_and_nrns[dest_pop]:
                input_connectivity[dest][index] = 1.0  # set the weight in the correct source and destination
            index += 1
        else:
            for dest in pops_and_nrns[dest_pop]:
                input_connectivity[dest][index] = 1.0
                index += 1

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Connections
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('-------------------')
        print('Setting connections')
        print('-------------------')
    for syn in range(len(network.connections)):
        source_pop = network.connections[syn]['source']
        dest_pop = network.connections[syn]['destination']
        g_max = network.connections[syn]['params']['max_conductance']
        del_e_val = None
        e_lo_val = None
        e_hi_val = None
        if network.connections[syn]['params']['electrical'] is False:  # electrical connection
            del_e_val = network.connections[syn]['params']['reversal_potential']

        if network.connections[syn]['params']['matrix']:  # pattern and matrix connections
            pop_size_source = len(pops_and_nrns[source_pop])
            pop_size_dest = len(pops_and_nrns[dest_pop])
            source_index = pops_and_nrns[source_pop][0]
            dest_index = pops_and_nrns[dest_pop][0]
            if network.connections[syn]['params']['spiking']:
                tau_s = network.connections[syn]['params']['synapticTimeConstant']
                g_inc = network.connections[syn]['params']['conductance_increment']
                g_max_spike[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = g_max
                del_e[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = del_e_val
                tau_syn[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = tau_s
                g_increment[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = g_inc
                if delay:
                    delay_val = network.connections[syn]['params']['synapticTransmissionDelay']
                    spike_delays[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = delay_val

                    for source in pops_and_nrns[source_pop]:
                        for dest in pops_and_nrns[dest_pop]:
                            buffer_nrns.append(source)
                            buffer_steps.append(delay)
                            spike_rows.append(dest)
                            spike_cols.append(source)
            else:
                e_lo_val = network.connections[syn]['params']['e_lo']
                e_hi_val = network.connections[syn]['params']['e_hi']
                g_max_non[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = g_max
                del_e[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = del_e_val
                e_lo[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = e_lo_val
                e_hi[dest_index:dest_index + pop_size_dest, source_index:source_index + pop_size_source] = e_hi_val
        elif network.connections[syn]['params']['electrical']:  # electrical connection
            for source in pops_and_nrns[source_pop]:
                for dest in pops_and_nrns[dest_pop]:
                    if network.connections[syn]['params']['rectified']:  # rectified
                        g_rectified[dest][source] = g_max / len(pops_and_nrns[source_pop])
                    else:
                        g_electrical[dest][source] = g_max / len(pops_and_nrns[source_pop])
                        g_electrical[source][dest] = g_max / len(pops_and_nrns[source_pop])
        else:  # chemical connection
            if network.connections[syn]['params']['spiking']:  # spiking chemical synapse
                tau_s = network.connections[syn]['params']['synapticTimeConstant']
                g_inc = network.connections[syn]['params']['conductance_increment']
                if delay:
                    delay_val = network.connections[syn]['params']['synapticTransmissionDelay']
                for source in pops_and_nrns[source_pop]:
                    for dest in pops_and_nrns[dest_pop]:
                        g_max_spike[dest][source] = g_max / len(pops_and_nrns[source_pop])
                        g_increment[dest][source] = g_inc
                        del_e[dest][source] = del_e_val
                        tau_syn[dest][source] = tau_s
                        if delay:
                            spike_delays[dest][source] = delay_val
                            buffer_nrns.append(source)
                            buffer_steps.append(delay_val)
                            spike_rows.append(dest)
                            spike_cols.append(source)
            else:  # nonspiking chemical synapse
                e_lo_val = network.connections[syn]['params']['e_lo']
                e_hi_val = network.connections[syn]['params']['e_hi']
                for source in pops_and_nrns[source_pop]:
                    for dest in pops_and_nrns[dest_pop]:
                        g_max_non[dest][source] = g_max / len(pops_and_nrns[source_pop])
                        del_e[dest][source] = del_e_val
                        e_lo[dest][source] = e_lo_val
                        e_hi[dest][source] = e_hi_val

    """
    --------------------------------------------------------------------------------------------------------------------
    Calculate Time Factors
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('------------------------')
        print('Calculating Time Factors')
        print('------------------------')
    time_factor_membrane = dt / (c_m / g_m)
    if spiking:
        time_factor_threshold = dt / tau_theta
        time_factor_synapse = dt / tau_syn

    """
    --------------------------------------------------------------------------------------------------------------------
    Initialize Propagation Delay
    --------------------------------------------------------------------------------------------------------------------
    """
    if delay:
        if debug:
            print('------------------------------')
            print('Initializing Propagation Delay')
            print('------------------------------')
        buffer_length = int(np.max(spike_delays) + 1)
        spike_buffer = np.zeros([buffer_length, num_neurons])

    """
    --------------------------------------------------------------------------------------------------------------------
    Set Outputs
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------')
        print('Setting Outputs')
        print('---------------')
    output_nodes = []
    index = 0
    for out in range(len(network.outputs)):
        source_pop = network.outputs[out]['source']
        num_source_neurons = network.populations[source_pop]['number']
        output_nodes.append([])
        for num in range(num_source_neurons):
            output_nodes[out].append(index)
            index += 1
    num_outputs = index

    output_voltage_connectivity = np.zeros(
        [num_outputs, num_neurons])  # initialize connectivity matrix
    if spiking:
        output_spike_connectivity = np.copy(output_voltage_connectivity)
    outputs = np.zeros(num_outputs)
    for out in range(len(network.outputs)):  # iterate over the connections in the network
        source_pop = network.outputs[out]['source']  # get the source
        for i in range(len(pops_and_nrns[source_pop])):
            if network.outputs[out]['spiking']:
                output_spike_connectivity[output_nodes[out][i]][
                    pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination
            else:
                output_voltage_connectivity[output_nodes[out][i]][
                    pops_and_nrns[source_pop][i]] = 1.0  # set the weight in the correct source and destination

    """
    --------------------------------------------------------------------------------------------------------------------
    Arrange states and parameters into dictionary
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('---------------------')
        print('Writing to Dictionary')
        print('---------------------')
    params = {'dt': dt,
              'name': name,
              'spiking': spiking,
              'delay': delay,
              'elec': electrical,
              'rect': electrical_rectified,
              'gated': gated,
              'numChannels': num_channels,
              'v': V,
              'vLast': V_last,
              'vRest': V_rest,
              'v0': V_0,
              'cM': c_m,
              'gM': g_m,
              'iB': i_b,
              'gMaxNon': g_max_non,
              'delE': del_e,
              'eLo': e_lo,
              'eHi': e_hi,
              'timeFactorMembrane': time_factor_membrane,
              'inputConn': input_connectivity,
              'numPop': num_populations,
              'numNeurons': num_neurons,
              'numConn': num_connections,
              'numInputs': num_inputs,
              'numOutputs': num_outputs,
              # 'r': R,
              'outConnVolt': output_voltage_connectivity}
    if spiking:
        params['spikes'] = spikes
        params['theta0'] = theta_0
        params['theta'] = theta
        params['thetaLast'] = theta_last
        params['m'] = m
        params['tauTheta'] = tau_theta
        params['gMaxSpike'] = g_max_spike
        params['gSpike'] = g_spike
        params['tauSyn'] = tau_syn
        params['timeFactorThreshold'] = time_factor_threshold
        params['timeFactorSynapse'] = time_factor_synapse
        params['outConnSpike'] = output_spike_connectivity
        params['thetaLeak'] = theta_leak
        params['thetaIncrement'] = theta_increment
        params['thetaFloor'] = theta_floor
        params['vReset'] = V_reset
        params['gIncrement'] = g_increment

        # Mark added these
        params['spikeTime'] = None
        params['post1_spike_diff'] = None
        params['post2_spike_diff'] = None
        params['post1_counter'] = None
        params['post2_counter'] = None
        params['in1_spike_diff'] = None
        params['in2_spike_diff'] = None
        params['in3_spike_diff'] = None
        params['in4_spike_diff'] = None

        # V2 Additions
        params['pre_spike_diff'] = None
        params['post_spike_diff'] = None
        params['post_counter'] = None
        params['num_pre'] = None
        params['num_post'] = None
        params['ltp_a'] = None
        params['ltp_t'] = None
        params['ltd_a'] = None
        params['ltd_t'] = None
        params['max_conductance'] = None

    if delay:
        params['spikeDelays'] = spike_delays
        params['spikeRows'] = spike_rows
        params['spikeCols'] = spike_cols
        params['bufferSteps'] = buffer_steps
        params['bufferNrns'] = buffer_nrns
        params['delayedSpikes'] = delayed_spikes
        params['spikeBuffer'] = spike_buffer
    if electrical:
        params['gElectrical'] = g_electrical
    if electrical_rectified:
        params['gRectified'] = g_rectified
    if gated:
        params['gIon'] = g_ion
        params['eIon'] = e_ion
        params['powA'] = pow_a
        params['slopeA'] = slope_a
        params['kA'] = k_a
        params['eA'] = e_a
        params['powB'] = pow_b
        params['slopeB'] = slope_b
        params['kB'] = k_b
        params['eB'] = e_b
        params['tauMaxB'] = tau_max_b
        params['powC'] = pow_c
        params['slopeC'] = slope_c
        params['kC'] = k_c
        params['eC'] = e_c
        params['tauMaxC'] = tau_max_c
        params['bGate'] = b_gate
        params['bGateLast'] = b_gate_last
        params['bGate0'] = b_gate_0
        params['cGate'] = c_gate
        params['cGateLast'] = c_gate_last
        params['cGate0'] = c_gate_0

    """
    --------------------------------------------------------------------------------------------------------------------
    Passing params to backend object
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('-------------------------------------------------')
        print('Passing states and parameters to SNS_Numpy object')
        print('-------------------------------------------------')
    model = SNS_Numpy_standard(params)

    """
    --------------------------------------------------------------------------------------------------------------------
    Final print
    --------------------------------------------------------------------------------------------------------------------
    """
    if debug:
        print('----------------------------')
        print('Final states and parameters:')
        print('----------------------------')
        print('Input Connectivity:')
        print(input_connectivity)
        print('g_max_non:')
        print(g_max_non)
        if spiking:
            print('GmaxSpike:')
            print(g_max_spike)
            print('Theta Increment:')
            print(g_increment)
        print('del_e:')
        print(del_e)
        print('e_lo:')
        print(e_lo)
        print('e_hi:')
        print(e_hi)
        if electrical:
            print('Gelectrical:')
            print(g_electrical)
        if electrical_rectified:
            print('GelectricalRectified:')
            print(g_rectified)
        print('Output Voltage Connectivity')
        print(output_voltage_connectivity)
        if spiking:
            print('Output Spike Connectivity:')
            print(output_spike_connectivity)
        print('v:')
        print(V)
        print('v_last:')
        print(V_last)
        print('v_rest:')
        print(V_rest)
        if spiking:
            print('theta_0:')
            print(theta_0)
            print('ThetaLast:')
            print(theta_last)
            print('Theta:')
            print(theta)
            print('ThetaLeak:')
            print(theta_leak)
            print('ThetaIncrement:')
            print(theta_increment)
            print('ThetaFloor:')
            print(theta_floor)
            print('v_reset:')
            print(V_reset)
        if gated:
            print('Number of Channels:')
            print(num_channels)
            print('Ionic Conductance:')
            print(g_ion)
            print('Ionic Reversal Potentials:')
            print(e_ion)
            print('A Gate Parameters:')
            print('Power:')
            print(pow_a)
            print('Slope:')
            print(slope_a)
            print('K:')
            print(k_a)
            print('Reversal Potential:')
            print(e_a)
            print('B Gate Parameters:')
            print('Power:')
            print(pow_b)
            print('Slope:')
            print(slope_b)
            print('K:')
            print(k_b)
            print('Reversal Potential:')
            print(e_b)
            print('Tau Max:')
            print(tau_max_b)
            print('B:')
            print(b_gate)
            print('B_last:')
            print(b_gate_last)
            print('C Gate Parameters:')
            print('Power:')
            print(pow_c)
            print('Slope:')
            print(slope_c)
            print('K:')
            print(k_c)
            print('Reversal Potential:')
            print(e_c)
            print('Tau Max:')
            print(tau_max_c)
            print('B:')
            print(c_gate)
            print('B_last:')
            print(c_gate_last)

    return model

"""
Simulation backends for synthetic nervous system networks. Each of these are python-based, and are constructed using a
Network. They can then be run for a step, with the inputs being a vector of neural states and applied currents and the
output being the next step of neural states.
"""

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IMPORTS
"""

from typing import Dict
import numpy as np

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BACKENDS
"""

class Backend:

    def __init__(self, params: Dict) -> None:
        self.set_params(params)

    def forward(self, x=None, current_time=None, dt=None, dynamic_threshold=None, g_update=None):
        raise NotImplementedError

    # Sets "self." parameters as pulled from dictionary in "/compilers.py" file. Variables need to be defined in dict
    def set_params(self, params: Dict) -> None:
        self.dt = params['dt']
        self.name = params['name']
        self.spiking = params['spiking']
        self.delay = params['delay']
        self.electrical = params['elec']
        self.electrical_rectified = params['rect']
        self.gated = params['gated']
        self.num_channels = params['numChannels']
        self.V = params['v']
        self.V_last = params['vLast']
        self.V_0 = params['v0']
        self.V_rest = params['vRest']
        self.c_m = params['cM']
        self.g_m = params['gM']
        self.i_b = params['iB']
        self.g_max_non = params['gMaxNon']
        self.del_e = params['delE']
        self.e_lo = params['eLo']
        self.e_hi = params['eHi']
        self.time_factor_membrane = params['timeFactorMembrane']
        self.input_connectivity = params['inputConn']
        self.output_voltage_connectivity = params['outConnVolt']
        self.num_populations = params['numPop']
        self.num_neurons = params['numNeurons']
        self.num_connections = params['numConn']
        self.num_inputs = params['numInputs']
        self.num_outputs = params['numOutputs']
        # self.R = params['r']
        if self.spiking:
            self.spikes = params['spikes']

            # Mark added these
            self.spike_time = params['spikeTime']
            # self.POST1_spike_diff = params['post1_spike_diff']
            # self.POST2_spike_diff = params['post2_spike_diff']
            # self.POST1_counter = params['post1_counter']
            # self.POST2_counter = params['post2_counter']
            # self.IN1_spike_diff = params['in1_spike_diff']
            # self.IN2_spike_diff = params['in2_spike_diff']
            # self.IN3_spike_diff = params['in3_spike_diff']
            # self.IN4_spike_diff = params['in4_spike_diff']

            # V2
            self.pre_spike_diff = params['pre_spike_diff']
            self.post_spike_diff = params['post_spike_diff']
            self.post_counter = params['post_counter']
            self.num_pre = params['num_pre']
            self.num_post = params['num_post']
            self.ltp_a = params['ltp_a']
            self.ltp_t = params['ltp_t']
            self.ltd_a = params['ltd_a']
            self.ltd_t = params['ltd_t']
            self.max_conductance = params['max_conductance']

            self.theta_0 = params['theta0']
            self.theta = params['theta']
            self.theta_last = params['thetaLast']
            self.m = params['m']
            self.tau_theta = params['tauTheta']
            self.g_max_spike = params['gMaxSpike']
            self.g_spike = params['gSpike']
            self.tau_syn = params['tauSyn']
            self.time_factor_threshold = params['timeFactorThreshold']
            self.time_factor_synapse = params['timeFactorSynapse']
            self.output_spike_connectivity = params['outConnSpike']
            self.theta_leak = params['thetaLeak']
            self.theta_increment = params['thetaIncrement']
            self.theta_floor = params['thetaFloor']
            self.V_reset = params['vReset']
            self.g_increment = params['gIncrement']
        if self.delay:
            self.spike_delays = params['spikeDelays']
            self.spike_rows = params['spikeRows']
            self.spike_cols = params['spikeCols']
            self.buffer_steps = params['bufferSteps']
            self.buffer_nrns = params['bufferNrns']
            self.delayed_spikes = params['delayedSpikes']
            self.spike_buffer = params['spikeBuffer']
        if self.electrical:
            self.g_electrical = params['gElectrical']
        if self.electrical_rectified:
            self.g_rectified = params['gRectified']
        if self.gated:
            self.g_ion = params['gIon']
            self.e_ion = params['eIon']
            self.pow_a = params['powA']
            self.slope_a = params['slopeA']
            self.k_a = params['kA']
            self.e_a = params['eA']
            self.pow_b = params['powB']
            self.slope_b = params['slopeB']
            self.k_b = params['kB']
            self.e_b = params['eB']
            self.tau_max_b = params['tauMaxB']
            self.pow_c = params['powC']
            self.slope_c = params['slopeC']
            self.k_c = params['kC']
            self.e_c = params['eC']
            self.tau_max_c = params['tauMaxC']
            self.b_gate = params['bGate']
            self.b_gate_last = params['bGateLast']
            self.b_gate_0 = params['bGate0']
            self.c_gate = params['cGate']
            self.c_gate_last = params['cGateLast']
            self.c_gate_0 = params['cGate0']

    # This runs when the compiled network is called to step the simulation forward
    def __call__(self, x=None, current_time=None, dt=None, dynamic_threshold=None, g_update=None):
        return self.forward(x, current_time, dt, dynamic_threshold, g_update)

    def reset(self):
        raise NotImplementedError

"""
OUR BELOVED MODIFIED NUMPY BACKEND
"""

class SNS_Numpy(Backend):
    def __init__(self, params: Dict) -> None:
        super().__init__(params)

    ''' This completes one timestep of the simulation. Variables are carried from one call to the next '''
    def forward(self, x=None, current_time=None, dt=None, dynamic_threshold=None, g_update=None):
        self.V_last = np.copy(self.V)
        if x is None:
            i_app = 0
        else:
            i_app = np.matmul(self.input_connectivity, x)  # Apply external current sources to their destinations
        
        # Allows for an input value to alter the spiking threshold of the neurons. Requires array input (not single int). Can handle multiple adjustments.
        # Needs array of thresholds in same order as neurons are added to network.
        if dynamic_threshold is not None:
            self.theta[-len(dynamic_threshold):] = dynamic_threshold

        """
        Updating the synaptic Conductance values.
        g_increment and g_max_spike are [n]x[n] maxricies, where n is the number of neurons. 
        Each column represents a neuron's connections. If a row is populated, there is a connection from that column's neuron to that row's neuron.
        """
        if g_update is not None:
            if g_update.shape == self.g_increment.shape: 
                # Force change the conductivity matrix used in calculations
                self.g_increment = g_update
                self.g_max_spike = g_update
            else:
                print('Uh oh! There is a mismatch between the shape of g_update and the reqired shape for the system')

        # YES
        g_syn = np.maximum(0, np.minimum(self.g_max_non * ((self.V_last - self.e_lo) / (self.e_hi - self.e_lo)), self.g_max_non))
        # In my work, this always resets g_syn to zero. No values in a 2x2 for two neurons
        
        # YES
        if self.spiking:
            self.theta_last = np.copy(self.theta)
            self.g_spike = self.g_spike * (1 - self.time_factor_synapse)
            g_syn += self.g_spike
            # From this point on, g_syn is equal to the current g_spike. g_spike slowly decays

        # YES
        i_syn = np.sum(g_syn * self.del_e, axis=1) - self.V_last * np.sum(g_syn, axis=1)

        """ Refractory Period """
        # If a postsynaptic neuron spiked within the past 30 time steps, do not allow any current to pass to it
        # Check over all postsynaptic neurons
        for post in range(self.num_post):
            # If a specific postsynaptic neuron's counter is less than 30 loop iterations
            if self.post_counter[post] < 30:
                # Up the counter by one
                self.post_counter[post] += 1
                # print('POST COUNTER: ' + str(self.post_counter[post]) + ' At Time: ' + str(current_time))
                # Prevent the neuron from receiving input current
                i_syn[self.num_pre + post] = 0

        """ END OF REFRACTORY PERIOD CODE """
        
        # NO
        if self.electrical:
            i_syn += (np.sum(self.g_electrical * self.V_last, axis=1) - self.V_last * np.sum(self.g_electrical, axis=1))
        if self.electrical_rectified:
            # create mask
            mask = np.subtract.outer(self.V_last, self.V_last).transpose() > 0
            masked_g = mask * self.g_rectified
            diag_masked = masked_g + masked_g.transpose() - np.diag(masked_g.diagonal())
            i_syn += np.sum(diag_masked * self.V_last, axis=1) - self.V_last * np.sum(diag_masked, axis=1)
        if self.gated:
            a_inf = 1 / (1 + self.k_a * np.exp(self.slope_a * (self.e_a - self.V_last)))
            b_inf = 1 / (1 + self.k_b * np.exp(self.slope_b * (self.e_b - self.V_last)))
            c_inf = 1 / (1 + self.k_c * np.exp(self.slope_c * (self.e_c - self.V_last)))

            tau_b = self.tau_max_b * b_inf * np.sqrt(self.k_b * np.exp(self.slope_b * (self.e_b - self.V_last)))
            tau_c = self.tau_max_c * c_inf * np.sqrt(self.k_c * np.exp(self.slope_c * (self.e_c - self.V_last)))

            self.b_gate_last = np.copy(self.b_gate)
            self.c_gate_last = np.copy(self.c_gate)

            self.b_gate = self.b_gate_last + self.dt * ((b_inf - self.b_gate_last) / tau_b)
            self.c_gate = self.c_gate_last + self.dt * ((c_inf - self.c_gate_last) / tau_c)

            i_ion = self.g_ion * (a_inf ** self.pow_a) * (self.b_gate ** self.pow_b) * (self.c_gate ** self.pow_c) * (
                        self.e_ion - self.V_last)
            i_gated = np.sum(i_ion, axis=0)

            self.V = self.V_last + self.time_factor_membrane * (
                        -self.g_m * (self.V_last - self.V_rest) + self.i_b + i_syn + i_app + i_gated)  # Update membrane potential
        
        # YES
        else:
            self.V = self.V_last + self.time_factor_membrane * (
                        -self.g_m * (self.V_last - self.V_rest) + self.i_b + i_syn + i_app)  # Update membrane potential

        # YES
        if self.spiking:
            self.theta = self.theta_last + self.time_factor_threshold * (self.theta_leak*(self.theta_0-self.theta_last) + self.m * (self.V_last - self.V_rest))  # Update the firing thresholds
            # print('before' + str(self.spikes))
            self.spikes = np.sign(np.minimum(0, self.theta - self.V))  # Compute which neurons have spiked
            # print('after' + str(self.spikes))

            # NO
            if self.delay:
                self.spike_buffer = np.roll(self.spike_buffer, 1, axis=0)  # Shift buffer entries down
                self.spike_buffer[0, :] = self.spikes  # Replace row 0 with the current spike data
                # Update a matrix with all of the appropriately delayed spike values
                self.delayed_spikes[self.spike_rows, self.spike_cols] = self.spike_buffer[
                    self.buffer_steps, self.buffer_nrns]

                self.g_spike += np.minimum((-self.delayed_spikes*self.g_increment), (-self.delayed_spikes)*(self.g_max_spike-self.g_spike))  # Update the conductance of connections which spiked
            
            # YES
            else:
                self.g_spike += np.minimum((-self.spikes*self.g_increment), (-self.spikes)*(self.g_max_spike-self.g_spike))  # Update the conductance of connections which spiked
            
            # YES
            self.V = ((self.V-self.V_reset) * (self.spikes + 1))+self.V_reset  # Reset the membrane voltages of neurons which spiked
            self.theta = np.maximum(self.theta_increment, self.theta_floor-self.theta)*(-self.spikes) + self.theta
        
        # YES
        self.outputs = np.matmul(self.output_voltage_connectivity, self.V) # Voltage outputs
        
        """
        STDP CODE
        VERSION 2
        FOR DYNAMIC NETWORK SIZES 
        """
        if self.spiking:
            # Checking PRE spikes LTD
            if np.sum(self.spikes[0:self.num_pre]) != 0:
                for i in range(self.num_pre):
                    # Check if PRE neuron i has spiked. If so, proceed
                    if self.spikes[i] != 0:
                        # If PRE neuron i has spiked, record that time in the spike_time matrix. This matrix records
                        # most recent spike times
                        self.spike_time[i] = current_time

                        # Since neuron i spiked, we can update the matrix of its relative spike times to the POST neurons
                        post_matrix = []
                        # For each postsynaptic neuron, calculate how long ago it spiked relative to now (PRE Spike)
                        for post in range(self.num_pre, self.num_pre + self.num_post):
                            post_matrix.append(self.spike_time[i] - self.spike_time[post])
                        self.pre_spike_diff[i] = post_matrix

                        # print('PreSpike. Time to most recent POST Spike')
                        # print(self.pre_spike_diff[i])

                        # Determine LTD. Iterate over all post-before-pre dt's
                        for j in range(self.num_post):
                            # If the post-before-pre is too great. Ignore
                            if post_matrix[j] < 30:
                                # Calculate weight update
                                weight_change = -self.ltd_a * np.exp(-post_matrix[j] / self.ltd_t)
                                # Apply weight update
                                self.g_max_spike[j + self.num_pre, i] = self.g_max_spike[j + self.num_pre, i] + weight_change
                                self.g_increment[j + self.num_pre, i] = self.g_increment[j + self.num_pre, i] + weight_change
                                # Check to ensure in bounds
                                if self.g_max_spike[j + self.num_pre, i] < 0.0:
                                    self.g_max_spike[j + self.num_pre, i] = 0.0
                                if self.g_max_spike[j + self.num_pre, i] > self.max_conductance:
                                    self.g_max_spike[j + self.num_pre, i] = self.max_conductance
                                if self.g_increment[j + self.num_pre, i] < 0.0:
                                    self.g_increment[j + self.num_pre, i] = 0.0
                                if self.g_increment[j + self.num_pre, i] > self.max_conductance:
                                    self.g_increment[j + self.num_pre, i] = self.max_conductance
                                
                                # print('CONDUCTANCE UPDATE! By: ' + str(weight_change))





                # print('PRE SPIKE: How long ago a postsynaptic neuron spiked')
                # print(self.pre_spike_diff)
            
            # Checking POST spikes LTP
            if np.sum(self.spikes[self.num_pre:]) != 0:
                for i in range(self.num_pre, self.num_pre + self.num_post):
                    if self.spikes[i] != 0:
                        self.spike_time[i] = current_time

                        pre_matrix = []
                        for pre in range(0, self.num_pre):
                            pre_matrix.append(self.spike_time[i] - self.spike_time[pre])
                        self.post_spike_diff[i - self.num_pre] = pre_matrix

                        # Reset the postsynaptic counter
                        self.post_counter[i - self.num_pre] = 0

                        # print('POST SPIKE. Time to most recent PRE spike')
                        # print(self.post_spike_diff[i - self.num_pre])

                        # Determine LTP. Iterate over all pre-before-post dt's
                        for j in range(self.num_pre):
                            # If the pre-before-post time difference is too great, ignore
                            if pre_matrix[j] < 40:
                                # Calculate weight update
                                weight_change = self.ltp_a * np.exp(-pre_matrix[j] / self.ltp_t)
                                # Apply weight update
                                self.g_max_spike[i, j] = self.g_max_spike[i, j] + weight_change
                                self.g_increment[i, j] = self.g_increment[i, j] + weight_change
                                # Check to ensure in bounds
                                if self.g_max_spike[i, j] < 0.0:
                                    self.g_max_spike[i, j] = 0.0
                                if self.g_max_spike[i, j] > self.max_conductance:
                                    self.g_max_spike[i, j] = self.max_conductance
                                if self.g_increment[i, j] < 0.0:
                                    self.g_increment[i, j] = 0.0
                                if self.g_increment[i, j] > self.max_conductance:
                                    self.g_increment[i, j] = self.max_conductance
                                                        
                                # print('CONDUCTANCE UPDATE! By: ' + str(weight_change))
        """ END OF STDP CODE """
        
        # YES
        if self.spiking:
            self.outputs += np.matmul(self.output_spike_connectivity, -self.spikes) # Spike outputs

        return self.outputs

    def reset(self):
        self.V = np.copy(self.V_0)
        self.V_last = np.copy(self.V_0)
        if self.spiking:
            self.theta = np.copy(self.theta_0)
            self.theta_last = np.copy(self.theta_0)
        if self.gated:
            self.b_gate = np.copy(self.b_gate_0)
            self.b_gate_last = np.copy(self.b_gate_0)
            self.c_gate = np.copy(self.c_gate_0)
            self.c_gate_last = np.copy(self.c_gate_0)

"""
Standard NUMPY Backend
"""

class SNS_Numpy_standard(Backend):
    def __init__(self, params: Dict) -> None:
        super().__init__(params)

    def forward(self, x=None, current_time=None, dt=None, dynamic_threshold=None, g_update=None):
        self.V_last = np.copy(self.V)
        if x is None:
            i_app = 0
        else:
            i_app = np.matmul(self.input_connectivity, x)  # Apply external current sources to their destinations
        g_syn = np.maximum(0, np.minimum(self.g_max_non * ((self.V_last - self.e_lo) / (self.e_hi - self.e_lo)), self.g_max_non))
        if self.spiking:
            self.theta_last = np.copy(self.theta)
            self.g_spike = self.g_spike * (1 - self.time_factor_synapse)
            g_syn += self.g_spike

        i_syn = np.sum(g_syn * self.del_e, axis=1) - self.V_last * np.sum(g_syn, axis=1)
        if self.electrical:
            i_syn += (np.sum(self.g_electrical * self.V_last, axis=1) - self.V_last * np.sum(self.g_electrical, axis=1))
        if self.electrical_rectified:
            # create mask
            mask = np.subtract.outer(self.V_last, self.V_last).transpose() > 0
            masked_g = mask * self.g_rectified
            diag_masked = masked_g + masked_g.transpose() - np.diag(masked_g.diagonal())
            i_syn += np.sum(diag_masked * self.V_last, axis=1) - self.V_last * np.sum(diag_masked, axis=1)
        if self.gated:
            a_inf = 1 / (1 + self.k_a * np.exp(self.slope_a * (self.e_a - self.V_last)))
            b_inf = 1 / (1 + self.k_b * np.exp(self.slope_b * (self.e_b - self.V_last)))
            c_inf = 1 / (1 + self.k_c * np.exp(self.slope_c * (self.e_c - self.V_last)))

            tau_b = self.tau_max_b * b_inf * np.sqrt(self.k_b * np.exp(self.slope_b * (self.e_b - self.V_last)))
            tau_c = self.tau_max_c * c_inf * np.sqrt(self.k_c * np.exp(self.slope_c * (self.e_c - self.V_last)))

            self.b_gate_last = np.copy(self.b_gate)
            self.c_gate_last = np.copy(self.c_gate)

            self.b_gate = self.b_gate_last + self.dt * ((b_inf - self.b_gate_last) / tau_b)
            self.c_gate = self.c_gate_last + self.dt * ((c_inf - self.c_gate_last) / tau_c)

            i_ion = self.g_ion * (a_inf ** self.pow_a) * (self.b_gate ** self.pow_b) * (self.c_gate ** self.pow_c) * (
                        self.e_ion - self.V_last)
            i_gated = np.sum(i_ion, axis=0)

            self.V = self.V_last + self.time_factor_membrane * (
                        -self.g_m * (self.V_last - self.V_rest) + self.i_b + i_syn + i_app + i_gated)  # Update membrane potential
        else:
            self.V = self.V_last + self.time_factor_membrane * (
                        -self.g_m * (self.V_last - self.V_rest) + self.i_b + i_syn + i_app)  # Update membrane potential
        if self.spiking:
            self.theta = self.theta_last + self.time_factor_threshold * (self.theta_leak*(self.theta_0-self.theta_last) + self.m * (self.V_last - self.V_rest))  # Update the firing thresholds
            self.spikes = np.sign(np.minimum(0, self.theta - self.V))  # Compute which neurons have spiked

            # New stuff with delay
            if self.delay:
                self.spike_buffer = np.roll(self.spike_buffer, 1, axis=0)  # Shift buffer entries down
                self.spike_buffer[0, :] = self.spikes  # Replace row 0 with the current spike data
                # Update a matrix with all of the appropriately delayed spike values
                self.delayed_spikes[self.spike_rows, self.spike_cols] = self.spike_buffer[
                    self.buffer_steps, self.buffer_nrns]

                self.g_spike += np.minimum((-self.delayed_spikes*self.g_increment), (-self.delayed_spikes)*(self.g_max_spike-self.g_spike))  # Update the conductance of connections which spiked
            else:
                self.g_spike += np.minimum((-self.spikes*self.g_increment), (-self.spikes)*(self.g_max_spike-self.g_spike))  # Update the conductance of connections which spiked
            self.V = ((self.V-self.V_reset) * (self.spikes + 1))+self.V_reset  # Reset the membrane voltages of neurons which spiked
            self.theta = np.maximum(self.theta_increment, self.theta_floor-self.theta)*(-self.spikes) + self.theta
        self.outputs = np.matmul(self.output_voltage_connectivity, self.V)
        if self.spiking:
            self.outputs += np.matmul(self.output_spike_connectivity, -self.spikes)

        return self.outputs

    def reset(self):
        self.V = np.copy(self.V_0)
        self.V_last = np.copy(self.V_0)
        if self.spiking:
            self.theta = np.copy(self.theta_0)
            self.theta_last = np.copy(self.theta_0)
        if self.gated:
            self.b_gate = np.copy(self.b_gate_0)
            self.b_gate_last = np.copy(self.b_gate_0)
            self.c_gate = np.copy(self.c_gate_0)
            self.c_gate_last = np.copy(self.c_gate_0)

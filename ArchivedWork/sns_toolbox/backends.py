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
            self.POST1_spike_diff = params['post1_spike_diff']
            self.POST2_spike_diff = params['post2_spike_diff']
            self.POST1_counter = params['post1_counter']
            self.POST2_counter = params['post2_counter']
            self.IN1_spike_diff = params['in1_spike_diff']
            self.IN2_spike_diff = params['in2_spike_diff']
            self.IN3_spike_diff = params['in3_spike_diff']
            self.IN4_spike_diff = params['in4_spike_diff']

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

        """
        Refractory Period
        """
        # This is not currently working as a variable. Just 10Hz. Do not change, it will not do what you want it to do
        max_spiking_frequency = 10 # Hz

        # Checks the spiking condition of the last iteration of the loop
        # If POST1 has spiked, set the counter to zero.
        if self.spikes[4] != 0:
            self.POST1_counter = 0
        if self.spikes[5] != 0:
            self.POST2_counter = 0
        
        # If the counter has not reached its 10ms time period, run this code
        if self.POST1_counter < int(max_spiking_frequency / dt):
            # Step up the counter by one
            self.POST1_counter += 1
            # Set the current to the neuron to zero
            i_syn[4] = 0.0

        # If the counter has not reached its 10ms time period, run this code
        if self.POST2_counter < int(max_spiking_frequency/dt):
            # Step up the counter by one
            self.POST2_counter += 1
            # Set the current to the neuron to zero
            i_syn[5] = 0.0

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
        self.outputs = np.matmul(self.output_voltage_connectivity, self.V)
        
        """
        STDP CODE
        CHECKS SPIKING, RECORDS SPIKE TIME IN self.spike_time
        """
        # Spikes only show up after line 200 for the rest of the loop. 
        if self.spiking:
            # IN1, IN2, IN3, IN4, POST1, POST2
            if np.sum(self.spikes) != 0:
                # print('Spike')
                if current_time is not None:
                    for i in range(len(self.spikes)):
                        # Checks to see if there is a spike in the current slot
                        if self.spikes[i] != 0:
                            self.spike_time[i] = -self.spikes[i] * current_time

                            """ IN1 Spike Time Difference """
                            if i == 0:
                                self.IN1_spike_diff = []
                                # Just checking POST1 and POST2
                                for i in range(4, 6):
                                    self.IN1_spike_diff.append(self.spike_time[0] - self.spike_time[i])
                                # print(self.IN1_spike_diff)

                            """ IN2 Spike Time Difference """
                            if i == 1:
                                self.IN2_spike_diff = []
                                # Just checking POST1 and POST2
                                for i in range(4, 6):
                                    self.IN2_spike_diff.append(self.spike_time[1] - self.spike_time[i])
                                # print(self.IN2_spike_diff)

                            """ IN3 Spike Time Difference """
                            if i == 2:
                                self.IN3_spike_diff = []
                                # Just checking POST1 and POST2
                                for i in range(4, 6):
                                    self.IN3_spike_diff.append(self.spike_time[2] - self.spike_time[i])
                                # print(self.IN3_spike_diff)

                            """ IN4 Spike Time Difference """
                            if i == 3:
                                self.IN4_spike_diff = []
                                # Just checking POST1 and POST2
                                for i in range(4, 6):
                                    self.IN4_spike_diff.append(self.spike_time[3] - self.spike_time[i])
                                # print(self.IN4_spike_diff)


                            """ POST1 Spike Time Difference """
                            if i == 4:
                                self.POST1_spike_diff = []
                                for i in range(4):
                                    if self.spike_time[i] != 0:
                                        self.POST1_spike_diff.append(self.spike_time[i] - self.spike_time[4])
                                    else:
                                        self.POST1_spike_diff.append(None)
                                # print('Post 1 Spiked')
                                # print(POST1_spike_diff)
                                # print(self.spike_time)

                            """ POST2 Spike Time Difference """
                            if i == 5:
                                self.POST2_spike_diff = []
                                for i in range(4):
                                    if self.spike_time[i] != 0:
                                        self.POST2_spike_diff.append(self.spike_time[i] - self.spike_time[5])
                                    else:
                                        self.POST2_spike_diff.append(-100)
                                # print('Post 2 Spiked')
                                # print(POST2_spike_diff)
                                # print(self.spike_time)

                    # print(self.spike_time)
        """ END OF MY ADDED SPIKE TIME DIFFERENCE CODE """

        # YES
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

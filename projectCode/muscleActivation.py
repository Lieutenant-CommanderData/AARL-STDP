import numpy as np
from random import randint


class MuscleAct():
    '''Functions to generate muscle activation signals.

    * Random activation through randActivation()
    * Pick specific muscles for activation through manualActivation()
    '''
    def randActivation(self, t, pair_freq, indi_freq, num_motors):
        '''Generates semi-random muscle activation.
        
        Args:
            t >>> Simulation time used to build array of correct length
            pair_freq >>> Number of time steps to switch active pair
            indi_freq >>> Number of time steps to switch active individual motor neuron in selected pair
            num_motors >>> Not implemented. Will use to allow for models with a larger number of muscles. For now, insert the number of motor neurons
        
        For extra clarity, pair_freq will pick one, both, or neither pair of agonistic muscles (grouped as 1 and 2 / 3 and 4). Once pair_freq picks a pair, indi_freq will select which of those neurons is active (if pair 1 and 2 are selected, indi_freq can select the on/off status of neuron 1 or 2 individually). 
        '''
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
        

    def manualActivation(self, t, num_motors, active):
        '''Generates muscle activation for specificed muscles. Activates the muscles you require for the whole time period.
        
        Args:
            t >>> Simulation time used to build array of correct length
            num_motors >>> Not implemented. Will allow for model scaling in the future. For now, insert the number of motor neurons
            active >>> Which muscle/muscles are active. Uses bitwise logic, so for four muscles, insert 0b0000, changing the 0 to a 1 for which neuron you want active (left to right = neuron 1 to 4)
        '''
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

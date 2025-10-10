If a spike occurs, the counter will be reset to zero.
Each loop cycle counts the counter up by one. 

The neuron will not be allowed to spike (the value i_syn will be set to zero) until the counter has reached a certain value.

so 100 Hz max, which is a 10ms frequency. This means that 10 / timestep will equal the number of loops that need to be completed before current can be allowed back into the neuron.


Where i am at: If the counter is not above 10/dt steps, set the current to POST1 to zero. Step up each time this condition is not met. 

If the counter is above the 10/dt steps, the current into the neuron is not limited. It is only limited if the counter is below the threshold.

If the POST1 spikes, the counter is reset to zero, disallowing the POST1 to spike
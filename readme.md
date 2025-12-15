# STDP Code Base
This provides an overview of the code for my thesis project, utilizing STDP to accomplish unsupervised learning in neural networks. 

## STDPCODE Folder
Contains the main python file ([main.py](main.py)) that runs the full simulation.

## projectCode Folder

### mujocoParameters.py
Stores two MuJoCo related functions.
* [showModel](./projectCode/mujocoParameters.py#L15): this function takes a MuJoCo model and data input, renders the scene, and saves a .png to the /results folder. 
* [popOutModel](./projectCode/mujocoParameters.py#L31): when the path to a MuJoCo model is passed in, this function will launch the interactive MuJoCo scene viewer. This will cause the code to hang until the window is closed. Inside this window, the model can be interacted with, enabling each muscle to be activated and for live view of system dynamics as well as a live feed of the sensor data.  

### constants.py
Defines and stores the bulk of the constants used in this code.

### netwroks.py
Defines the parameters for the neurons/synapses in the networks and constructs the networks using SNS-Toolbox.

### movementToCurrent.py
This file contains the two functions necessary to convert the sensor data from MuJoCo into a electrical current value needed to simulate the Ia sensory neurons. 
* [vel2cur](./projectCode/movementToCurrent.py#L19) takes the length and velocity sensor data from the MuJoCo model and, using Prochazka's equation, calculates the necessary sensory neuron spiking frequency.
* [freq2cur](./projectCode/movementToCurrent.py#L12) is an accompanying function to vel2cur, for once vel2cur calculates the necessary spike frequency, it needs to convert it into an amount of electrical current which will invoke this spike rate in the sensory neurons.  

### calculations.py
Contains a couple of basic functions used throughout the code.
* [getFreq](./projectCode/calculations.py#L11) calculates the spike frequency for an array containing the spikes of an intividiaul neuron. Takes an input of an array of 0s and 1s (spiking nature of a neuron) alongside the timestep for that array and returns the spike frequency as an equal sized array.
* [NoisyAmps](./projectCode/calculations.py#L24) adds noise to an array of electrical current sent to a neuron. Takes in an array and a percent noise level and adds that percent noise to the entire array. 

### muscleActivation.py
Contains the functions that generate the electrical current patterns used to activate the spiking motor neuorns. 
* [randActivation](./projectCode/muscleActivation.py#L11) creates semi-random muscle activation patterns. Allows for two inputs controlling the number of timesteps between re-randomizing which synergistic pairs are active (using pair_freq) and which individual motor neurons are active (using indi_freq). Currently only implemented for a total of four muscles.
* [manualActivation](./projectCode/muscleActivation.py#L52) creates "always-on" muscle activation for specific muscles in the model. Function uses bitwise logic to turn on specific muscles for the entirity of the simulation. Using the input "active" allows the user to define which muscle is active by changing an element in the all-off state of 0b0000 to a 1. (Left to right for the four 0s are the four muscles in the model in order of definition in MuJoCo)

### NetworkGeneration_MN.py
This file contains the function ([MNNetworkGenerator](./projectCode/NetworkGeneration_MN.py#L6)) which creates the SNS-Toolbox neural network for the motor neurons that are used to generate spiking signals in order to activate the muscles in the MuJoCo model. Given the number of muscles in the model, it will create a network with a spiking neuron (using the neuron model as input when calling the function) for each muscle in the model as well as inputs and outputs for each individual muscle as needed. This function will return an uncompiled SNS-Toolbox network. 

### NetworkGeneration.py
This file contains the function ([STDPNetworkGenerator](./projectCode/NetworkGeneration.py#L8)) which creates the SNS-Toolbox nerual network for Ia sensory neuron learning with STDP. Given the number of presynaptic and postsynaptic neurons (as well as the nueron/synapse type as defined by SNS-Toolbox neuorns and synapses), it will populate the network, connect the two layers, and return the uncompiled SNS-Toolbox network.

### plotGenerator.py
Creates plots based on the data recorded during a simulation trial. A variety of plots can be generated from the [plotGenerator](./projectCode/plotGenerator.py) class. More information on the specifics of each plot type can be found in the .py file, but as a basic overview:
* plot_mj plots the velocity and length data recorded from the sensors in the MuJoCo model alongside the current to activate the Ia sensory neurons as calculated from the sensor data.
* plot_postSpikes plots each spike from the postsynaptic Ia interneurons.
* plot_postFreq plots the spike frequency of the postsynaptic Ia interneurons.
* plot_postFreq_sample plots the spike frequency of a selected group of postsynaptic Ia interneurons (if you want to see only a set group).
* plot_conductance plots the change in conductance for the synaptic connections between the Ia sensory neuorns and the Ia interneurons. This is the main plot used in this research to determine the change in connection strength between the sensory neurons and the interneurons. 
* plot_mnActivity plots the activity of the motor neuorns, both the individual spikes sent out from the motor neurons as well as their spiking frequency.
* plot_postFreqIaCurr plots the spike frequency of the postsynaptic neurons alongside the current that is sent to the Ia sensory neurons. 
* plot_postFreqIaCurr plots the spike frequency for a selected set of postsynaptic neurons alongside the current that is sent to the Ia sensory neurons.
* plot_essentials plots what I consider to be the essential plots needed to analyze the results of a simulation trial. Running this will run and save the image of: plot_mj, plot_conductive, plot_postFreq, and plot_mnActivity.

### simulation.py
This code contains the main simulation loop ([runSimulation](./projectCode/simulation.py#L15)) for a experimental trial. Calling this function in the main.py file will run a full simulation, updating all of the variables which track model sensors and network features. It requires the inputs of a MuJoCo model and data, with optional inputs to change the simulation time (natively uses the total simulation time in [constants](./projectCode/constants.py)) or select whether the simulation loop renders and saves a video of the MuJoCo model.

## Results Folder
This folder contains the .png / .mp4 files which can be generated from the simulation. Examples include: 
* Still images of the MuJoCo .xml model in its initial position
    * Created using [mujocoParameters.showModel](./projectCode/mujocoParameters.py#L9)
* Plots of the data recorded from the simulation
    * Created using the functions inside the [plotGenerator](./projectCode/plotGenerator.py) class. Various functions allow for the creation of plots for the data (specifics explained in plotGenerator section above)
* Video of the MuJoCo leg swinging throughout simulation
    * Created by setting captureVideo to True in [runSimulation](./projectCode/simulation.py#L15)

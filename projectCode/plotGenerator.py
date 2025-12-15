import numpy as np
import matplotlib
from typing import Optional, Tuple
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import projectCode.constants as c
from .calculations import getFreq

class Plot:
    '''Container for all of your plotting needs!
    
    Args:
        dpi >>> Adjust the dpi of the saved .png files. Defaults to 600dpi (High Quality / Slower Processing)

    Plotting Options:
    * plot_mj >>> plots the MuJoCo data (muscle length/velocity and generated current)
    * plot_postSpikes >>> plots the spike instances of the postsynaptic neurons
    * plot_postFreq >>> plots the frequency of the postsynaptic neurons
    * plot_postFreq_sample >>> plots the frequency of a selection of postsynaptic neurons (if you just want one or two plotted)
    * plot_conductance >>> plots the chance in conductance throughout the simulation
    * plot_mnActivity >>> plots the activity of the motor neurons (spike instances and frequency)
    * plot_postFreqIaCurr >>> plots the postsynaptic neuron spike frequency alongside the Ia current sent to the presynaptic neurons
    * plot_postFreqIaCurr_sample >>> plots the postsynaptic neuron spike frequency for a selection of neurons alongside the Ia current sent to the presynaptic neurons
    * plot_essentials >>> plots the key plots all in one command (plot_mj, plot_postFreq, plot_conductance, plot_mnActivity)
    '''
    def __init__(self, dpi: Optional[int]=600):
        self.dpi = dpi
        
        pass

    def plot_mj(self, x_zoom: Optional[Tuple[float, float]]=None):
        '''Plots the data from MuJoCo: muscle length, muscle velocity, and Ia current produced from muscle movement.
        
        Args:
            x_zoom >>> 1x2 matrix in [ ] to zoom into a specific section of the plot. Defined by seconds. By default, it will show the entire plot
        '''
        # Length
        plt.figure(figsize=[12, 6])
        plt.subplot(3, 1, 1)
        for i in range(c.MJ_MUSCLE_NUM):
            if i % 2 == 0:
                plt.plot(c.mj_t, 1000*c.mj_length_data[i], label=('Muscle ' + str(i + 1)), linewidth=3.5, alpha=0.7)
            else:
                plt.plot(c.mj_t, 1000*c.mj_length_data[i], label=('Muscle ' + str(i + 1)), linewidth=0.75)
        plt.title(' MuJoCo Muscle Length Data')
        plt.ylabel('Length (mm)')
        if x_zoom is not None:
            plt.xlim(x_zoom)
        plt.legend(loc='upper right')

        # Velocity
        plt.subplot(3, 1, 2)
        for i in range(c.MJ_MUSCLE_NUM):
            if i % 2 == 0:
                plt.plot(c.mj_t, 1000*c.mj_velocity_data[i], label=('Muscle ' + str(i + 1)), linewidth=3.5, alpha=0.7)
            else:
                plt.plot(c.mj_t, 1000*c.mj_velocity_data[i], label=('Muscle ' + str(i + 1)), linewidth=0.75, alpha=1.0)
        plt.title('MuJoCo Muscle Velocity Data')
        plt.ylabel('Velocity (mm/s)')
        if x_zoom is not None:
            plt.xlim(x_zoom)
        plt.legend(loc='upper right')

        # Current into STDP SNS
        plt.subplot(3, 1, 3)
        for i in range(c.MJ_MUSCLE_NUM):
            if i % 2 == 0:
                plt.plot(c.mj_t, c.stdp_activation_current[i], label=('Ia Sensory Neuron ' + str(i + 1)), linewidth=3.5, alpha=0.7)
            else:
                plt.plot(c.mj_t, c.stdp_activation_current[i], label=('Ia Sensory Neuron ' + str(i + 1)), linewidth=0.75, alpha=1.0)
        plt.title('Ia Sensory Neuron Activation Current')
        plt.legend(loc='upper right')
        plt.ylabel('Curernt (nA)')
        plt.xlabel('Time (s)')
        if x_zoom is not None:
            plt.xlim(x_zoom)
        plt.subplots_adjust(hspace=0.5)

        plt.savefig('./results/mjdata.png', dpi=self.dpi)
        
    def plot_postSpikes(self, x_zoom: Optional[Tuple[float, float]]=None):
        '''Plots the spikes of the postsynaptic neuorns. This will plot every single spike that occurs, so it is only helpful when looking at a very small time window. Otherwise the spikes just form blocks of color.
        
        Args:
            x_zoom >>> 1x2 matrix in [ ] to zoom into a specific section of the plot. Defined by seconds. By default, it will show the entire plot
        '''
        plt.figure(figsize=[25, 5])
        for i in range(c.STDP_POST_NUM):
            plt.plot(c.mj_t, c.stdp_data[i] * (1 - 0.05*i), linewidth='1', label=('Post: ' + str(i + 1)))
        if x_zoom is not None:
            plt.xlim(x_zoom)
        plt.ylabel('Spike')
        plt.xlabel('Time (s)')
        plt.title('Postsynaptic Spikes')
        plt.legend(loc='upper right')

        plt.savefig('./results/postSpikes.png', dpi=self.dpi)

    def plot_postFreq(self, x_zoom: Optional[Tuple[float, float]]=None):
        '''Plots the frequency of all postsynaptic neurons.
        
        Args:
            x_zoom >>> 1x2 matrix in [ ] to zoom into a specific section of the plot. Defined by seconds. By default, it will show the entire plot
        '''
        plt.figure(figsize=[12, 7])
        for i in range(c.STDP_POST_NUM):
            freq, loc = getFreq(data=c.stdp_data[i,:], dt=c.sns_dt)
            # print(np.mean(freq))
            plt.plot(loc/1000, freq, label=('Interneuron ' + str(i + 1)))
        plt.ylim()
        if x_zoom is not None:
            plt.xlim(x_zoom)
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title('Spike Frequency of Postsynaptic Interneurons')
        plt.legend(loc='upper right')

        plt.savefig('./results/postFreq.png', dpi=self.dpi)

    def plot_postFreq_sample(self, whichNeurons: Tuple[int, ...], x_zoom: Optional[Tuple[float, float]]=None):
        '''Plots the frequency of a specific set of postsynaptic neurons.
        
        Args:
            whichNeurons >>> In square brackets, specificy which of the postsynaptic neurons you want included on the plot
            x_zoom >>> 1x2 matrix in [ ] to zoom into a specific section of the plot. Defined by seconds. By default, it will show the entire plot

        For example, to plot postsynaptic neurons 4 and 6, set whichNeurons = [4, 6]
        '''
        plt.figure(figsize=[12, 7])
        for i in range(len(whichNeurons)):
            freq, loc = getFreq(data=c.stdp_data[whichNeurons[i],:], dt=c.sns_dt)
            plt.plot(loc/1000, freq, label=('Interneuron ' + str(whichNeurons[i])))
        plt.ylim()
        if x_zoom is not None:
            plt.xlim(x_zoom)
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title('Spike Frequency of Postsynaptic Interneurons')
        plt.legend(loc='upper right')

        plt.savefig('./results/postFreq_sample.png', dpi=self.dpi)

    def plot_conductance(self, x_zoom: Optional[Tuple[float, float]]=None, conductance_timestep: Optional[int]=10):
        '''Plots the change conductance to each postsynaptic neuron on a separate plot. This is the primary plot used to evaluate STDP's effect on a network.
        
        Args:
            x_zoom >>> 1x2 matrix in [ ] to zoom into a specific section of the plot. Defined by seconds. By default, it will show the entire plot
            conductance_timestep >>> Timestep for plotting the conductance data. Rather than plotting for every timestep (slow), this allows for accurate plotting at a higher speed. Default is set to plot every 10 timesteps
        '''
        plt.figure(figsize=[12, c.STDP_POST_NUM * 3])

        for post in range(c.STDP_POST_NUM):
            plt.subplot(c.STDP_POST_NUM, 1, post + 1)
            plt.title('↓ Postsynaptic Interneuron ' + str(post + 1) + ' ↓')
            for pre in range(c.STDP_PRE_NUM):
                plt.plot(c.mj_t[::conductance_timestep], c.g_track[::conductance_timestep, post, pre], label=('Sensory Neuron ' + str(pre + 1)))
            plt.ylim([0, c.max_condutance+1])
            if x_zoom is not None:
                plt.xlim(x_zoom)
            plt.ylabel('Conductance (uS)')
            if post + 1 == c.STDP_POST_NUM:
                plt.xlabel('Time (s)')
            plt.legend(loc='lower right')
        plt.subplots_adjust(hspace=0.5)

        plt.savefig('./results/conductance.png', dpi=self.dpi)

    def plot_mnActivity(self, x_zoom: Optional[Tuple[float, float]]=None):
        '''Plots the spiking activity of the motor neurons.
        
        Args:
            x_zoom >>> 1x2 matrix in [ ] to zoom into a specific section of the plot. Defined by seconds. By default, it will show the entire plot
        '''
        # Plot motor neuron data if needed
        plt.figure(figsize=[10, 4])
        plt.subplot(2, 1, 1)

        for mn in range(c.MJ_MUSCLE_NUM):
            plt.plot(c.mj_t, c.mn_data[mn] * (0.1*(10-mn)), label=('Motor Neuron ' + str(mn + 1)), linewidth=0.8)
        plt.title('Muscle Activation Data')
        plt.ylabel('Spikes')
        plt.yticks(ticks=[])
        plt.xlim([-c.mj_tmax*0.04, c.mj_tmax*1.04])
        if x_zoom is not None:
            plt.xlim(x_zoom)
        plt.legend(loc='upper right')

        plt.subplot(2, 1, 2)
        for mn in range(c.MJ_MUSCLE_NUM):
            freq, loc = getFreq(data=c.mn_data[mn,:], dt=c.sns_dt)
            freq = np.asarray(freq)
            loc = np.asarray(loc)
            # Create logic mask of True/False for the different values in the array
            mask = freq >= 10
            # Slice the two arrays, cutting out both freq and loc values
            freq = freq[mask]
            loc = loc[mask]
            # Plot this motor neuron's data
            plt.plot(loc/1000, freq, '.', label=('Motor Neuron ' + str(mn + 1)), markersize=5)
        plt.ylim(0, 60)
        plt.ylabel('Frequency (Hz)')
        plt.xlim([-c.mj_tmax*0.04, c.mj_tmax*1.04])
        if x_zoom is not None:
            plt.xlim(x_zoom)
        plt.xlabel('Time (s)')
        plt.title('Spike Frequency of Motor Neurons')
        plt.legend(loc='upper right')
        plt.subplots_adjust(hspace=0.6)

        plt.savefig('./results/mnActivity.png', dpi=self.dpi)

    def plot_postFreqIaCurr_sample(self, whichPost: Tuple[int, ...], x_zoom: Optional[Tuple[float, float]]=None):
        '''Plots the spike frequency of a sample of postsynaptic neurons alongside the Ia activation current being sent to the presynaptic Ia neurons. This allows for a visual comparison between presynaptic activation and postsynaptic response.
        
        Args:
            whichPost >>> Allows for the selection of which postsynaptic neurons are included on the plot. Specify by placing desired neurons in square brackets
            x_zoom >>> 1x2 matrix in [ ] to zoom into a specific section of the plot. Defined by seconds. By default, it will show the entire plot
        '''


        plt.figure(figsize=[12, 5])
        plt.subplot(2, 1, 1)
        for i in range(c.MJ_MUSCLE_NUM):
            if i % 2 == 0:
                plt.plot(c.mj_t, c.stdp_activation_current[i], label=('Ia Sensory Neuron ' + str(i + 1)), linewidth=3.5, alpha=0.7)
            else:
                plt.plot(c.mj_t, c.stdp_activation_current[i], label=('Ia Sensory Neuron ' + str(i + 1)), linewidth=0.75, alpha=1.0)
        plt.title('Ia Sensory Neuron Activation Current')
        plt.legend(loc='upper right')
        plt.ylabel('Curernt (nA)')
        if x_zoom is not None:
            plt.xlim(x_zoom)
        plt.subplots_adjust(hspace=0.5)

        plt.subplot(2, 1, 2)
        for i in range(len(whichPost)):
            freq, loc = getFreq(data=c.stdp_data[whichPost[i],:], dt=c.sns_dt)
            plt.plot(loc/1000, freq, label=('Interneuron ' + str(whichPost[i])))
        plt.ylim()
        if x_zoom is not None:
            plt.xlim(x_zoom)
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title('Spike Frequency of Postsynaptic Interneurons')
        plt.legend(loc='upper right')

        plt.savefig('./results/postFreqIaCurr_sample.png', dpi=self.dpi)

    def plot_postFreqIaCurr(self, x_zoom: Optional[Tuple[float, float]]=None):
        '''Plots the spike frequency of all the postsynaptic neurons alongside the Ia activation current being sent to the presynaptic Ia neurons. This allows for a visual comparison between presynaptic activation and postsynaptic response.
        
        Args:
            x_zoom >>> 1x2 matrix in [ ] to zoom into a specific section of the plot. Defined by seconds. By default, it will show the entire plot
        '''
        fig = plt.figure(figsize=[12, 5])
        gs = GridSpec(nrows=3, ncols=1, figure=fig)

        plt1 = fig.add_subplot(gs[0, 0])
        plt2 = fig.add_subplot(gs[1:, 0])
        plt.subplots_adjust(hspace=0.5)

        for i in range(c.MJ_MUSCLE_NUM):
            if i % 2 == 0:
                plt1.plot(c.mj_t, c.stdp_activation_current[i], label=('Ia Sensory Neuron ' + str(i + 1)), linewidth=3.5, alpha=0.7)
            else:
                plt1.plot(c.mj_t, c.stdp_activation_current[i], label=('Ia Sensory Neuron ' + str(i + 1)), linewidth=0.75, alpha=1.0)
        plt1.set_title('Ia Sensory Neuron Activation Current')
        plt1.legend(loc='upper right')
        plt1.set_ylabel('Curernt (nA)')
        # plt.xlabel('Time (s)')
        if x_zoom is not None:
            plt1.set_xlim(x_zoom)

        for i in range(c.STDP_POST_NUM):
            freq, loc = getFreq(data=c.stdp_data[i,:], dt=c.sns_dt)
            # print(np.mean(freq))
            plt2.plot(loc/1000, freq, label=('Interneuron ' + str(i + 1)))
        plt2.set_ylim()
        if x_zoom is not None:
            plt2.set_xlim(x_zoom)
        plt2.set_ylabel('Frequency (Hz)')
        plt2.set_xlabel('Time (s)')
        plt2.set_title('Spike Frequency of Postsynaptic Interneurons')
        plt2.legend()

        plt.savefig('./results/postFreqIaCurr.png', dpi=self.dpi)

    def plot_essentials(self, x_zoom: Optional[Tuple[float, float]]=None):
        ''' Plot the essential plots!

        Args:
            x_zoom >>> 1x2 matrix in [ ] to zoom into a specific section of the plot. Defined by seconds. By default, it will show the entire plot

        '''
        self.plot_mj(x_zoom)
        self.plot_postFreq(x_zoom)
        self.plot_conductance(x_zoom)
        self.plot_mnActivity(x_zoom)
        

"""
def plot_mj(x_zoom=None, dpi=600):
    # Length
    plt.figure(figsize=[12, 6])
    plt.subplot(3, 1, 1)
    for i in range(c.MJ_MUSCLE_NUM):
        if i % 2 == 0:
            plt.plot(c.mj_t, 1000*c.mj_length_data[i], label=('Muscle ' + str(i + 1)), linewidth=3.5, alpha=0.7)
        else:
            plt.plot(c.mj_t, 1000*c.mj_length_data[i], label=('Muscle ' + str(i + 1)), linewidth=0.75)
    plt.title(' MuJoCo Muscle Length Data')
    plt.ylabel('Length (mm)')
    if x_zoom is not None:
        plt.xlim(x_zoom)
    plt.legend(loc='upper right')

    # Velocity
    plt.subplot(3, 1, 2)
    for i in range(c.MJ_MUSCLE_NUM):
        if i % 2 == 0:
            plt.plot(c.mj_t, 1000*c.mj_velocity_data[i], label=('Muscle ' + str(i + 1)), linewidth=3.5, alpha=0.7)
        else:
            plt.plot(c.mj_t, 1000*c.mj_velocity_data[i], label=('Muscle ' + str(i + 1)), linewidth=0.75, alpha=1.0)
    plt.title('MuJoCo Muscle Velocity Data')
    plt.ylabel('Velocity (mm/s)')
    if x_zoom is not None:
        plt.xlim(x_zoom)
    plt.legend(loc='upper right')

    # Current into STDP SNS
    plt.subplot(3, 1, 3)
    for i in range(c.MJ_MUSCLE_NUM):
        if i % 2 == 0:
            plt.plot(c.mj_t, c.stdp_activation_current[i], label=('Ia Sensory Neuron ' + str(i + 1)), linewidth=3.5, alpha=0.7)
        else:
            plt.plot(c.mj_t, c.stdp_activation_current[i], label=('Ia Sensory Neuron ' + str(i + 1)), linewidth=0.75, alpha=1.0)
    plt.title('Ia Sensory Neuron Activation Current')
    plt.legend(loc='upper right')
    plt.ylabel('Curernt (nA)')
    plt.xlabel('Time (s)')
    if x_zoom is not None:
        plt.xlim(x_zoom)
    plt.subplots_adjust(hspace=0.5)

    plt.savefig('./results/mjdata.png', dpi=dpi)
    
def plot_postSpikes(x_zoom=None, dpi=600):
    plt.figure(figsize=[25, 5])
    for i in range(c.STDP_POST_NUM):
        plt.plot(c.mj_t, c.stdp_data[i] * (1 - 0.05*i), linewidth='1', label=('Post: ' + str(i + 1)))
    if x_zoom is not None:
        plt.xlim(x_zoom)
    plt.ylabel('Spike')
    plt.xlabel('Time (s)')
    plt.title('Postsynaptic Spikes')
    plt.legend(loc='upper right')

    plt.savefig('./results/postSpikes.png', dpi=dpi)

def plot_postFreq(x_zoom=None, dpi=600):
    plt.figure(figsize=[12, 7])
    for i in range(c.STDP_POST_NUM):
        freq, loc = getFreq(data=c.stdp_data[i,:], dt=c.sns_dt)
        # print(np.mean(freq))
        plt.plot(loc/1000, freq, label=('Interneuron ' + str(i + 1)))
    plt.ylim()
    if x_zoom is not None:
        plt.xlim(x_zoom)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Spike Frequency of Postsynaptic Interneurons')
    plt.legend(loc='upper right')

    plt.savefig('./results/postFreq.png', dpi=dpi)

def plot_postFreq_sample(whichNeurons, x_zoom=None, dpi=600):
    plt.figure(figsize=[12, 7])
    for i in range(len(whichNeurons)):
        freq, loc = getFreq(data=c.stdp_data[whichNeurons[i],:], dt=c.sns_dt)
        plt.plot(loc/1000, freq, label=('Interneuron ' + str(whichNeurons[i])))
    plt.ylim()
    if x_zoom is not None:
        plt.xlim(x_zoom)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Spike Frequency of Postsynaptic Interneurons')
    plt.legend(loc='upper right')

    plt.savefig('./results/postFreq_sample.png', dpi=dpi)

def plot_conductance(conductance_timestep=10, x_zoom=None, dpi=600):
    plt.figure(figsize=[12, c.STDP_POST_NUM * 3])

    for post in range(c.STDP_POST_NUM):
        plt.subplot(c.STDP_POST_NUM, 1, post + 1)
        plt.title('↓ Postsynaptic Interneuron ' + str(post + 1) + ' ↓')
        for pre in range(c.STDP_PRE_NUM):
            plt.plot(c.mj_t[::conductance_timestep], c.g_track[::conductance_timestep, post, pre], label=('Sensory Neuron ' + str(pre + 1)))
        plt.ylim([0, c.max_condutance+1])
        if x_zoom is not None:
            plt.xlim(x_zoom)
        plt.ylabel('Conductance (uS)')
        if post + 1 == c.STDP_POST_NUM:
            plt.xlabel('Time (s)')
        plt.legend(loc='lower right')
    plt.subplots_adjust(hspace=0.5)

    plt.savefig('./results/conductance.png', dpi=dpi)

def plot_mnActivity(x_zoom=None, dpi=600):
    # Plot motor neuron data if needed
    plt.figure(figsize=[10, 4])
    plt.subplot(2, 1, 1)

    for mn in range(c.MJ_MUSCLE_NUM):
        plt.plot(c.mj_t, c.mn_data[mn] * (0.1*(10-mn)), label=('Motor Neuron ' + str(mn + 1)), linewidth=0.8)
    plt.title('Muscle Activation Data')
    plt.ylabel('Spikes')
    plt.yticks(ticks=[])
    plt.xlim([-c.mj_tmax*0.04, c.mj_tmax*1.04])
    if x_zoom is not None:
        plt.xlim(x_zoom)
    plt.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    for mn in range(c.MJ_MUSCLE_NUM):
        freq, loc = getFreq(data=c.mn_data[mn,:], dt=c.sns_dt)
        freq = np.asarray(freq)
        loc = np.asarray(loc)
        # Create logic mask of True/False for the different values in the array
        mask = freq >= 10
        # Slice the two arrays, cutting out both freq and loc values
        freq = freq[mask]
        loc = loc[mask]
        # Plot this motor neuron's data
        plt.plot(loc/1000, freq, '.', label=('Motor Neuron ' + str(mn + 1)), markersize=5)
    plt.ylim(0, 60)
    plt.ylabel('Frequency (Hz)')
    plt.xlim([-c.mj_tmax*0.04, c.mj_tmax*1.04])
    if x_zoom is not None:
        plt.xlim(x_zoom)
    plt.xlabel('Time (s)')
    plt.title('Spike Frequency of Motor Neurons')
    plt.legend(loc='upper right')
    plt.subplots_adjust(hspace=0.6)

    plt.savefig('./results/mnActivity.png', dpi=dpi)

def plot_postFreqIaCurr_sample(whichPost, x_zoom=None, dpi=600):
    plt.figure(figsize=[12, 5])
    plt.subplot(2, 1, 1)
    for i in range(c.MJ_MUSCLE_NUM):
        if i % 2 == 0:
            plt.plot(c.mj_t, c.stdp_activation_current[i], label=('Ia Sensory Neuron ' + str(i + 1)), linewidth=3.5, alpha=0.7)
        else:
            plt.plot(c.mj_t, c.stdp_activation_current[i], label=('Ia Sensory Neuron ' + str(i + 1)), linewidth=0.75, alpha=1.0)
    plt.title('Ia Sensory Neuron Activation Current')
    plt.legend(loc='upper right')
    plt.ylabel('Curernt (nA)')
    if x_zoom is not None:
        plt.xlim(x_zoom)
    plt.subplots_adjust(hspace=0.5)

    plt.subplot(2, 1, 2)
    for i in range(len(whichPost)):
        freq, loc = getFreq(data=c.stdp_data[whichPost[i],:], dt=c.sns_dt)
        plt.plot(loc/1000, freq, label=('Interneuron ' + str(whichPost[i])))
    plt.ylim()
    if x_zoom is not None:
        plt.xlim(x_zoom)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Spike Frequency of Postsynaptic Interneurons')
    plt.legend(loc='upper right')

    plt.savefig('./results/postFreqIaCurr_sample.png', dpi=dpi)

def plot_postFreqIaCurr(x_zoom=None, dpi=600):
    fig = plt.figure(figsize=[12, 5])
    gs = GridSpec(nrows=3, ncols=1, figure=fig)

    plt1 = fig.add_subplot(gs[0, 0])
    plt2 = fig.add_subplot(gs[1:, 0])
    plt.subplots_adjust(hspace=0.5)

    for i in range(c.MJ_MUSCLE_NUM):
        if i % 2 == 0:
            plt1.plot(c.mj_t, c.stdp_activation_current[i], label=('Ia Sensory Neuron ' + str(i + 1)), linewidth=3.5, alpha=0.7)
        else:
            plt1.plot(c.mj_t, c.stdp_activation_current[i], label=('Ia Sensory Neuron ' + str(i + 1)), linewidth=0.75, alpha=1.0)
    plt1.set_title('Ia Sensory Neuron Activation Current')
    plt1.legend(loc='upper right')
    plt1.set_ylabel('Curernt (nA)')
    # plt.xlabel('Time (s)')
    if x_zoom is not None:
        plt1.set_xlim(x_zoom)

    for i in range(c.STDP_POST_NUM):
        freq, loc = getFreq(data=c.stdp_data[i,:], dt=c.sns_dt)
        # print(np.mean(freq))
        plt2.plot(loc/1000, freq, label=('Interneuron ' + str(i + 1)))
    plt2.set_ylim()
    if x_zoom is not None:
        plt2.set_xlim(x_zoom)
    plt2.set_ylabel('Frequency (Hz)')
    plt2.set_xlabel('Time (s)')
    plt2.set_title('Spike Frequency of Postsynaptic Interneurons')
    plt2.legend()

    plt.savefig('./results/postFreqIaCurr.png', dpi=dpi)

def plot_essentials():
    ''' Plot the Essentials

    Args:
        x_zoom([1, 2]) Matrix. Zooms into a specific section of the plot

    '''
    plot_mj()
    plot_postFreq()
    plot_conductance()
    plot_mnActivity()
    
"""
''' This code contains the simulation loop '''
import numpy as np
from typing import Optional
import mujoco as mj
import projectCode.constants as c
from .movementToCurrent import vel2cur
from .calculations import NoisyAmps
from .mujocoParameters import mj_camera
from mediapy import write_video
from .muscleActivation import MuscleAct

class Sim():
    '''Simulation Time!
    '''
    def runSimulation(self, mjmodel, mjdata, simulationTime: Optional[int] = None, captureVideo: Optional[bool] = True):
        '''Activates the simulation and runs through all the time steps.
        
        Args:
            mjmodel >>> The desired MuJoCo model
            mjdata >>> The data that accompanies the MuJoCo model
            simulationTime >>> While the simulation time is stored in the "constants.py" file, this is an optional location to adjust it. Numbers above 99 are assumed to be in ms, numbers below are assumed to be in seconds.
            captureVideo >>> Enables the rendering and movie creation of the MuJoCo model during the simulation
            '''
        if simulationTime is not None:
            if simulationTime < 44:
                # Assume its in seconds
                simulationTime = simulationTime * 1000
            # Reload all variables that depend on the max time
            c.sns_tmax = simulationTime
            c.sns_t = np.arange(0, c.sns_tmax, c.sns_dt)
            c.mj_tmax =  c.sns_tmax / 1000
            c.mj_t = np.arange(0, c.mj_tmax, c.mj_dt)
            c.stdp_activation_current = np.zeros(shape=(len(c.sns_t), c.STDP_PRE_NUM))
            c.stdp_data = np.zeros([len(c.sns_t), c.STDP_POST_NUM + 2])
            c.mn_activation_current = np.zeros(shape=(len(c.sns_t), c.MOTOR_NEURON_NUM))
            c.mn_data = np.zeros(shape=(len(c.sns_t), c.MJ_MUSCLE_NUM))
            c.mn_activation = MuscleAct().randActivation(c.sns_t, 2000, 500, num_motors=c.MJ_MUSCLE_NUM)
            c.mn_activation_current[:, 0] = c.mn_activation[:, 0] * c.activation_level
            c.mn_activation_current[:, 1] = c.mn_activation[:, 1] * c.activation_level
            c.mn_activation_current[:, 2] = c.mn_activation[:, 2] * c.activation_level
            c.mn_activation_current[:, 3] = c.mn_activation[:, 3] * c.activation_level
            c.mn_activation_current = NoisyAmps(c.mn_activation_current, 5)
            c.g_track = np.zeros(shape=[len(c.sns_t), c.STDP_POST_NUM, c.STDP_PRE_NUM])
            c.mj_length_data = np.zeros(shape=[len(c.mj_t), c.MJ_MUSCLE_NUM])
            c.mj_velocity_data = np.zeros(shape=[len(c.mj_t), c.MJ_MUSCLE_NUM])

        # Parameter info
        print('SIMULATION PARAMETERS')
        if simulationTime is not None:
            print('USING UPDATED SIMULATION TIME: ' + str(simulationTime/1000) + 's')
        else:
            print('USING DEFAULT SIMULATION TIME FROM CONSTANTS.PY: ' + str(c.sns_tmax/1000) + 's')

        # Reset Simulation
        mj.mj_resetData(mjmodel, mjdata)
        # Restart Renderer
        renderer = mj.Renderer(mjmodel)

        for i in range(len(c.sns_t)):
            ''' Motor Neuron SNS '''
            c.mn_data[i,:] = c.sns_mn_network(c.mn_activation_current[i,:])

            ''' Spiking Motor Neuron === Muscle Activation '''
            if sum(c.sns_mn_network.__dict__.get('spikes')) != 0:
                mn_spike = c.sns_mn_network.__dict__.get('spikes')

                # If a spike occured for a specific motor, activate it for a timestep
                for muscle in range(c.MJ_MUSCLE_NUM):
                    if mn_spike[muscle] != 0:
                        mjdata.act[muscle] = c.MAX_MUSCLE_POWER
                    else:
                        mjdata.act[muscle] = 0.0
            else:
                mjdata.act[:] = 0.0

            ''' Advance MuJoCo Simulation '''
            mj.mj_step(mjmodel, mjdata)

            if captureVideo == True:
                # Capture frame data if it corresponds to framerate demands
                if len(c.frames) < mjdata.time*c.framerate:
                    renderer.update_scene(mjdata, camera=mj_camera[0])
                    pixels = renderer.render().copy()
                    c.frames.append(pixels)
            

            ''' Record MuJoCo Sensor Outputs '''
            c.mj_length_data[i] = mjdata.sensordata[0:c.MJ_MUSCLE_NUM]
            c.mj_velocity_data[i] = mjdata.sensordata[c.MJ_MUSCLE_NUM:]

            # Subtract resting length to get displacement. Not sure if this is the right way?
            c.mj_length_data[i] = c.mj_length_data[i] - c.mj_length_resting
            
            # Convert Ia feedback (length & velocity) to current input into neuron
            c.stdp_activation_current[i] = vel2cur(length=c.mj_length_data[i], velocity=c.mj_velocity_data[i], current_time=c.sns_t[i])

            ''' STDP SNS '''
            # At the first call, update the conductance matrix. Afterwards, do not
            if i == 0:
                c.stdp_data[i, :] = c.sns_stdp_network(c.stdp_activation_current[i, :], current_time=c.sns_t[i], dt=c.sns_dt, g_update=c.RANDOMIZED_CONDUCTIVITY)
            else:
                c.stdp_data[i, :] = c.sns_stdp_network(c.stdp_activation_current[i, :], current_time=c.sns_t[i], dt=c.sns_dt)

            # Record conductance values to plot
            c.g_track[i] = c.sns_stdp_network.g_max_spike[c.STDP_PRE_NUM:, 0:c.STDP_PRE_NUM]

        # Fix data orientation for better plotting
        c.mn_data = c.mn_data.transpose()
        c.mj_length_data = c.mj_length_data.transpose()
        c.mj_velocity_data = c.mj_velocity_data.transpose()
        c.stdp_activation_current = c.stdp_activation_current.transpose()
        c.stdp_data = c.stdp_data.transpose()

        if captureVideo == True:
            # Output file name
            output_name = './results/LegSimulationVideo.mp4'
            # Write frames to video
            write_video(output_name, images=c.frames, fps=c.framerate, )

        c.sns_stdp_network.reset()
        renderer.close()
''' Standard Imports '''
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

''' SNS Imports '''
# from NetworkGeneration import STDPNetworkGenerator
# from NetworkGeneration_MN import MNNetworkGenerator
from sns_toolbox.renderer import render
from sns_toolbox.neurons import SpikingNeuron
from sns_toolbox.connections import SpikingSynapse

''' Custom imports '''
import custom_methods

''' MuJoCo Imports '''
import mujoco as mj
import mediapy as media

''' Other Misc Imports '''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from random import Random, randint

''' Make the plots pretty and slooooooow '''
mpl.rcParams['figure.dpi'] = 600 # 600 is pretty high definition

''' Load MuJoCo File '''
# Load model
mjmodel = mj.MjModel.from_xml_path('resources/MJ_Leg.xml')
mjdata = mj.MjData(mjmodel)
# Create MuJoCo renderer
renderer = mj.Renderer(mjmodel)

# Camera options for the fun of it
mj_camera = ['fixed', 'Angle2', 'Angle3', 'Angle4', 'Angle5']
active_camera = 5

# Show image of XML file for reference
mj.mj_forward(mjmodel, mjdata)
renderer.update_scene(mjdata, camera=mj_camera[0])
image = renderer.render()
# media.show_image(renderer.render())
media.write_image('results/combo_creation_velocity_ia.png', image)
renderer.close() # Needed to prevent crashing?

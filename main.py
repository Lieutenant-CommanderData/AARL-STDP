''' Main Code to Run Simulation '''
# Imports
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import projectCode.constants as const
import projectCode.plotGenerator as plot
from time import sleep

from projectCode.plotGenerator import Plot
from projectCode.mujocoParameters import MuJoCoExtras
from projectCode.simulation import Sim


def main():
    plot = Plot(dpi=100)
    mje = MuJoCoExtras()
    s = Sim()

    # Creates png of MuJoCo model
    mje.showModel(mjmodel=const.mjmodel, mjdata=const.mjdata, camera=0)
    
    # Runs the simulation
    s.runSimulation(mjmodel=const.mjmodel, mjdata=const.mjdata, captureVideo=False)
    print('Simulation Complete')
    print('Plotting Data')

    # Creates and saves the essential plots for a simulation!
    plot.plot_all()
    plot.plot_postVoltsSpikes([0, 0.1])
    # plot.plot_conductance([0, 0.1])
    print('Plotting Complete')


if __name__ == '__main__':
    main()

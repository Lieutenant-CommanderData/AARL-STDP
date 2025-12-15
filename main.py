''' Main Code to Run Simulation '''
# Imports
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import projectCode.constants as const
import projectCode.plotGenerator as plot

from projectCode.plotGenerator import Plot
from projectCode.mujocoParameters import MuJoCoExtras
from projectCode.simulation import Sim

def main():
    plot = Plot()
    mje = MuJoCoExtras()
    s = Sim()


    # Creates png of MuJoCo model
    mje.showModel(mjmodel=const.mjmodel, mjdata=const.mjdata, camera=0)
    
    # Runs the simulation
    s.runSimulation(mjmodel=const.mjmodel, mjdata=const.mjdata, simulationTime=10, captureVideo=True)

    # Creates and saves the essential plots for a simulation!
    plot.plot_essentials()

if __name__ == '__main__':
    main()
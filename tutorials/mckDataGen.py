#!/usr/bin/python3

import numpy as np

import random
from datetime import datetime

from numpy.lib.index_tricks import c_
random.seed(datetime.now())

#################################################
def generate_simulation_input(nSims, m, mB, c, cB, k, kB, rRange, uRange, FRange):
    simInput = np.zeros((nSims,6))
    """ Get what Kennedy & O'Hagan call t and x.
    t - the value of the parameters that we want to calibrate
    x - the input parameters at which simulation is carried out
    :simInput[:,0]: the mass with which the simulation is carried out
    :simInput[:,1]: the damping with which the simulation is carried out
    :simInput[:,2]: the spring stiffness with which the simulation is carried out
    :simInput[:,3]: the value of the spring deformation used in the "simulation"
    :simInput[:,4]: the value of the body velocity used in the "simulation"
    :simInput[:,5]: the force experienced by the body, and which is used in the "simulation"
    """
    for i in range(nSims):
        simInput[i][0] = m*(1.-random.uniform(-mB,mB)) # I hope this doesn't hit zero :-)
        simInput[i][1] = c*(1.-random.uniform(-cB,cB))
        simInput[i][2] = k*(1.-random.uniform(-kB,kB))
        simInput[i][3] = random.uniform(-rRange,rRange)
        simInput[i][4] = random.uniform(-uRange,uRange)
        simInput[i][5] = random.uniform(-FRange,FRange)

    return simInput


#################################################
def run_simulations(simInput):
    """ This is data that is produced by the simulator 
    :simInput[:,0]: the mass with which the simulation is carried out
    :simInput[:,1]: the damping with which the simulation is carried out
    :simInput[:,2]: the spring stiffness with which the simulation is carried out
    :simInput[:,3]: the value of the spring deformation used in the "simulation"
    :simInput[:,4]: the value of the body velocity used in the "simulation"
    :simInput[:,5]: the force experienced by the body, and which is used in the "simulation"

    In this function call, we perform a bunch of simulations - as many as rows in simInput
    """
    one_over_m = 1.0/simInput[:,0]
    c_over_m = simInput[:,1]/simInput[:,0]
    k_over_m = simInput[:,2]/simInput[:,0]
    acc = -k_over_m * simInput[: , 3] - c_over_m*simInput[: , 4] + one_over_m*simInput[: , 5]
    return acc

################################################
# A single-run version of spring--damper model
def evalModel(m, c, k, r, u, F):
    one_over_m = 1.0/m
    c_over_m = c/m
    k_over_m = k/m
    acc = -k_over_m * r - c_over_m * u + one_over_m * F
    return acc

#################################################
def mckDataGen(nSims, m=5.0, c=5.0, k=125.0, mB=0.1, cB=0.1, kB=0.1, rRange=1., uRange=7., FRange=8.):
    # mass, damping, stiffness. This is what i want to "find" at the end of calibration
    # get acceleration for a given set of m, c, k and r, u, F
    simulationInput = generate_simulation_input(nSims, m, mB, c, cB, k, kB, rRange, uRange, FRange)
    accelSimulation = run_simulations(simulationInput)
    return np.concatenate((simulationInput, accelSimulation.reshape((-1,1))), axis=1)


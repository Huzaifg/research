#!/usr/bin/python3

import numpy as np
import mckDataGen as DG
from pymcmcstat.MCMC import MCMC
from pymcmcstat.settings.DataStructure import DataStructure
import matplotlib.pyplot as plt


def main():
	#Generate the data using Rouchun script

	ndata = 1000 # number of data samples
	# This will give data of the form [m,c,k,r,u,F,a] = [mass,damping,stiffness,length,velocity,force,acceleration]
	# The true values being used can be seen in mckDataGen.py - [5,5,125]
	data = DG.mckDataGen(ndata)

	#initialize the mcmc object
	mcstat = MCMC()
	mcstat.data.add_data_set(x = data[:,:-1], y = data[:,-1].flatten())

	#Intitial parameter values
	q_init = {'m' : 1.,'c' : 1.,'k' : 1.}
	# Add the calibration parameters - Although there are 6 input parameters, we only calibrate the m,c and k. The r,u and F are the simulation parameters.

	mcstat.parameters.add_model_parameter(name = 'm', theta0 = q_init['m'] , minimum = 0.,sample = True) # Adding a minimum as we have that information about our prior
	mcstat.parameters.add_model_parameter(name = 'c', theta0 = q_init['c'] , minimum = 0.,sample = True) # Adding a minimum as we have that information about our prior
	mcstat.parameters.add_model_parameter(name = 'k', theta0 = q_init['k'], minimum = 0.,sample = True) # Adding a minimum as we have that information about our prior

	#Define the model settings and provide the sos function - using the default uniform distribution prior

	mcstat.model_settings.define_model_settings(sos_function=ssfun)

	#Define simualtion settings - I use 10e4 number of simulations with the dram algorithm. I set update sigma to true to add the measurement error vairance as a parameter
	
	mcstat.simulation_options.define_simulation_options(nsimu = 10.0e4, method = 'dram',updatesigma=True)

	# Now run the simulation
	
	mcstat.run_simulation()

	#extract results
	results = mcstat.simulation_results.results
	chain = results['chain']
	s2chain = results['s2chain']
	#Lets do some burnin - burnin first 1/10th of the chain
	burnin = int(chain.shape[0]/10)
	bur_chain = chain[burnin:,:]
	bur_s2chain = s2chain[burnin:,:]
	mcstat.chainstats(bur_chain,results)

	# Plotting some results
	# The chain
	chain_plot = mcstat.mcmcplot.plot_chain_panel(bur_chain,names = ['m','c','k'],figsizeinches = [8,8])
	chain_plot.savefig('parameter_chain.png')
	#The posterior probabilities
	density_plot = mcstat.mcmcplot.plot_density_panel(bur_chain,names = ['m','c','k'],figsizeinches = [8,8])
	density_plot.savefig('posterior.png')
	p3 = mcstat.mcmcplot.plot_density_panel(bur_s2chain,names = ['m','c','k'],figsizeinches = [8,8])
	p3.savefig('posterior_s2.png')



#Define the Sum of Squares function
def ssfun(q,data):
	params = {'m' : q[0] , 'c' : q[1] , 'k' : q[2]} #Unpack the parameters into a dict
	#xdata is an array of 
	r = data.xdata[0][:,3] # I think 0 xdata[0] refers to the number of the data set added ... cant seem to understand whey you would need more than 1 dataset 
	u = data.xdata[0][:,4]
	F = data.xdata[0][:,5]

	acc_model = DG.evalModel(params['m'], params['c'], params['k'], r, u, F)
	res = data.ydata[0] - acc_model.reshape(-1,1)
	ss = (res ** 2).sum(axis = 0) 
	return ss


if __name__ == "__main__":
    main()

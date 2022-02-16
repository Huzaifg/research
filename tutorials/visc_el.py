import numpy as np
import scipy.io as sio
from time import time as timetest
import ctypes
from numpy.ctypeslib import ndpointer
from pymcmcstat.MCMC import MCMC
from pymcmcstat.settings.DataStructure import DataStructure
import matplotlib.pyplot as plt
import os
import pandas as pd

#Defining some font families to use for all the plots
font_title = {'family':'sans-serif',
		'color' : 'black',
		'weight':'bold',
		'size' : 16,
		}

font_label = {'family':'sans-serif',
		'color' : 'black',
		'weight':'normal',
		'size' : 14,
		}

#Function to plot any data
def data_plot(stretch,stress,pred=False,model_stress=None):
	plt.figure(figsize = (8,8))
	if(pred == False):
		plt.plot(stretch,stress,'k')
		plt.title("Data at strain rate 0.67",fontdict=font_title)
		plt.xlabel('stretch',fontdict=font_label)
		plt.ylabel('stress (kPa)',fontdict=font_label)
		path = 'images/'
		if(os.path.isdir(path)):
			plt.savefig(path + 'visc_data.png')
		else:
			os.mkdir(path)
			plt.savefig(path + 'visc_data.png')
	else:
		plt.plot(stretch,stress,'k',label = 'Data')
		plt.plot(stretch,model_stress,'r',label = 'Prediction')
		plt.title("Prediction using means at strain rate 0.67",fontdict=font_title)
		plt.xlabel('stretch',fontdict=font_label)
		plt.ylabel('stress (kPa)',fontdict=font_label)
		plt.legend()
		path = 'images/'
		if(os.path.isdir(path)):
			plt.savefig(path + 'visc_pred.png')
		else:
			os.mkdir(path)
			plt.savefig(path + 'visc_pred.png')
		
	


	
#Defining the "Nonaffine Hyperelastic" model - Do not undersrand the physics of this model very well - Copied from code in the paper
#The parameters of this model are [G_c,G_e,lambda_max]
def nonaffine_hyperelastic_model(theta,stretch):
	Gc = theta['Gc']
	Ge = theta['Ge']
	lam_max = theta['lam_max']

	#stretch invariant I is defined as follows
	I1 = stretch**2 + 2/stretch

	#Hydrostatic pressure is defined as follows
	p = (Gc/3/stretch*((9*lam_max**2 - I1)/(3*lam_max**2 - I1))) + Ge/stretch**0.5*(1 - stretch)

	#1st P-K stress is defined as
	Hc = 1/3*Gc*stretch*((9*lam_max**2 - I1)/(3*lam_max**2 - I1))
	He = Ge*(1-1/stretch**2)
	#Thus stress is
	sigma_inf = Hc + He - p/stretch

	return sigma_inf.reshape(-1,1)

def pred_model(theta,time,stretch):

	#Exctract the parameters from the theta
	params = {'Gc' : theta[0], 'Ge' : theta[1] , 'lam_max' : theta[2], 'eta' : theta[3], 'gamma' : theta[4]}

	#Evaluate the hyperelastic and viscoelastic stresses
	hyp_s = nonaffine_hyperelastic_model(params,stretch)
	visc_s = visc_mod(params['eta'],params['gamma'],stretch,time,len(time))

	#The actual stress is just the sum of these two stresses
	model_stress = hyp_s + visc_s
	return model_stress


# The sum-of-squares function to caluclate the model loss comapred to the data
def sos_fun(theta,data):
	#Extract the stress from the data
	stress = data.ydata[0]
	time = data.xdata[0][:,0]
	stretch = data.xdata[0][:,1]

	#The actual stress is just the sum of these two stresses
	model_stress = pred_model(theta,time,stretch)
	#Calculate the residual
	res = model_stress - stress
	#Use the numpy function for better speed
	ss = np.dot(res.T,res)
	return ss





################# Unfortunately cant put all this in the main function as I cant supply the ssfunction with the viscoelastic model made in c++ ##########


#First lets import the data and have a look
vhbdata = sio.loadmat('vhb4910_data.mat')
#This matlab file has xdata = [time,stretch] and ydata = [stress]. It is arranged as multiple dicts and is tedious to extract from
#The data is only for one paticular strain rate of 0.67
time = vhbdata['data']['xdata'][0][0][:,0]
stretch = vhbdata['data']['xdata'][0][0][:,1]
stress = vhbdata['data']['ydata'][0][0][:,0]
#number of timesteps
nds = len(time)

#Visualise the data in /images/visc_data.png
data_plot(stretch = stretch,stress = stress,pred=False)

#initialize the mcmc object
mcstat = MCMC()

#add data to the mcmc object - x is both our time column and our stretch column
mcstat.data.add_data_set(x = vhbdata['data']['xdata'][0][0], y = stress.reshape(-1,1))

#Define the initial parameter values
theta0 = {'Gc' : 7.5541, 'Ge' : 17.69, 'lam_max' : 4.8333, 'eta' : 708, 'gamma' : 31}
#This needs to be converted to a list to be compatible with pymcmcstat
theta0vec = list(theta0.values())

#Add all the calibration parameters - I will first try sampling all the parameters as I want to see the dependecy plots

mcstat.parameters.add_model_parameter(name = 'Gc',theta0 = theta0['Gc'],minimum = 0,sample = False)
mcstat.parameters.add_model_parameter(name = 'Ge',theta0 = theta0['Ge'],minimum = 0,sample = False)
mcstat.parameters.add_model_parameter(name = 'lam_max',theta0 = theta0['lam_max'],minimum = 0,sample = False)
mcstat.parameters.add_model_parameter(name = 'eta',theta0 = theta0['eta'],minimum = 0,sample = True)
mcstat.parameters.add_model_parameter(name = 'gamma',theta0 = theta0['gamma'],minimum = 0,sample = True)

#Bring in the model made in C++ usign ctypes - Just doing this to learn, there is not much difference in time as the model is small

#Load the dynamic pre compiled library
lib = ctypes.cdll.LoadLibrary('./visc_mod.so')
#Excract the function out of that library
visc_mod = lib.linear_viscoelastic_model
#State that the return type of the function is a pointer to double
visc_mod.restype = ndpointer(dtype = ctypes.c_double, shape=(nds,1))
#State the argument types to that function as a list
visc_mod.argtypes = [ctypes.c_double,ctypes.c_double,ndpointer(ctypes.c_double),ndpointer(ctypes.c_double),ctypes.c_int]

#Test run this function
# st = timetest()
# for i in range(100):
# 	q = visc_mod(theta0['Gc'],theta0['Ge'],stretch,time,nds)
# et = timetest()
# print(f'The time taken by the model is {(et-st)/1e5}')

#Define the model settings - where we give mcstat the sos function
mcstat.model_settings.define_model_settings(sos_function=sos_fun)

#Define the sumulation settings - Using Delayed Rejection Metropolis algorithm and also sampling the data noise
nsimu = 2e5
savedir='./savedir'
mcstat.simulation_options.define_simulation_options(
	nsimu = nsimu,
	chainfile ='run2',
	save_to_txt=True,
	updatesigma = True,
	savedir=savedir,
	savesize=int(nsimu/2))

# Now we run the simulation
mcstat.run_simulation()

#Store the results
results  = mcstat.simulation_results.results
names = results['names']
#Extract all the chains and the number of simulations run
fullchain = results['chain']
fulls2chain = results['s2chain']
nsimu = results['nsimu']

#Apply a burnin
burnin = int(nsimu/2)
chain = fullchain[burnin:,:]
s2chain = fulls2chain[burnin:,:]

#Print the chain stats
stats = mcstat.chainstats(chain,results,returnstats=True)

#Plot the chain
chain_plot = mcstat.mcmcplot.plot_chain_panel(chain,names = names,figsizeinches = [8,8])
chain_plot.savefig('images/chain_allp.png')

#Plot the posterior probability distribution
density_plot = mcstat.mcmcplot.plot_density_panel(chain,names = names,figsizeinches = [8,8])
density_plot.savefig('images/post_den_allp.png')

#Plot all the pairwise correlations
pair_plot = mcstat.mcmcplot.plot_pairwise_correlation_panel(chain,names,figsizeinches = [8,8])
pair_plot.savefig('images/pair_allp.png')

####################################################################################################################################################
########################## Additional Post Processing ##############################################################################################

#Sadly this only works with 5 thetas as of now :(
theta_means = stats['mean']
model_stress = pred_model(theta_means,time,stretch)
data_plot(stretch = stretch,stress = stress,model_stress = model_stress,pred=True)

## Also save the stats in a csv file
df = pd.DataFrame(list(zip(stats['mean'],stats['std'],stats['mcerr'],stats['geweke']['p'],stats['iact']['tau'])),
	columns = ['mean','std','mcerr','geweke','tau'],
	index = names)
df.to_csv('./savedir/results.csv')




######################## End of main function #########################################


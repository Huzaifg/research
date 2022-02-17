import numpy as np
import ctypes
import scipy.io as sio
from pymcmcstat.MCMC import MCMC
from numpy.ctypeslib import ndpointer
from pymcmcstat.ParallelMCMC import ParallelMCMC, load_parallel_simulation_results
from pymcmcstat.chain import ChainStatistics as CS
from pymcmcstat.chain import ChainProcessing as CP
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import mcmcplot.mcseaborn as mcsns


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
	free_mem(visc_s)
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



############# Unfortunately cannot define a main function as pred model needs visc_mod which is handled from the sos_fun which is handled by pymcmcstat###############

#First lets import the data and have a look
vhbdata = sio.loadmat('vhb4910_data.mat')
#This matlab file has xdata = [time,stretch] and ydata = [stress]. It is arranged as multiple dicts and is tedious to extract from
#The data is only for one paticular strain rate of 0.67
time = vhbdata['data']['xdata'][0][0][:,0]
stretch = vhbdata['data']['xdata'][0][0][:,1]
stress = vhbdata['data']['ydata'][0][0][:,0]
#number of timesteps
nds = len(time)

#Bring in the model made in C++ usign ctypes - Just doing this to learn, there is not much difference in time as the model is small

#Load the dynamic pre compiled library
lib = ctypes.cdll.LoadLibrary('./visc_mod.so')
#Excract the function out of that library
visc_mod = lib.linear_viscoelastic_model
#State that the return type of the function is a pointer to double
visc_mod.restype = ndpointer(dtype = ctypes.c_double, shape=(nds,1))
#State the argument types to that function as a list
visc_mod.argtypes = [ctypes.c_double,ctypes.c_double,ndpointer(ctypes.c_double),ndpointer(ctypes.c_double),ctypes.c_int]

#Our C function allocates memory on the heap which needs to be freed
free_mem = lib.free_mem
free_mem.restype = None 
free_mem.argtypes = [ndpointer(ctypes.c_double)]




#As we have three chains, we have to provide initial values to each of the three chains
theta0 = {'Gc' : 7.5541, 'Ge' : 17.69, 'lam_max' : 4.8333, 'eta' : 708, 'gamma' : 31}
#This needs to be converted to a list to be compatible with pymcmcstat
theta0vec = list(theta0.values())

theta1 = {'Gc' : 7.5541, 'Ge' : 17.69, 'lam_max' : 4.8333, 'eta' : 650, 'gamma' : 50}
theta1vec = list(theta1.values())

theta2 = {'Gc' : 7.5541, 'Ge' : 17.69, 'lam_max' : 4.8333, 'eta' : 750, 'gamma' : 22}
theta2vec = list(theta2.values())
initial_values = np.array([theta0vec,theta1vec,theta2vec])
#The folder to save the multiple chains in
date = datetime.now().strftime('%Y%m%d_%H%M%S')
savedir = str('{}_{}'.format(date,'pll_chains'))
no_of_chains = 3
for ii in range(no_of_chains):
	#Define mcmc object and add the data
	mcstat = MCMC()
	mcstat.data.add_data_set(x = vhbdata['data']['xdata'][0][0], y = stress.reshape(-1,1))

	#Define the model settings - where we give mcstat the sos function
	mcstat.model_settings.define_model_settings(sos_function=sos_fun)

	#Add all the calibration parameters 
	#Different initial parameters for each chain are specified - also tried to add a maximum upper limit - Only sampling eta and gamma
	mcstat.parameters.add_model_parameter(name = 'Gc',theta0 = initial_values[ii][0],minimum = 0,maximum = 10000,sample = False)
	mcstat.parameters.add_model_parameter(name = 'Ge',theta0 = initial_values[ii][1],minimum = 0,maximum = 10000,sample = False)
	mcstat.parameters.add_model_parameter(name = 'lam_max',theta0 = initial_values[ii][2],minimum = 0,maximum = 10000,sample = False)
	mcstat.parameters.add_model_parameter(name = 'eta',theta0 = initial_values[ii][3],minimum = 0,maximum = 10000,sample = True)
	mcstat.parameters.add_model_parameter(name = 'gamma',theta0 = initial_values[ii][4],minimum = 0,maximum = 10000,sample = True)
	#Define a folder to save all the chain statistics
	nsimu=2e5
	mcstat.simulation_options.define_simulation_options(
		nsimu = nsimu,
		chainfile = 'chain_'+f'{ii}',
		save_to_txt=True,
		save_to_json = True,
		updatesigma = True,
		savedir=savedir
		)
	mcstat.run_simulation()




### Now here we set up a parallel simulation 



# parMC = ParallelMCMC()


# #Provide initial values for the 3 chains
# initial_values = np.array([theta0vec,theta1vec,theta2vec])
# parMC.setup_parallel_simulation(mcset=mcstat,
# 	initial_values=initial_values,
# 	num_chain=initial_values.shape[0]
# 	)

# #Now run the simulation
# parMC.run_parallel_simulation()
# parMC.display_individual_chain_statistics()







import numpy as np
import ctypes
import scipy.io as sio
import scipy.integrate
import scipy as sp
import matplotlib.pyplot as plt
import theano.tensor as tt
import pymc3 as pm
import arviz as az
from numpy.ctypeslib import ndpointer
import pandas as pd
import pickle
import os
from datetime import datetime
import sys


#First lets import the data
vhbdata = sio.loadmat('vhb4910_data.mat')
#This matlab file has xdata = [time,stretch] and ydata = [stress]. It is arranged as multiple dicts and is tedious to extract from
#The data is only for one paticular strain rate of 0.67
time = vhbdata['data']['xdata'][0][0][:,0]
stretch = vhbdata['data']['xdata'][0][0][:,1]
stress = vhbdata['data']['ydata'][0][0][:,0]
#number of timesteps
nds = len(time)
y0 = stress[0]

### import functions from C library
###Using the visc model defined in the c++ file
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


#For saving all necesarry files
date = datetime.now().strftime('%Y%m%d_%H%M%S')
savedir = str('{}'.format(date))


### If I want to use odeint to integrate the linear viscoelastic ODE, I need the below functions

def lam_rate(t):
    if(t > time[np.argmax(stress)]):
        return -0.67
    else:
        return 0.67
def visc_el(y,t,p):
    return p[0]*lam_rate(t) - ((p[0]/p[1]) * y[0])


### Define nonaffine hyperelastic model

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


### Define Overall prediction model
def pred_model(theta):
	#If I did decide to use odeint, I am integrating at the timesteps of my data therefore maintaining same frequency
    t = time
    gamma,eta,sigma = theta
    # args = tuple([[gamma,eta]])
#     y_visc = scipy.integrate.odeint(visc_el,t=t,y0=y0,args=args)
	#Call the c++ model to calculate the stress from the viscoelastic model
    y_visc = visc_mod(eta,gamma,stretch,time,len(time))
    #These are the parameters of the non affine model that we are not sampling
    theta0 = {'Gc' : 7.5541, 'Ge' : 17.69, 'lam_max' : 4.8333}
    #Call the nonafine hyperelastic model to get the stress from it
    y_hyper = nonaffine_hyperelastic_model(theta0,stretch)
    #Add the stresses
    y_total = y_visc + y_hyper
    #Need to free this memory
    #Need to free the visc stress as it was alloted on the heap
    free_mem(y_visc)
    return y_total

#This is just a gaussian log likelihood
def my_loglike(theta,data):
    model = pred_model(theta)
    gamma,eta,sigma = theta
    res = model - data.T.reshape(-1,1)
    logp = -len(data) * np.log(np.sqrt(2. * np.pi) * sigma)
    logp += (-1)*np.sum((res)**2/(2.*sigma**2))
    
    return logp
#This is the gradient of the log likelihood function
def grad_loglike(theta,data):
    def loglike(theta):
        return my_loglike(theta,data)
    
    #We are not doing finite difference approximation and we define delx as the finite precision 
    delx = np.sqrt(np.finfo(float).eps)
    #We are approximating the partial derivative of the log likelihood with finite difference
    #We have to define delx for each partial derivative of loglike with theta's
    grads = sp.optimize.approx_fprime(theta,loglike,delx*np.ones(len(theta)))
    return grads



# Copy paste the theano Operation class from stackoverflow - 
#https://stackoverflow.com/questions/41109292/solving-odes-in-pymc3

# define a theano Op for our likelihood function
class LogLike(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """
        
        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.loglike_grad = LoglikeGrad(self.data)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables

        # call the log-likelihood function
        logp = self.likelihood(theta, self.data)

        outputs[0][0] = np.array(logp) # output the log-likelihood
    def grad(self,inputs,grad_outputs):
        theta, = inputs
        grads = self.loglike_grad(theta)
        return [grad_outputs[0] * grads]
        
        
#Similarly wrapper class for loglike gradient
class LoglikeGrad(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self,data):
        self.der_likelihood = grad_loglike
        self.data = data

    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        grads = self.der_likelihood(theta, self.data)
        outputs[0][0] = grads



def main():
	#Specify number of draws as a command line argument
	if(len(sys.argv) < 2):
		print("Please provide the number of draws and the stepping method")
	ndraws = int(sys.argv[1])
	#Always burn half - can chnage this at some point
	nburn = int(ndraws/2)
	#Intiatting the loglikelihood object which is a theano operation (op)
	loglike = LogLike(my_loglike,stress)
	with pm.Model() as model:
		#Our two parameters prior distribution is uniform
	    gamma = pm.Uniform("gamma",lower=0,upper=10000,testval = 31)
	    eta = pm.Uniform("eta",lower=0,upper=10000,testval = 708)
	    #We are also sampling our observation noise - Seems like a standard to use Half normal for this
	    sigma = pm.HalfNormal("sigma",sigma = 0.6,testval=0.1)
	    #Convert the parameters into a theano tensor
	    theta = tt.as_tensor_variable([gamma,eta,sigma])
	    
	    #According to this thread, we should use pm.Potential
	    #https://stackoverflow.com/questions/64267546/blackbox-likelihood-example
	    pm.Potential("like",loglike(theta))

	#Now we sample!
	with model:
		#We use metropolis as the algorithm with parameters to be sampled supplied through vars
		if(sys.argv[2] == "nuts"):
			step = pm.NUTS(vars=[eta, gamma, sigma])
		elif(sys.argv[2] == "met"):
			step = pm.Metropolis(vars=[eta, gamma, sigma])
		else:
			print("Please provide nuts or met as the stepping method")
		#We provide starting values for our parameters
		trace = pm.sample(ndraws,step=step, tune=nburn, discard_tuned_samples=True,start={'gamma': 31., 'eta':708., 'sigma':0.1},cores=2)
		#Print the summary of all the parameters
		print(pm.summary(trace).to_string())
		df_sum = pm.summary(trace)
		#Save the trace in date directory
	pm.save_trace(directory = savedir + '_' + str(ndraws),trace=trace)

	#Plot a scatter matrix to show correlations
	df_trace =  pm.trace_to_dataframe(trace)
	pd.plotting.scatter_matrix(df_trace, diagonal='kde',figsize = (12,12));
	
	path = 'images/'
	if(os.path.isdir(path)):
		plt.savefig('./images/'+savedir+'_scatter_'+str(ndraws)+'.png')
	else:
		os.mkdir(path)
		plt.savefig('./images/'+savedir+'_scatter_'+str(ndraws)+'.png')

	#Plot the traces and the posterior distributions
	pm.plots.traceplot(trace,figsize=(14,14))
	plt.savefig('./images/'+savedir+'_trace._'+str(ndraws)+'.png')



	#Save the stats in a csv file
	path = 'results/'
	if(os.path.isdir(path)):
		df_sum.to_csv('./results/'+savedir+'_stats_'+str(ndraws)+'.csv')
	else:
		os.mkdir(path)
		df_sum.to_csv('./results/'+savedir+'_stats_'+str(ndraws)+'.csv')


if __name__ == "__main__":
    main()
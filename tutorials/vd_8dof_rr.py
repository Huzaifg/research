import scipy.io as sio
import scipy as sp
import matplotlib.pyplot as mpl
import aesara
import aesara.tensor as tt
import pymc as pm
import arviz as az
import pandas as pd
import os
from datetime import datetime
import sys
import numpy as np
import pickle
import transcript
from vd_class import vd_8dof
from point import Point
import time


#This is just a gaussian log likelihood
def loglike(theta,data):

    sigmas = np.array(theta[-(data.shape[0]):]).reshape(-1,1)

	#Need to update the parameters using the update params method
    vehicle.update_params(m=2097.85,muf=127.866,mur=129.98,a= 1.6889,b =1.6889,h = 0.713,cf = 1.82,cr = 1.82,Jx = 1289,Jz = 4519,Jxz = 3.265,
    Cf=39000,Cr=48000,r0=0.47,ktf=326332,ktr=326332,krof=31000.0,kror=31000.0,brof=3300.000,bror=3300.000,hrcf=0.379,hrcr=0.327,Jw=11,
    Cxf = 17481,Cxr = 17700,rr=theta[0])


    mod = vehicle.solve_half_impl(t_span = [t_eval[0],t_eval[-1]],t_eval = t_eval,tbar = 1e-2)
    mod = np.transpose(mod)

    vehicle.reset_state(init_state=st)

    return -np.sum(np.sum((mod[[2],:] - data)**2/(2.*sigmas**2))/np.linalg.norm(data,axis = 1))

#The gradient of the log likelihood using finite differences - Needed for gradient based methods
def grad_loglike(theta,data):
	def ll(theta,data):
		return loglike(theta,data)
	
	#We are not doing finite difference approximation and we define delx as the finite precision 
	eps = np.sqrt(np.finfo(float).eps)
	delx = eps * np.sqrt(np.abs(theta))

	return sp.optimize.approx_fprime(theta,ll,delx,data)



# define a aesara Op for our likelihood function
class LogLike(tt.Op):
	itypes = [tt.dvector] # expects a vector of parameter values when called
	otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

	def __init__(self, loglike,data):

		# add inputs as class attributes
		self.likelihood = loglike
		self.data = data
		self.loglike_grad = LoglikeGrad(self.data)

	def perform(self, node, inputs, outputs):
		# the method that is used when calling the Op
		theta, = inputs  # this will contain my variables

		# call the log-likelihood function
		logp = self.likelihood(theta,self.data)

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
		grads = self.der_likelihood(theta,self.data)
		outputs[0][0] = grads


def main():

    # Specify number of draws as a command line argument
    if(len(sys.argv) < 2):
        print("Please provide the number of draws and the stepping method")
    ndraws = int(sys.argv[1])
    #Always burn half - can chnage this at some point
    nburn = int(ndraws/2)
    with open('vd_chrono_rr_1.npy', 'rb') as f:
        data = np.load(f)
    

    #For saving all necesarry files
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    savedir = str('{}'.format(date))



    #Intiatting the loglikelihood object which is a theano operation (op)
    like = LogLike(loglike,data)
    with pm.Model() as model:
        rr = pm.Uniform("rr",lower=0.005,upper=0.03,initval = 0.01)
        sigmaLOV = pm.HalfNormal("sigmaLOV",sigma = 0.1,initval=0.09)

        theta_ = [rr,sigmaLOV]
        theta = tt.as_tensor_variable(theta_)

        pm.Potential("like",like(theta))


        #Now we sample!
        #We use metropolis as the algorithm with parameters to be sampled supplied through vars
        transcript.start('./results/' + savedir + '_dump.log')
        if(sys.argv[2] == "nuts"):
            # step = pm.NUTS()
            # pm.sampling.init_nuts()
            idata = pm.sample(ndraws ,tune=nburn,discard_tuned_samples=True,return_inferencedata=True,target_accept = 0.9, cores=4)
        elif(sys.argv[2] == "met"):
            step = pm.Metropolis()
            idata = pm.sample(ndraws,step=step, tune=nburn,discard_tuned_samples=True,return_inferencedata=True,cores=4)
        elif(sys.argv[2] == "smc"):
            idata = pm.sample_smc(draws = ndraws,parallel=True,cores=8,return_inferencedata=True,progressbar = True)
        else:
            print("Please provide nuts or met as the stepping method")

        # idata.extend(pm.sample_prior_predictive())
        # idata.extend(pm.sample_posterior_predictive(idata))
        idata.to_netcdf('./results/' + savedir + ".nc")
        transcript.start('./results/' + savedir + '.log')


        for i in range(0,len(theta_)):
            print(f"{theta_[i]}")

        try:
            print(az.summary(idata).to_string())
        except KeyError:
            idata.to_netcdf('./results/' + savedir + ".nc")


        transcript.stop('./results/' + savedir + '.log')


if __name__ == "__main__":
    # A zero steering function
    def zero_st(t):
        return 0 *t
    
    def zero_throt(t):
        return 0 *t



    # Used for obtaining the state from the chrono vehicle
    n1  = 560
    n2 = 1060


    # The time duration of the simulation
    st_time = 0.
    end_time = 5.


    # The times for which we want the simulation
    t_eval  = np.arange(st_time,end_time,0.01)


    state = pd.read_csv("rr_1.csv",sep=',',header='infer')

    st = {'x' : state['x'][n1],'y':state['y'][n1],'u':state['vx'][n1],'v':state['vy'][n1],'psi':state['yaw'][n1],
    'phi':state['roll'][n1],'wx':state['roll_rate'][n1],'wz':state['yaw_rate'][n1],
    'wlf' : state['wlf'][n1],'wlr' : state['wlr'][n1],'wrf' : state['wrf'][n1],'wrr' : state['wrr'][n1]}

    vehicle = vd_8dof(states = st)

    # Set the steering and the throttle functions we just created above
    vehicle.set_steering(zero_st)
    vehicle.set_throttle(zero_throt,gr=0.3*0.2)
    vehicle.debug = 0
    main()


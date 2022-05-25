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
from scipy.integrate import solve_ivp 



#Ramp steering function
def ramp_st2(t):
	return 2.4418*np.pi/180.*(t)
def ramp_st(t):
	if(t<1):
		return 0
	else:
		return 1.2*np.pi/180.*(t-1)

def zero_st(t):
	return 0

#Drive torque functions
def step_torque(t):
	if(t<1):
		return 0
	else:
		return 20.

def zero_torque(t):
	return 0.

def ramp_torque(t):
	if(t<1):
		return 0
	else:
		return 5.*(t-1)

def const_torque(t):
	return -138

#Now set up the likelihood functions and so

#This is just a gaussian log likelihood
def loglike(theta,data):
	vehicle = vd_8dof()

	#Set vehicle steering
	vehicle.set_steering(ramp_st2)

	#Set step torque on all 4 wheels
	vehicle.set_torque(const_torque)
	sigmas = np.array(theta[-(data.shape[0]):]).reshape(-1,1)

	vehicle.update_states(x = state['x'][700],y=state['y'][700],u=state['vx'][700],v=state['vy'][700],psi=state['yaw'][700],phi=state['roll'][700],wx=state['roll_rate'][700],
		wz=state['yaw_rate'][700],wlf = state['wlf'][700],wlr = state['wlr'][700],wrf = state['wrf'][700],wrr = state['wrr'][700])
	#Need to update the parameters using the update params method
	vehicle.update_params(Cf=theta[0],Cr=theta[1],krof=theta[2],kror=theta[3],brof=theta[4],bror=theta[5],m=2097.85,muf=127.866,mur=129.98,a= 1.6889,b =1.6889,h = 0.713,
		cf = 1.82,cr = 1.82,Jx = 1317,Jz = 4523,Jxz = 1.4133,r0=0.47,ktf=326332,ktr=326332,
		hrcf=0.379,hrcr=0.327,Jw=3,Cxf = 10000,Cxr = 10000)


	mod = solve_ivp(vehicle.model,t_span=[time[0],time[-1]],y0 = list(vehicle.states.values()),method = 'RK45',t_eval = time,rtol = 10e-4,atol = 10e-9)
	vehicle.reset_state()
	#Using only lateral velocity and yaw rate to compare for now
	return -np.sum(np.sum((mod['y'][[3,5,6,7],:] - data)**2/(2.*sigmas**2))/np.linalg.norm(data,axis = 1))


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


	with open('vd_chrono_ramp.npy', 'rb') as f:
		data = np.load(f)

	#Only taking the left front and left rear wheels
	# data = data[[0,1],:]
	#For saving all necesarry files
	date = datetime.now().strftime('%Y%m%d_%H%M%S')
	savedir = str('{}'.format(date))


	#Intiatting the loglikelihood object which is a theano operation (op)
	like = LogLike(loglike,data)
	with pm.Model() as model:
	
		Cf = pm.Uniform("Cf",lower=-80000,upper=-20000,initval = -40000)
		Cr = pm.Uniform("Cr",lower=-80000,upper=-20000,initval = -40000)
		krof = pm.Uniform("krof",lower=5000,upper=80000,initval = 20000)
		# kror = pm.Uniform("kror",lower=5000,upper=80000,initval = 20000)
		kror = pm.Deterministic('kror',krof)
		brof = pm.Uniform("brof",lower=100,upper=30000,initval = 2000)
		bror = pm.Uniform("bror",lower=100,upper=30000,initval = 2000)
		sigmaRA = pm.HalfNormal("sigmaRA",sigma = 0.008,initval=0.008)
		sigmaRR = pm.HalfNormal("sigmaRR",sigma = 0.003,initval=0.003)
		sigmaLV = pm.HalfNormal("sigmaLV",sigma = 0.009,initval=0.003)
		sigmaYR = pm.HalfNormal("sigmaYR",sigma = 0.03,initval=0.03)

		# Cxf = pm.Uniform("Cxf",lower=2000,upper=7000,initval = 5000)
		# Cxr = pm.Uniform("Cxr",lower=2000,upper=7000,initval = 5000)
		# ktf = pm.Uniform("ktf",lower=150000,upper=250000,initval = 150000)
		# ktr = pm.Uniform("ktr",lower=150000,upper=250000,initval = 150000)
		# sigmaWLF = pm.HalfNormal("sigmaWLF",sigma = 0.2,initval=0.2)
		# sigmaWLR = pm.HalfNormal("sigmaWLR",sigma = 0.2,initval=0.2)


		## All of these will be a tensor 
		# theta_ = [mass,Jx,Jy,Jz,a,b,Jxz,Jw,g,h,cf,cr,muf,mur,ktf,ktr,Cf,Cr,Cxf,Cxr,r0,hrcf,hrcr,krof,kror,brof,bror,sigmaLat_acc,sigmaVy]

		theta_ = [Cf,Cr,krof,kror,brof,bror,sigmaLV,sigmaRA,sigmaRR,sigmaYR]
		# theta_ = [Cf,Cr,sigmaLV,sigmaYR]
		theta = tt.as_tensor_variable(theta_)

		#According to this thread, we should use pm.Potential
		#https://stackoverflow.com/questions/64267546/blackbox-likelihood-example
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
	#Set up the vehicle model with its default parameters and initial conditions
	vehicle = vd_8dof()

	#Set vehicle steering
	vehicle.set_steering(ramp_st2)

	#Set step torque on all 4 wheels
	vehicle.set_torque(const_torque)

	# vehicle.update_params(Cf=-42315,Cr=-44884)

	#time stepping - same as the data
	time  = np.arange(0,3.7,0.01)
	# theta = [6.37190006e+03,5.60269824e+03,0.6,0.6]
	# print(logp_op(theta).eval())
	# print(logprob_grad_op(theta).eval())
	#/home/huzaifa/build_chrono/bin/DEMO_OUTPUT/WHEELED_JSON/Calibration/
	state = pd.read_csv("calib_mod.csv",sep=',',header='infer')

	main()

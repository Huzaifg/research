import scipy.io as sio
import scipy as sp
import matplotlib.pyplot as mpl
import aesara
import aesara.tensor as at
import pymc as pm
import arviz as az
import pandas as pd
import os
from datetime import datetime
import sys
import numpy as np
import pickle
import transcript
from vd_bi_mod_aes import vehicle_bi
import jax.numpy as jnp
from jax import grad,vmap,jit
from jax.test_util import check_grads
from jax.experimental.ode import odeint
import pymc.sampling_jax


#Fake noise function for data
def add_noise(a):
	return a - np.random.normal(loc = 0., scale = abs(a.mean()/20),size = a.shape)



@jit
def loglike(theta,state,time,targets):
	#Evaluate the model
	mod = odeint(vehicle_bi, state , time,theta,rtol=1e-6, atol=1e-5, mxstep=1000)
	#Get the sigmas from theta
	sigmas = jnp.array(theta[-(targets.shape[1]):],float)
	#Evaluate negetive of log likelihood
	return -jnp.sum(jnp.sum((mod[:,[0,2]] - targets)**2/(2.*sigmas**2))/jnp.linalg.norm(targets,axis = 0))


@jit
def grad_loglike(theta,state,time,target):
	#calclate the gradient using jax
	return grad(loglike)(jnp.array(theta,float),state,time,target)




# define a custom aesera operation
class LogLike(at.Op):

	itypes = [at.dvector] # expects a vector of parameter values when called
	otypes = [at.dscalar] # outputs a single scalar value (the log likelihood)

	def __init__(self, loglike,state,time,target):
		# add inputs as class attributes
		self.likelihood = loglike
		self.state = state #Initial conditions
		self.time = time #time steps we want to evaluate the ODE
		self.target = target #data
		self.loglike_grad = LoglikeGrad(self.state,self.time,self.target)

	def perform(self, node, inputs, outputs):
		# the method that is used when calling the Op
		theta, = inputs  # this will contain my variables

		# call the log-likelihood function
		logp = self.likelihood(theta,self.state,self.time,self.target)

		outputs[0][0] = np.array(logp) # output the log-likelihood
	def grad(self,inputs,grad_outputs):
		theta, = inputs
		grads = self.loglike_grad(theta)
		return [grad_outputs[0] * grads]


#Similarly wrapper class for loglike gradient
class LoglikeGrad(at.Op):
	itypes = [at.dvector]
	otypes = [at.dvector]

	def __init__(self,state,time,target):
		self.der_likelihood = grad_loglike
		self.state = state
		self.time = time
		self.target = target


	def perform(self, node, inputs, outputs):
		(theta,) = inputs
		grads = self.der_likelihood(theta,self.state,self.time,self.target)
		outputs[0][0] = grads



def main():
	original_stdout = sys.stdout

	# Specify number of draws as a command line argument
	if(len(sys.argv) < 2):
		print("Please provide the number of draws and the stepping method")
	ndraws = int(sys.argv[1])
	#Always burn half - can chnage this at some point
	nburn = int(ndraws/2)


	# Get the target data
	datafile = 'vd_14dof_470.mat'
	vbdata = sio.loadmat(datafile)
	lat_vel_o = vbdata['lat_vel'].reshape(-1,)
	yaw_rate_o = vbdata['yaw_rate'].reshape(-1,)
	time_o = jnp.asarray(vbdata['tDash'].reshape(-1,),float)

	#Apply fake noise
	lat_vel_o = add_noise(lat_vel_o)
	yaw_rate_o = add_noise(yaw_rate_o)

	#Stack the data
	target = jnp.stack([lat_vel_o,yaw_rate_o],axis=-1)

	#For file saving
	date = datetime.now().strftime('%Y%m%d_%H%M%S')
	savedir = str('{}'.format(date))



	#Initial state
	wf = 50./(3.6 * 0.285) #Angular velocity of front wheel
	wr = 50./(3.6 * 0.285) #Angular velocity of rear wheel
	Vx = 50./3.6 #Longitudanal Velocity
	Vy = 0. #Lateral velocity
	yr = 0. #Yaw rate
	state = jnp.array([Vy,Vx,yr,wf,wr],float)


	like = LogLike(loglike,state,time_o,target)

	with pm.Model() as model:

		Cf = pm.Uniform('Cf',lower = -150000, upper = -50000,initval = -80000) # front axle cornering stiffness (N/rad)
		Cr = pm.Uniform('Cr',lower = -150000, upper = -50000,initval = -80000) # rear axle cornering stiffness (N/rad)
		Iz = pm.Uniform('Iz',lower = 500, upper = 3000,initval = 2450) # yaw moment of inertia (kg.m^2)
		sigmaVy = pm.HalfNormal("sigmaVy",sigma = 0.006,initval=0.005) # Noise for lateral velocity
		sigmaYr = pm.HalfNormal("sigmaLat_acc",sigma = 0.03,initval=0.03) #Noise for yaw rate

		#Hopefully this jnp array works
		theta = at.as_tensor_variable([Cf,Cr,Iz,sigmaVy,sigmaYr])

		#Sample using our custom likelihood
		pm.Potential("like",like(theta))

		#Now we sample!
		#We use metropolis as the algorithm with parameters to be sampled supplied through vars
		transcript.start('./results/' + savedir + '_dump.log')
		if(sys.argv[2] == "nuts"):
			# step = pm.NUTS()
			# idata = pymc.sampling_jax.sample_numpyro_nuts(ndraws,tune=nburn,target_accept = 0.9)
			idata = pm.sample(ndraws ,tune=nburn,discard_tuned_samples=True,return_inferencedata=True,target_accept = 0.9, cores=4)
		elif(sys.argv[2] == "met"):
			step = pm.Metropolis()
			idata = pm.sample(ndraws,step=step, tune=nburn,discard_tuned_samples=True,return_inferencedata=True,cores=4)
		else:
			print("provide nuts or met as the stepping method")





if __name__ == "__main__":
	main()

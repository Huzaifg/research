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
from aesara.graph import Apply, Op


#Fake noise function for data
def add_noise(a):
	return a - np.random.normal(loc = 0., scale = abs(a.mean()/20),size = a.shape)



@jit
def loglike(theta,state,time,targets):
	#Evaluate the model
	mod = odeint(vehicle_bi, state , time,theta,rtol=1e-4, atol=1e-7, mxstep=10000)
	#Get the sigmas from theta
	sigmas = jnp.array(theta[-(targets.shape[1]):],float)
	#Evaluate negetive of log likelihood
	return -jnp.sum(jnp.sum((mod[:,[0,2]] - targets)**2/(2.*sigmas**2))/jnp.linalg.norm(targets,axis = 0))



#Gradient of loglikelihood function
grad_loglike = jit(grad(loglike,argnums=list(range(4))))



# define a custom aesera operation
class LogLike(at.Op):

	def make_node(self, *inputs):
		# Convert our inputs to symbolic variables
		inputs = [at.as_tensor_variable(inp) for inp in inputs]
		# Define the type of the output returned by the wrapped JAX function
		outputs = [at.dscalar()]
		return Apply(self, inputs, outputs)

	# def __init__(self, loglike,state,time,target):
	# 	# add inputs as class attributes
	# 	self.likelihood = loglike
	# 	self.state = state #Initial conditions
	# 	self.time = time #time steps we want to evaluate the ODE
	# 	self.target = target #data
	# 	self.loglike_grad = LoglikeGrad(self.state,self.time,self.target)

	def perform(self, node, inputs, outputs):
		result = loglike(*inputs)
		# Aesara raises an error if the dtype of the returned output is not
		# exactly the one expected from the Apply node (in this case
		# `dscalar`, which stands for float64 scalar), so we make sure
		# to convert to the expected dtype. To avoid unecessary conversions
		# you should make sure the expected output defined in `make_node`
		# is already of the correct dtype
		outputs[0][0] = np.asarray(result, dtype=node.outputs[0].dtype)

	def grad(self, inputs, output_gradients):
		gradients = logprob_grad_op(*inputs)
		return [output_gradients[0] * gradient for gradient in gradients]


#Similarly wrapper class for loglike gradient
class LoglikeGrad(at.Op):

	def make_node(self, *inputs):
		inputs = [at.as_tensor_variable(inp) for inp in inputs]
		# This `Op` wil return one gradient per input. For simplicity, we assume
		# each output is of the same type as the input. In practice, you should use
		# the exact dtype to avoid overhead when saving the results of the computation
		# in `perform`
		outputs = [inp.type() for inp in inputs]
		return Apply(self, inputs, outputs)

	def perform(self, node, inputs, outputs):
		# If there are inputs for which the gradients will never be needed or cannot
		# be computed, `aesara.gradient.grad_not_implemented` should  be used
		results = grad_loglike(*inputs)
		for i, result in enumerate(results):
			outputs[i][0] = np.asarray(result, dtype=node.outputs[i].dtype)

# Initialize our `Op`s
logp_op = LogLike()
logprob_grad_op = LoglikeGrad()



def main():
	# Specify number of draws as a command line argument
	if(len(sys.argv) < 2):
		print("Please provide the number of draws and the stepping method")
	ndraws = int(sys.argv[1])
	#Always burn half - can chnage this at some point
	nburn = int(ndraws/2)




	datafile = 'vd_14dof_470.mat'
	vbdata = sio.loadmat(datafile)
	lat_vel_o = vbdata['lat_vel'].reshape(-1,)
	yaw_rate_o = vbdata['yaw_rate'].reshape(-1,)
	# time_o = jnp.asarray(vbdata['tDash'].reshape(-1,),float)
	time_o = vbdata['tDash'].reshape(-1,)

	#Apply fake noise
	lat_vel_o = add_noise(lat_vel_o)
	yaw_rate_o = add_noise(yaw_rate_o)

	#Stack the data
	target = np.stack([lat_vel_o,yaw_rate_o],axis=-1)

	#For file saving
	date = datetime.now().strftime('%Y%m%d_%H%M%S')
	savedir = str('{}'.format(date))



	#Initial state
	wf = 50./(3.6 * 0.285) #Angular velocity of front wheel
	wr = 50./(3.6 * 0.285) #Angular velocity of rear wheel
	Vx = 50./3.6 #Longitudanal Velocity
	Vy = 0. #Lateral velocity
	yr = 0. #Yaw rate
	# state = jnp.array([Vy,Vx,yr,wf,wr],float)
	state = [Vy,Vx,yr,wf,wr]

	Cf = -88000.
	Cr = -88000.
	Iz = 1000.
	sigmaVy = 0.006
	sigmaYr = 0.04
	theta = [Cf,Cr,Iz,sigmaVy,sigmaYr]

	logp_op(theta,state,time_o,target).eval()
	logprob_grad_op(theta,state,time_o,target)[1].eval()

	with pm.Model() as model:

		Cf = pm.Uniform('Cf',lower = -150000, upper = -50000,initval = -80000) # front axle cornering stiffness (N/rad)
		Cr = pm.Uniform('Cr',lower = -150000, upper = -50000,initval = -80000) # rear axle cornering stiffness (N/rad)
		Iz = pm.Uniform('Iz',lower = 100, upper = 3000,initval = 2450) # yaw moment of inertia (kg.m^2)
		sigmaVy = pm.HalfNormal("sigmaVy",sigma = 0.006,initval=0.005) # Noise for lateral velocity
		sigmaYr = pm.HalfNormal("sigmaLat_acc",sigma = 0.03,initval=0.03) #Noise for yaw rate

		#Hopefully this jnp array works
		theta_ = [Cf,Cr,Iz,sigmaVy,sigmaYr]
		theta = at.as_tensor_variable([Cf,Cr,Iz,sigmaVy,sigmaYr])

		#Sample using our custom likelihood
		pm.Potential("like",logp_op(theta,state,time_o,target))


		#Sampling
		transcript.start('./results/' + savedir + '_dump.log')
		if(sys.argv[2] == "nuts"):
			# step = pm.NUTS()
			# idata = pymc.sampling_jax.sample_numpyro_nuts(ndraws,tune=nburn,target_accept = 0.9)
			idata = pm.sample(ndraws ,tune=nburn,discard_tuned_samples=True,return_inferencedata=True,target_accept = 0.9, cores=4)
		elif(sys.argv[2] == "met"):
			step = pm.Metropolis()
			idata = pm.sample(ndraws,step=step, tune=nburn,discard_tuned_samples=True,return_inferencedata=True,cores=4)
		elif(sys.argv[2] == "smc"):
			idata = pm.sample_smc(draws = ndraws,return_inferencedata=True,cores=4)
		else:
			print("provide nuts or met as the stepping method")


		transcript.start('./results/' + savedir + '.log')

		print(f"{datafile=}")

		for i in range(0,len(theta_)):
			print(f"{theta_[i]}")

		try:
			print(az.summary(idata).to_string())
		except KeyError:
			idata.to_netcdf('./results/' + savedir + ".nc")

		idata.to_netcdf('./results/' + savedir + ".nc")
		transcript.stop('./results/' + savedir + '.log')


if __name__ == "__main__":
	main()

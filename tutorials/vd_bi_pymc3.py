import scipy.io as sio
import scipy as sp
import matplotlib.pyplot as mpl
import theano.tensor as tt
import pymc3 as pm
import arviz as az
import pandas as pd
import os
from datetime import datetime
import sys
import numpy as np


from sklearn.preprocessing import normalize
#Import our bi vehicle model
from vd_bi_mod import vehicle_bi

## Function for scaling a vector to [0,1]
def scale_0_1(a):
	return (a - np.min(a))/np.ptp(a)


## Function to add noise to generate the "fake data"
def add_noise(a):
	return a - np.random.uniform(-a.mean()/20,a.mean()/20)   

#This is just a gaussian log likelihood - Right now our data is just a vector so its almost the same as the previous example
def loglike(theta,time_o,st_inp,init_cond,data):
    sigma = theta[-1]
    mod_data = vehicle_bi(theta,time_o,st_inp,init_cond)
    # Calculate the difference
    res = mod_data - data
    # We will calculate the ssq for each vector and divide it with the norm of the data vector. This way we can
    # then add all the individual ssqs without scaling problems
    indi_ss = [None]*res.shape[0]
    for i in range(0,res.shape[0]):
        ss = np.sum(res[i,:]**2)
        ss = ss / np.linalg.norm(data[i,:])
        indi_ss[i] = ss
    # We will just use the sum of all these values as our sum of square - this gives equal weight
    ssq = np.sum(indi_ss)

    logp = -data.shape[0] * np.log(np.sqrt(2. * np.pi) * sigma)
    logp += (-1)*ssq/(2.*sigma**2)
    return logp


#This is the gradient of the likelihood - Needed for the Hamiltonian Monte Carlo (HMC) method
def grad_loglike(theta,time_o,st_inp,init_cond,data):
	def ll(theta,time_o,st_inp,init_cond,data):
		return loglike(theta,time_o,st_inp,init_cond,data)
	
	#We are not doing finite difference approximation and we define delx as the finite precision 
	delx = np.sqrt(np.finfo(float).eps)
	#We are approximating the partial derivative of the log likelihood with finite difference
	#We have to define delx for each partial derivative of loglike with theta's
	grads = sp.optimize.approx_fprime(theta,ll,delx*np.ones(len(theta)),time_o,st_inp,init_cond,data)
	return grads


#Copy paste the theano Operation class from stackoverflow - 
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

	def __init__(self, loglike,time_o,st_inp,init_cond,data):
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
		self.st_inp = st_inp
		self.init_cond = init_cond
		self.time_o = time_o
		self.loglike_grad = LoglikeGrad(self.time_o,self.st_inp,self.init_cond,self.data)

	def perform(self, node, inputs, outputs):
		# the method that is used when calling the Op
		theta, = inputs  # this will contain my variables

		# call the log-likelihood function
		logp = self.likelihood(theta,self.time_o,self.st_inp,self.init_cond,self.data)

		outputs[0][0] = np.array(logp) # output the log-likelihood
	def grad(self,inputs,grad_outputs):
		theta, = inputs
		grads = self.loglike_grad(theta)
		return [grad_outputs[0] * grads]
		
		
#Similarly wrapper class for loglike gradient
class LoglikeGrad(tt.Op):
	itypes = [tt.dvector]
	otypes = [tt.dvector]

	def __init__(self,time_o,st_inp,init_cond,data):
		self.der_likelihood = grad_loglike
		self.data = data
		self.st_inp = st_inp
		self.time_o = time_o
		self.init_cond = init_cond

	def perform(self, node, inputs, outputs):
		(theta,) = inputs
		grads = self.der_likelihood(theta,self.time_o,self.st_inp,self.init_cond,self.data)
		outputs[0][0] = grads

def main():
	# Specify number of draws as a command line argument
	if(len(sys.argv) < 2):
		print("Please provide the number of draws and the stepping method")
	ndraws = int(sys.argv[1])
	#Always burn half - can chnage this at some point
	nburn = int(ndraws/2)


	# First lets load all our data - In this case, our data is from the 14dof model - Should probably add noise to it
	vbdata = sio.loadmat('vd_14dof_470.mat')
	time_o = vbdata['tDash'].reshape(-1,)
	st_inp_o = vbdata['delta4'].reshape(-1,)
	# st_inp_rad = st_inp_o*np.pi/180
	lat_acc_o = vbdata['ay1'].reshape(-1,)
	lat_vel_o = vbdata['lat_vel'].reshape(-1,)
	long_vel_o = vbdata['long_vel'].reshape(-1,)
	roll_angle_o = vbdata['roll_angle'].reshape(-1,)
	yaw_rate_o = vbdata['yaw_rate'].reshape(-1,)
	psi_angle_o = vbdata['psi_angle'].reshape(-1,)

	#Make all these vectors into column vecctors in the matrix data
	data = np.array([long_vel_o,lat_vel_o,psi_angle_o,yaw_rate_o,lat_acc_o])

	## Add noise to all the outputs

	noOutputs= data.shape[0]
	for i in range(noOutputs):
		data[i,:] = add_noise(data[i,:])
		


	#For saving all necesarry files
	date = datetime.now().strftime('%Y%m%d_%H%M%S')
	savedir = str('{}'.format(date))

	### The initial conditions of the bicycle - Maybe in the future this should also come from the data
	init_cond = { 'Vy' : 0, 'Vx' : 50./3.6 , 'psi' : 0, 'psi_dot' : 0, 'Y' : 0, 'X' : 0}


	#Intiatting the loglikelihood object which is a theano operation (op)
	like = LogLike(loglike,time_o,st_inp_o,init_cond,data)
	with pm.Model() as model:
		# Now we declare all the thetas, we will sample everythign because we only have 10 parameters

		# a = pm.Uniform('a',lower = 0.001, upper = 5,testval = 2)  # distance of c.g. from front axle (m) - Maximum length of a truck is 65 feet
		a = 1.14
		# b = pm.Uniform('b',lower = 0.001, upper = 5,testval = 2) # distance of c.g. from rear axle  (m)
		b = 1.4
		Cf = pm.Uniform('Cf',lower = -100000, upper = -80000,testval = -80001) # front axle cornering stiffness (N/rad)
		# Cr = pm.Uniform('Cr',lower = -100000, upper = -40000,testval = -50000) # rear axle cornering stiffness (N/rad)
		Cr = pm.Deterministic("Cr",Cf) 
		Cxf = pm.Uniform('Cxf',lower = 8000, upper = 12000,testval = 8001) # front axle longitudinal stiffness (N)
		# Cxr = pm.Uniform('Cxr',lower = 4000, upper = 20000,testval = 5000) # rear axle longitudinal stiffness (N)
		Cxr = pm.Deterministic("Cxr",Cxf)
		# m = pm.Uniform('m',lower = 1, upper = 5000,testval = 2000)  # the mass of the vehicle (kg)
		m = 1720
		Iz = pm.Uniform('Iz',lower = 2000, upper = 2500,testval = 2100) # yaw moment of inertia (kg.m^2)
		# Rr = pm.Uniform('Rr',lower = 0.001, upper = 2,testval = 1) # wheel radius
		Rr = 0.285
		# Jw = pm.Uniform('Jw',lower = 0.001, upper = 5,testval = 1) # wheel roll inertia
		Jw = 2


		#We are also sampling our observation noise - Seems like a standard to use Half normal for this
		sigma = pm.HalfNormal("sigma",sigma = 0.6,testval=0.1)

		## Convert our theta into a theano tensor
		theta = tt.as_tensor_variable([a,b,Cf,Cr,Cxf,Cxr,m,Iz,Rr,Jw])

		#According to this thread, we should use pm.Potential
		#https://stackoverflow.com/questions/64267546/blackbox-likelihood-example
		pm.Potential("like",like(theta))

		#Now we sample!
		with model:
			#We use metropolis as the algorithm with parameters to be sampled supplied through vars
			if(sys.argv[2] == "nuts"):
				# step = pm.NUTS()
				pm.sampling.init_nuts()
				idata = pm.sample(ndraws ,tune=nburn,discard_tuned_samples=True,return_inferencedata=True,cores=2)
			elif(sys.argv[2] == "met"):
				step = pm.Metropolis()
				idata = pm.sample(ndraws,step=step, tune=nburn,discard_tuned_samples=True,return_inferencedata=True,cores=2)
			else:
				print("Please provide nuts or met as the stepping method")
			#We provide starting values for our parameters
			# start = pm.find_MAP()
			
			#Print the summary of all the parameters
			trace = idata.posterior
			# print(pm.summary(trace).to_string())
			# print(az.summary(idata,var_names = ['a','b','Cf','Cr','Cxf','Cxr','m','Iz','Rr','Jw']).to_string())
			print(az.summary(idata,var_names = ['Cf','Cxf','Iz']).to_string())
			# df_sum = pm.summary(trace)
			#Save the trace in date directory - Using the ArviZ inference data
			idata.to_netcdf(savedir + ".nc")
			# pm.save_trace(directory = savedir + '_' + str(ndraws),trace=trace)
			#Save the stats in a csv file


		# path = 'results/'
		# if(os.path.isdir(path)):
		# 	df_sum.to_csv('./results/'+savedir+'_stats_'+str(ndraws)+'.csv')
		# else:
		# 	os.mkdir(path)
		# 	df_sum.to_csv('./results/'+savedir+'_stats_'+str(ndraws)+'.csv')


if __name__ == "__main__":
	main()


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
import pickle
import transcript


# from sklearn.preprocessing import normalize
#Import our bi vehicle model
from vd_bi_mod import vehicle_bi

## Function for scaling a vector to [0,1]
def scale_0_1(a):
	return (a - np.min(a))/np.ptp(a)


## Function to add noise to generate the "fake data"
# def add_noise(a):
# 	return a - np.random.uniform(-a.mean()/20,a.mean()/20)   

def long_vel_noise(a):
	return a - np.random.normal(loc = 0., scale = abs(a.mean()/200),size = a.shape)
## Adding normally distributed noise at each time instant
def add_noise(a):
	return a - np.random.normal(loc = 0., scale = abs(a.mean()/20),size = a.shape)

#This is just a gaussian log likelihood - Right now our data is just a vector so its almost the same as the previous example
def loglike(theta,time_o,st_inp,init_cond,data):
    sigma_vec = theta[-(data.shape[0]):]

    mod_data = vehicle_bi(theta,time_o,st_inp,init_cond)
    # Using only the lateeral acceleration and the lateral veclocity
    mod_data = mod_data[[0,3],:]
    # Discard data that is not around the manuever


    # ## Find the index for which steering input is non zero 
    # non_z_ind = np.where(st_inp != 0 )[0][0]

    # #We will take 20 points before and 100 points after
    # start = non_z_ind - 20
    # end = non_z_ind + 100

    # #Now we filter that data out - Taking only the lateral acceleration - if taking all- remove the reshape from all three below lines
    # mod_data = mod_data[:,start:end]
    # data = data[:,start:end]


    # Calculate the difference
    res = (mod_data - data)
 
    # We will calculate the ssq for each vector and divide it with the norm of the data vector. This way we can
    # then add all the individual ssqs without scaling problems
    # To prevent divide by zero error in ss as sigma can be 0
    eps = np.finfo(float).eps
    norm_ss = [None]*res.shape[0]
    for i in range(0,res.shape[0]):
        ss = np.sum(res[i,:]**2/(2.*(sigma_vec[i] + eps)**2.))
        ss = ss / np.linalg.norm(data[i,:])
        norm_ss[i] = ss
    # We will just use the sum of all these values as our sum of square - this gives equal weight
    ssq = np.sum(norm_ss)
    #Adding twice the ss for lat velocity
    # ssq = norm_ss[0] * 2 + norm_ss[1]

    # logp = -data.shape[1] * np.log(np.sqrt(2. * np.pi) * sigma)
    # logp += (-1)*ssq/(2.*sigma**2)
    logp = (-1)*ssq
    return logp


# Log like for common sigma to eleviate the pains of sampling so many sigmas
# def loglike(theta,time_o,st_inp,init_cond,data):
#     sigma_vec = theta[-1]
#     # sigma_vec = theta[-1:]
#     mod_data = vehicle_bi(theta,time_o,st_inp,init_cond)
#     # Using only the lateeral acceleration and the lateral veclocity
#     mod_data = mod_data[[0,3],:]
#     # Discard data that is not around the manuever


#     # ## Find the index for which steering input is non zero 
#     # non_z_ind = np.where(st_inp != 0 )[0][0]

#     # #We will take 20 points before and 100 points after
#     # start = non_z_ind - 20
#     # end = non_z_ind + 100

#     # #Now we filter that data out - Taking only the lateral acceleration - if taking all- remove the reshape from all three below lines
#     # mod_data = mod_data[:,start:end]
#     # data = data[:,start:end]


#     # Calculate the difference
#     res = (mod_data - data)
 
#     # We will calculate the ssq for each vector and divide it with the norm of the data vector. This way we can
#     # then add all the individual ssqs without scaling problems
#     norm_ss = [None]*res.shape[0]
#     for i in range(0,res.shape[0]):
#         ss = np.sum(res[i,:]**2)
#         ss = ss / np.linalg.norm(data[i,:])
#         norm_ss[i] = ss
#     # We will just use the sum of all these values as our sum of square - this gives equal weight
#     ssq = np.sum(norm_ss)

#     # logp = -data.shape[1] * np.log(np.sqrt(2. * np.pi) * sigma)
#     # logp += (-1)*ssq/(2.*sigma**2)
#     logp = (-1)*ssq/(2.*sigma**2)
#     return logp


# #This is just a gaussian log likelihood - This is when we onyl want to use lateral acceleration
# def loglike(theta,time_o,st_inp,init_cond,data):
#     sigma_vec = theta[-1:]
#     mod_data = vehicle_bi(theta,time_o,st_inp,init_cond)
#     # Discard data that is not around the manuever


#     # ## Find the index for which steering input is non zero 
#     # non_z_ind = np.where(st_inp != 0 )[0][0]

#     # #We will take 20 points before and 100 points after
#     # start = non_z_ind - 20
#     # end = non_z_ind + 250

#     # #Now we filter that data out - Taking only the lateral acceleration - if taking all- remove the reshape from all three below lines
#     # mod_data = mod_data[-1:,start:end]
#     # data = data[-1:,start:end]


#     # Calculate the difference
#     res = (data - mod_data)
#     # print(res,res.shape)
# #     logp = -data.shape[1] * np.log(np.sqrt(2. * np.pi) * sigma_vec[0])
#     logp = -np.sum(res**2/(2.*sigma_vec[0]**2.))
    
#     return logp

#This is the gradient of the likelihood - Needed for the Hamiltonian Monte Carlo (HMC) method
def grad_loglike(theta,time_o,st_inp,init_cond,data):
	def ll(theta,time_o,st_inp,init_cond,data):
		return loglike(theta,time_o,st_inp,init_cond,data)
	
	#We are not doing finite difference approximation and we define delx as the finite precision 
	# l = len(theta)
	# delxs = [0] * l
	delx = np.sqrt(np.finfo(float).eps)
	# delxs = delx * np.sqrt(np.abs(theta))
	#We are approximating the partial derivative of the log likelihood with finite difference
	#We have to define delx for each partial derivative of loglike with theta's
	# grads = sp.optimize.approx_fprime(theta,ll,delx*np.ones(len(theta)),time_o,st_inp,init_cond,data)

	return sp.optimize.approx_fprime(theta,ll,delx*np.ones(len(theta)),time_o,st_inp,init_cond,data)
	# return sp.optimize.approx_fprime(theta,ll,delxs,time_o,st_inp,init_cond,data)


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
	original_stdout = sys.stdout

	# Specify number of draws as a command line argument
	if(len(sys.argv) < 2):
		print("Please provide the number of draws and the stepping method")
	ndraws = int(sys.argv[1])
	#Always burn half - can chnage this at some point
	nburn = int(ndraws/2)


	# First lets load all our data - In this case, our data is from the 14dof model - Should probably add noise to it
	datafile = 'vd_14dof_470.mat'
	vbdata = sio.loadmat(datafile)
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
	# data = np.array([long_vel_o,lat_vel_o,psi_angle_o,yaw_rate_o,lat_acc_o])
	# data = np.array([lat_vel_o,psi_angle_o,yaw_rate_o,lat_acc_o])
	data = np.array([lat_vel_o,lat_acc_o])
	# data = np.array([lat_vel_o])
	## Add noise to all the outputs

	# Add all the noise
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

		# a = pm.Uniform('a',lower = 0.5, upper = 2,testval = 1.25)  # distance of c.g. from front axle (m) - Maximum length of a truck is 65 feet
		a = 1.14
		# a = pm.Normal('a',mu=1., sigma = 0.25,testval = 1.25)
		# b= pm.Normal('b',mu=1.25, sigma = 0.25,testval = 1.4)
		# b = pm.Uniform('b',lower = 0.5, upper = 1.8,testval = 1.25) # distance of c.g. from rear axle  (m)
		b = 1.4
		Cf = pm.Uniform('Cf',lower = -150000, upper = -50000,testval = -80000) # front axle cornering stiffness (N/rad)
		# Cf = pm.Uniform('Cf',lower = -1.3, upper = -0.6, testval = -0.7)
		# Cf = pm.TruncatedNormal('Cf',mu = -80000.,sigma = 40000,upper = -10000,lower = -10**6,testval = -100000)
		Cr = pm.Uniform('Cr',lower = -150000, upper = -50000,testval = -80000) # rear axle cornering stiffness (N/rad)
		# Cr = pm.Uniform('Cr',lower = -1.3, upper = -0.6, testval = -0.7)
		# Cr = pm.TruncatedNormal('Cr',mu = -80000.,sigma = 40000,upper = -10000,lower = -10**6,testval = -100000)
		# Cr = -88000
		# Cr = pm.Deterministic("Cr",Cf) 
		# Cxf = pm.Uniform('Cxf',lower = 3000, upper = 14000,testval = 8000) # front axle longitudinal stiffness (N)
		# Cxf = pm.TruncatedNormal('Cxf',mu = 9000, sigma = 5000, lower = 1000)
		# Cxf = pm.Normal('Cxf', mu = 9000. , sigma = 5000, testval = 8000)
		# Cxf = 10000
		# Cxr = pm.Uniform('Cxr',lower = 3000, upper = 14000,testval = 8000) # rear axle longitudinal stiffness (N)
		# Cxr = pm.Normal('Cxr', mu = 9000. , sigma = 5000, testval = 8000)
		# Cxr = pm.TruncatedNormal('Cxr',mu = 9000, sigma = 5000, lower = 1000)
		# Cxr = 10000
		# Cxr = pm.Deterministic("Cxr",Cxf)
		# Non centred mass implementation
		# m_ = pm.Normal('m_',mu = 0., sigma = 1.,testval = 0.5)  # the mass of the vehicle (kg)
		# m = pm.Deterministic('m', 1600 + m_ * 300)
		# m = pm.TruncatedNormal('m',mu = 1300., sigma = 600.,lower = 100,upper = 3500,testval = 1400)
		# m = pm.Uniform('m',lower = 500, upper = 3500, testval = 1400)
		# m = 1720
		# Iz = 2420
		# Iz = pm.TruncatedNormal('Iz',mu = 2000, sigma = 1000, lower = 300,upper = 3500,testval= 2450)
		Iz = pm.Uniform('Iz',lower = 500, upper = 3000,testval = 2450) # yaw moment of inertia (kg.m^2)
		# Rr = pm.Uniform('Rr',lower = 0.001, upper = 2,testval = 1) # wheel radius
		Rr = 0.285
			# Jw = pm.Uniform('Jw',lower = 0.001, upper = 5,testval = 1) # wheel roll inertia
		Jw = 2
		
		



		#We are also sampling our observation noise - Seems like a standard to use Half normal for this - Expect the same amount of precicsion so same prior
		# sigmaVx = pm.HalfNormal("sigmaVx",sigma = 0.14,testval=0.1)
		sigmaVy = pm.HalfNormal("sigmaVy",sigma = 0.006,testval=0.005)
		# sigmaPsi = pm.HalfNormal("sigmaPsi",sigma = 0.6,testval=0.1)
		# sigmaPsi_dot = pm.HalfNormal("sigmaPsi_dot",sigma = 0.6,testval=0.1)
		# sigmaLat_acc = pm.Normal("sigmaLat_acc",mu = 0,sigma = 0.06,testval=0.05)
		sigmaLat_acc = pm.HalfNormal("sigmaLat_acc",sigma = 0.3,testval=0.3)
		# sigmaCom = pm.HalfNormal("sigmaCom",sigma = 0.06,testval=0.05)

		## Convert our theta into a theano tensor
		# theta_ = [a,b,Cf,Cr,Cxf,Cxr,m,Iz,Rr,Jw,sigmaVy,sigmaLat_acc]
		# theta_ = [a,b,Cf,Cr,Cxf,Cxr,m,Iz,Rr,Jw,m_,sigmaLat_acc,sigmaVy]
		theta_ = [Cf,Cr,Iz,sigmaVy,sigmaLat_acc]
		# theta_ = [Cf,Cr,m,sigmaVy]
		# theta = tt.as_tensor_variable([a,b,Cf,Cr,Cxf,Cxr,m,Iz,Rr,Jw,sigmaVy,sigmaPsi,sigmaPsi_dot,sigmaLat_acc])
		theta = tt.as_tensor_variable(theta_)




		#According to this thread, we should use pm.Potential
		#https://stackoverflow.com/questions/64267546/blackbox-likelihood-example
		pm.Potential("like",like(theta))

		#Now we sample!
		#We use metropolis as the algorithm with parameters to be sampled supplied through vars
		transcript.start('./results/' + savedir + '_dump.log')
		if(sys.argv[2] == "nuts"):
			# step = pm.NUTS()
			
			idata = pm.sample(ndraws ,tune=nburn,discard_tuned_samples=True,return_inferencedata=True,target_accept = 0.9, cores=4)
		elif(sys.argv[2] == "met"):
			step = pm.Metropolis()
			idata = pm.sample(ndraws,step=step, tune=nburn,discard_tuned_samples=True,return_inferencedata=True,cores=4)
		else:
			print("Please provide nuts or met as the stepping method")


		trace = idata.posterior


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


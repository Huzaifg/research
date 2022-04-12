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
#Import our 8dof vehicle model
from vd_8dof_mod import vehicle_8dof


## Adding normally distributed noise at each time instant
def add_noise(a):
	return a - np.random.normal(loc = 0., scale = abs(a.mean()/20),size = a.shape)


#This is just a gaussian log likelihood - Right now our data is just a vector so its almost the same as the previous example
def loglike(theta,time_o,st_inp,init_cond,data):
	sigma_vec = theta[-(data.shape[0]):]

	mod_data = vehicle_8dof(theta,time_o,st_inp,init_cond)
	# Using only the lateeral acceleration and the lateral veclocity
	mod_data = mod_data[[1,3],:]

	# Calculate the difference
	res = (mod_data - data)
 
	# We will calculate the ssq for each vector and divide it with the norm of the data vector. This way we can
	# then add all the individual ssqs without scaling problems
	norm_ss = [None]*res.shape[0]
	for i in range(0,res.shape[0]):
		ss = np.sum(res[i,:]**2/(2.*sigma_vec[i]**2.))
		ss = ss / np.linalg.norm(data[i,:])
		norm_ss[i] = ss
	# We will just use the sum of all these values as our sum of square - this gives equal weight
	ssq = np.sum(norm_ss)

	# logp = -data.shape[1] * np.log(np.sqrt(2. * np.pi) * sigma)
	# logp += (-1)*ssq/(2.*sigma**2)
	logp = (-1)*ssq
	return logp


#This is the gradient of the likelihood - Needed for the Hamiltonian Monte Carlo (HMC) method
def grad_loglike(theta,time_o,st_inp,init_cond,data):
	def ll(theta,time_o,st_inp,init_cond,data):
		return loglike(theta,time_o,st_inp,init_cond,data)
	
	#We are not doing finite difference approximation and we define delx as the finite precision 
	delx = np.sqrt(np.finfo(float).eps)
	#We are approximating the partial derivative of the log likelihood with finite difference
	#We have to define delx for each partial derivative of loglike with theta's
	# grads = sp.optimize.approx_fprime(theta,ll,delx*np.ones(len(theta)),time_o,st_inp,init_cond,data)

	# Choose a delx that is propotional to the parameter rather than 10^-16 for all
	l = len(theta)
	delxs = [0]*l
	delxs = delx * np.sqrt(np.abs(theta))


	# return sp.optimize.approx_fprime(theta,ll,delx*np.ones(len(theta)),time_o,st_inp,init_cond,data)
	return sp.optimize.approx_fprime(theta,ll,delxs,time_o,st_inp,init_cond,data)

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

# The true values of the parameters commented out

# m=1400  # Sprung mass (kg)
# Jx=900  # Sprung mass roll inertia (kg.m^2)
# Jy=2000  # Sprung mass pitch inertia (kg.m^2)
# Jz=2420  # Sprung mass yaw inertia (kg.m^2)
# a=1.14  # Distance of sprung mass c.g. from front axle (m)
# b=1.4   # Distance of sprung mass c.g. from rear axle (m)
# Jxz=90  # Sprung mass XZ product of inertia
# Jw=1    #tire/wheel roll inertia kg.m^2
# g=9.8    # acceleration of gravity 
# h=0.75   # Sprung mass c.g. height (m)
# cf=1.5   # front track width (m)
# cr=1.5   # rear track width (m)
# muf=80     #front unsprung mass (kg)
# mur=80     #rear unsprung mass (kg)
# ktf=200000   #front tire stiffness (N/m)
# ktr=200000   #rear tire stiffness (N/m)
# Cf=-44000  #front tire cornering stiffness (N/rad)
# Cr=-47000   #rear tire cornering stiffness (N/rad)
# Cxf=5000  #front tire longitudinal stiffness (N)
# Cxr=5000  #rear tire longitudinal stiffness (N)
# r0=0.285  #nominal tire radius (m)
# hrcf=0.65  #front roll center distance below sprung mass c.g.
# hrcr=0.6   #rear roll center distance below sprung mass c.g.
# krof=29000  #front roll stiffness (Nm/rad)
# kror=29000  #rear roll stiffness (Nm/rad)
# brof=3000   #front roll damping coefficient (Nm.s/rad)
# bror=3000   #rear roll damping coefficient (Nm.s/rad)



def main():
	# Specify number of draws as a command line argument
	if(len(sys.argv) < 2):
		print("Please provide the number of draws and the stepping method")
	ndraws = int(sys.argv[1])
	#Always burn half - can chnage this at some point
	nburn = int(ndraws/2)



	# First lets load all our data - In this case, our data is from the 14dof model - Should probably add noise to it
	datafile = 'vd_8dof_470.mat'
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

	# data = np.array([roll_angle_o,lat_acc_o,yaw_rate_o,lat_vel_o,psi_angle_o,long_vel_o])
	data = np.array([lat_vel_o,lat_acc_o])

	noOutputs= data.shape[0]
	for i in range(noOutputs):
		data[i,:] = add_noise(data[i,:])
	

	#For saving all necesarry files
	date = datetime.now().strftime('%Y%m%d_%H%M%S')
	savedir = str('{}'.format(date))

	#Lets define the initial conditions of the model

	init_cond = { 'u' : 50./3.6, 'v' : 0 , 'u_dot' : 0, 'v_dot' : 0, 'phi' : 0, 'psi' : 0, 'dphi' : 0, 'dpsi' : 0, 'wx' : 0, 'wy' : 0,
'wz' : 0, 'wx_dot' : 0, 'wz_dot' : 0 }

	#Intiatting the loglikelihood object which is a theano operation (op)
	like = LogLike(loglike,time_o,st_inp_o,init_cond,data)
	with pm.Model() as model:
		#We are not sampling the mass so supplying it as a deterministic parameter
	#     mass = pm.Deterministic("mass",1400)
	# 	mass = 1400
	# 	# Sprung mass roll inertia (kg.m^2)
	# 	# Jx = pm.Uniform("Jx",lower=0.0001,upper=10000,testval = 300)
	# 	Jx = 900  # Sprung mass roll inertia (kg.m^2)
	# 	# Sprung mass pitch inertia (kg.m^2)
	# 	# Jy = pm.Uniform("Jy",lower=0.0001,upper=10000,testval = 300)
	# 	Jy  = 2000
	# 	# Sprung mass yaw inertia (kg.m^2)
	# 	# Jz = pm.Uniform("Jz",lower=0.0001,upper=10000,testval = 300)
	# 	Jz = 2420
	# 	# Distance of sprung mass c.g. from front axle (m)
	# 	# a = pm.Uniform("a",lower=0.0001,upper=10,testval = 0.5)
	# 	a = 1.14
	# 	# Distance of sprung mass c.g. from rear axle (m)
	# 	# b = pm.Uniform("b",lower=0.0001,upper=10,testval = 0.5)
	# 	b = 1.4
	# 	# Sprung mass XZ product of inertia
	# 	# Jxz = pm.Uniform("Jxz",lower=0.0001,upper=1000,testval = 300)
	# 	Jxz = 90
	# 	#tire/wheel roll inertia kg.m^2
	# 	# Jw = pm.Uniform("Jw",lower=0.0001,upper=100,testval = 30)
	# 	Jw = 1
	# 	# acceleration of gravity
	# #     g = pm.Deterministic("g",9.8)
	# 	g = 9.8
	# 	# Sprung mass c.g. height (m)
	# 	# h = pm.Uniform("h",lower=0.0001,upper=20,testval = 0.5)
	# 	h = 0.75
	# 	# front track width (m)
	# 	# cf = pm.Uniform("cf",lower=0.0001,upper=20,testval = 0.5)
	# 	cf = 1.5
	# 	# rear track width (m)
	# 	# cr = pm.Uniform("cr",lower=0.0001,upper=20,testval = 0.5)
	# 	cr = 1.5
	# 	#front unsprung mass (kg)
	# 	# muf = pm.Uniform("muf",lower = 0.0001, upper = mass,testval = 100)
	# 	muf = 80
	# 	#rear unsprung mass (kg)
	# 	# mur = pm.Uniform("mur",lower = 0.0001, upper = mass,testval = 100)
	# 	mur = 80
	# 	#front tire stiffness (N/m) - Over here we are assuming that all the tires are identical to reduce the number of parameters
	# 	# ktf = pm.Uniform("ktf",lower=0.0001,upper=500000,testval = 20000)
	# 	ktf = 200000
	# 	#rear tire stiffness (N/m) - Since rear tire is identical to front tire, we supply it as a deterministic variable
	# 	# ktr = pm.Deterministic("ktr",ktf)
	# 	ktr = 200000
		#front tire cornering stiffness (N/rad) - Over here we are assuming that all the tires are identical to reduce the number of parameters
		Cf = pm.Uniform("Cf",lower=-60000,upper=-30000,testval = -40000)
		#rear tire stiffness (N/m) - Since rear tire is identical to front tire, we supply it as a deterministic variable
		# Cr = pm.Deterministic("Cr",Cf)
		Cr = pm.Uniform("Cr",lower=-60000,upper=-30000,testval = -40000)
		#front tire longitudinal stiffness (N)
		# Cxf = pm.Uniform("cxf",lower=0.0001,upper=10000,testval = 1000)
	# 	Cxf = 5000
	# 	#rear tire longitudinal stiffness (N) - Same as front tire
	# 	# Cxr = pm.Deterministic("Cxr",Cxf)
	# 	Cxr = 5000
	# 	#nominal tire radius (m) - Easily measurable so not sampled
	# #     r0 = pm.Deterministic("r0", 0.285)
	# 	r0 = 0.285
	# 	#front roll center distance below sprung mass c.g.
	# 	# hrcf = pm.Uniform("hrcf",lower=0.0001,upper=20,testval = 0.5)
	# 	hrcf = 0.65
	# 	#rear roll center distance below sprung mass c.g.
	# 	# hrcr = pm.Uniform("hrcr",lower=0.0001,upper=20,testval = 0.5)
	# 	hrcr = 0.6
	# 	#front roll stiffness (Nm/rad)
	# 	# krof = pm.Uniform("krof",lower=0.0001,upper=100000,testval = 10000)
	# 	krof = 29000
	# 	#rear roll stiffness (Nm/rad)
	# 	# kror = pm.Uniform("kror",lower=0.0001,upper=100000,testval = 10000)
	# 	kror = 29000
	# 	#front roll damping coefficient (Nm.s/rad)
	# 	# brof = pm.Uniform("brof",lower=0,upper=10000,testval = 1000)
	# 	brof = 3000
	# 	#rear roll damping coefficient (Nm.s/rad)
	# 	# bror = pm.Uniform("bror",lower=0.0001,upper=10000,testval = 1000)
	# 	bror = 3000



		#We are also sampling our observation noise - Seems like a standard to use Half normal for this - Expect the same amount of precicsion so same prior
		# sigmaVx = pm.HalfNormal("sigmaVx",sigma = 0.6,testval=0.1)
		sigmaVy = pm.HalfNormal("sigmaVy",sigma = 0.006,testval=0.005)
		# sigmaPsi = pm.HalfNormal("sigmaPsi",sigma = 0.6,testval=0.1)
		# sigmaPsi_dot = pm.HalfNormal("sigmaPsi_dot",sigma = 0.6,testval=0.1)
		# sigmaLat_acc = pm.Normal("sigmaLat_acc",mu = 0,sigma = 0.06,testval=0.05)
		sigmaLat_acc = pm.HalfNormal("sigmaLat_acc",sigma = 0.6,testval=0.5)


		## All of these will be a tensor 
		# theta_ = [mass,Jx,Jy,Jz,a,b,Jxz,Jw,g,h,cf,cr,muf,mur,ktf,ktr,Cf,Cr,Cxf,Cxr,r0,hrcf,hrcr,krof,kror,brof,bror,sigmaLat_acc,sigmaVy]
		theta_ = [Cf,Cr,sigmaVy,sigmaLat_acc]
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
		else:
			print("Please provide nuts or met as the stepping method")

		idata.to_netcdf('./results/' + savedir + ".nc")
		transcript.start('./results/' + savedir + '.log')

		print(f"{datafile=}")

		for i in range(0,len(theta_)):
			print(f"{theta_[i]}")

		try:
			print(az.summary(idata).to_string())
		except KeyError:
			idata.to_netcdf('./results/' + savedir + ".nc")

		
		transcript.stop('./results/' + savedir + '.log')


if __name__ == "__main__":
	main()


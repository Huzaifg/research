import ctypes
import scipy.io as sio
import scipy as sp
import matplotlib.pyplot as mpl
import theano.tensor as tt
import pymc3 as pm
import arviz as az
from numpy.ctypeslib import ndpointer
import pandas as pd
import pickle
import os
from datetime import datetime
import sys
import numpy as np
#Import our 8dof vehicle model
from vd_8dof_mod import vehicle_8dof

# First lets load all our data - In this case, our data is from the 14dof model - Should probably add noise to it

vbdata = sio.loadmat('vd_14dof.mat')
time_o = vbdata['tDash']
time_o = time_o.reshape(-1,) #reshape it to a column vector
st_inp_o = vbdata['delta4']
# st_inp_rad = st_inp_o*np.pi/180 # This is the steering input used for the data and will hence be used for our model as well - converted from deg to redian
st_inp = st_inp_o.reshape(-1,) # Reshaping it so that it is of the form of a list which is needed by my prediction model
lat_acc_o = vbdata['ay1'] # Lateral Acceleration
lat_vel_o = vbdata['lat_vel'] #Laterial Velocity
roll_angle_o = vbdata['roll_angle'] #Roll Angle

## For the first attempt, I am only using the yaw rate as the data and so only reshaping that
yaw_rate_o = vbdata['yaw_rate'] #Yaw Rate
yaw_rate_o = yaw_rate_o.reshape(-1,) #reshape it to a column vector
yaw_rate_o = yaw_rate_o - np.random.uniform(-yaw_rate_o.mean()/20,yaw_rate_o.mean()/20,size = (yaw_rate_o.shape[0],)) ## Adding random uniform noise to it
psi_angle_o = vbdata['psi_angle'] #Yaw Angle

#For saving all necesarry files
date = datetime.now().strftime('%Y%m%d_%H%M%S')
savedir = str('{}'.format(date))

#Lets define the initial conditions of the model

init_cond = { 'u' : 50./3.6, 'v' : 0 , 'u_dot' : 0, 'v_dot' : 0, 'phi' : 0, 'psi' : 0, 'dphi' : 0, 'dpsi' : 0, 'wx' : 0, 'wy' : 0,
'wz' : 0, 'wx_dot' : 0, 'wz_dot' : 0 }


#This is just a gaussian log likelihood - Right now our data is just a vector so its almost the same as the previous example
def loglike(theta,time_o,st_inp,init_cond,data):
	sigma = theta[-1]
	yaw_rate_mod = vehicle_8dof(theta,time_o,st_inp,init_cond)
	res = yaw_rate_mod - data
	logp = -len(yaw_rate_o) * np.log(np.sqrt(2. * np.pi) * sigma)
	logp += (-1)*np.sum((res)**2/(2.*sigma**2))
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


	#Intiatting the loglikelihood object which is a theano operation (op)
	like = LogLike(loglike,time_o,st_inp,init_cond,yaw_rate_o)
	with pm.Model() as model:
	    #We are not sampling the mass so supplying it as a deterministic parameter
	#     mass = pm.Deterministic("mass",1400)
	    mass = 1400
	    # Sprung mass roll inertia (kg.m^2)
	    Jx = pm.Uniform("Jx",lower=0.0001,upper=10000,testval = 300)
	    # Sprung mass pitch inertia (kg.m^2)
	    Jy = pm.Uniform("Jy",lower=0.0001,upper=10000,testval = 300)
	    # Sprung mass yaw inertia (kg.m^2)
	    Jz = pm.Uniform("Jz",lower=0.0001,upper=10000,testval = 300)
	    # Distance of sprung mass c.g. from front axle (m)
	    a = pm.Uniform("a",lower=0.0001,upper=10,testval = 0.5)
	    # Distance of sprung mass c.g. from rear axle (m)
	    b = pm.Uniform("b",lower=0.0001,upper=10,testval = 0.5)
	    # Sprung mass XZ product of inertia
	    Jxz = pm.Uniform("Jxz",lower=0.0001,upper=1000,testval = 300)
	    #tire/wheel roll inertia kg.m^2
	    Jw = pm.Uniform("Jw",lower=0.0001,upper=100,testval = 30)
	    # acceleration of gravity
	#     g = pm.Deterministic("g",9.8)
	    g = 9.8
	    # Sprung mass c.g. height (m)
	    h = pm.Uniform("h",lower=0.0001,upper=20,testval = 0.5)
	    # front track width (m)
	    cf = pm.Uniform("cf",lower=0.0001,upper=20,testval = 0.5)
	    # rear track width (m)
	    cr = pm.Uniform("cr",lower=0.0001,upper=20,testval = 0.5)
	    #front unsprung mass (kg)
	    muf = pm.Uniform("muf",lower = 0.0001, upper = mass,testval = 100)
	    #rear unsprung mass (kg)
	    mur = pm.Uniform("mur",lower = 0.0001, upper = mass,testval = 100)
	    #front tire stiffness (N/m) - Over here we are assuming that all the tires are identical to reduce the number of parameters
	    ktf = pm.Uniform("ktf",lower=0.0001,upper=500000,testval = 20000)
	    #rear tire stiffness (N/m) - Since rear tire is identical to front tire, we supply it as a deterministic variable
	    ktr = pm.Deterministic("ktr",ktf)
	    #front tire cornering stiffness (N/rad) - Over here we are assuming that all the tires are identical to reduce the number of parameters
	    Cf = pm.Uniform("Cf",lower=-100000,upper=0.0001,testval = -20000)
	    #rear tire stiffness (N/m) - Since rear tire is identical to front tire, we supply it as a deterministic variable
	    Cr = pm.Deterministic("Cr",Cf)
	    #front tire longitudinal stiffness (N)
	    Cxf = pm.Uniform("cxf",lower=0.0001,upper=10000,testval = 1000)
	    #rear tire longitudinal stiffness (N) - Same as front tire
	    Cxr = pm.Deterministic("Cxr",Cxf)
	    #nominal tire radius (m) - Easily measurable so not sampled
	#     r0 = pm.Deterministic("r0", 0.285)
	    r0 = 0.285
	    #front roll center distance below sprung mass c.g.
	    hrcf = pm.Uniform("hrcf",lower=0.0001,upper=20,testval = 0.5)
	    #rear roll center distance below sprung mass c.g.
	    hrcr = pm.Uniform("hrcr",lower=0.0001,upper=20,testval = 0.5)
	    #front roll stiffness (Nm/rad)
	    krof = pm.Uniform("krof",lower=0.0001,upper=100000,testval = 10000)
	    #rear roll stiffness (Nm/rad)
	    kror = pm.Uniform("kror",lower=0.0001,upper=100000,testval = 10000)
	    #front roll damping coefficient (Nm.s/rad)
	    brof = pm.Uniform("brof",lower=0,upper=10000,testval = 1000)
	    #rear roll damping coefficient (Nm.s/rad)
	    bror = pm.Uniform("bror",lower=0.0001,upper=10000,testval = 1000)



	    #We are also sampling our observation noise - Seems like a standard to use Half normal for this
	    sigma = pm.HalfNormal("sigma",sigma = 0.6,testval=0.1)


	    ## All of these will be a tensor 
	    theta = tt.as_tensor_variable([mass,Jx,Jy,Jz,a,b,Jxz,Jw,g,h,cf,cr,muf,mur,ktf,ktr,Cf,Cr,Cxf,Cxr,r0,hrcf,hrcr,krof,kror,brof,bror,sigma])

	    #According to this thread, we should use pm.Potential
	    #https://stackoverflow.com/questions/64267546/blackbox-likelihood-example
	    pm.Potential("like",like(theta))
	    theta_print = tt.printing.Print("theta")(theta)


	#Now we sample!
	with model:
		#We use metropolis as the algorithm with parameters to be sampled supplied through vars
		if(sys.argv[2] == "nuts"):
			step = pm.NUTS()
		elif(sys.argv[2] == "met"):
			step = pm.Metropolis()
		else:
			print("Please provide nuts or met as the stepping method")
		#We provide starting values for our parameters
		# start = pm.find_MAP()
		trace = pm.sample(ndraws,step=step, tune=nburn, discard_tuned_samples=True,return_inferencedata=False,cores=2)
		#Print the summary of all the parameters
		print(pm.summary(trace).to_string())
		df_sum = pm.summary(trace)
		#Save the trace in date directory
		pm.save_trace(directory = savedir + '_' + str(ndraws),trace=trace)
		#Save the stats in a csv file


	path = 'results/'
	if(os.path.isdir(path)):
		df_sum.to_csv('./results/'+savedir+'_stats_'+str(ndraws)+'.csv')
	else:
		os.mkdir(path)
		df_sum.to_csv('./results/'+savedir+'_stats_'+str(ndraws)+'.csv')


if __name__ == "__main__":
    main()


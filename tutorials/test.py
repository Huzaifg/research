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
import time
#Import our 8dof vehicle model
from vd_8dof_mod import vehicle_8dof




def add_noise(a):
	return a - np.random.normal(loc = 0., scale = abs(a.mean()/5),size = a.shape)





#This is just a gaussian log likelihood - Right now our data is just a vector so its almost the same as the previous example
def loglike(theta,time_o,st_inp,init_cond,data):
	sigma_vec = theta[-(data.shape[0]):]

	mod_data = vehicle_8dof(theta,time_o,st_inp,init_cond)
	# Using only the lateeral acceleration and the lateral veclocity
	mod_data = mod_data[[1,3],:]

	# Calculate the difference
	res = (mod_data - data)
	# print(res.shape)
 
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


	return sp.optimize.approx_fprime(theta,ll,delx*np.ones(len(theta)),time_o,st_inp,init_cond,data)





# datafile = 'vd_8dof_470.mat'
# vbdata = sio.loadmat(datafile)
# time_o = vbdata['tDash'].reshape(-1,)
# st_inp_o = vbdata['delta4'].reshape(-1,)
# # st_inp_rad = st_inp_o*np.pi/180
# lat_acc_o = vbdata['ay1'].reshape(-1,)
# lat_vel_o = vbdata['lat_vel'].reshape(-1,)
# long_vel_o = vbdata['long_vel'].reshape(-1,)
# roll_angle_o = vbdata['roll_angle'].reshape(-1,)
# yaw_rate_o = vbdata['yaw_rate'].reshape(-1,)
# psi_angle_o = vbdata['psi_angle'].reshape(-1,)

# # data = np.array([roll_angle_o,lat_acc_o,yaw_rate_o,lat_vel_o,psi_angle_o,long_vel_o])
# data = np.array([lat_vel_o,lat_acc_o])

# noOutputs= data.shape[0]
# for i in range(noOutputs):
# 	data[i,:] = add_noise(data[i,:])
with open('vd_8dof_470.npy', 'rb') as f:
	data = np.load(f)
   
with open('time.npy', 'rb') as f:
	time_o = np.load(f)


with open('st_inp.npy', 'rb') as f:
	st_inp_o = np.load(f)

	
init_cond = { 'u' : 50./3.6, 'v' : 0 , 'u_dot' : 0, 'v_dot' : 0, 'phi' : 0, 'psi' : 0, 'dphi' : 0, 'dpsi' : 0, 'wx' : 0, 'wy' : 0,
'wz' : 0, 'wx_dot' : 0, 'wz_dot' : 0 }



Cf = -44000
Cr = -47000

sigmaVy = 0.003
sigmaLat_acc = 0.3



theta = [Cf,Cr,sigmaVy,sigmaLat_acc]


start = time.time()
# T = 100
# for i in range(0,T):
# 	mod_data = vehicle_8dof(theta,time_o,st_inp_o,init_cond)
# end = time.time()
# print({(end - start)/100})
# Using only the lateeral acceleration and the lateral veclocity

mod_data = vehicle_8dof(theta,time_o,st_inp_o,init_cond)
mpl.plot(time_o,mod_data[0,:],time_o,data[0,:])
mpl.show()
# loglike_ = loglike(theta,time_o,st_inp_o,init_cond,data)

# print(loglike_)





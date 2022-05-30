import numpy as np
import ctypes
import scipy.io as sio
from scipy.integrate import solve_ivp
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
from math import atan,cos,sin
import time
from vd_bi_mod import vehicle_bi



def main():


	vbdata = sio.loadmat('vd_14dof_11000_sin.mat')
	time_o = vbdata['tDash'].reshape(-1,)
	st_inp_o = vbdata['delta4'].reshape(-1,)
	# st_inp_rad = st_inp_o*np.pi/180
	lat_acc_o = vbdata['ay1'].reshape(-1,)
	lat_vel_o = vbdata['lat_vel'].reshape(-1,)
	roll_angle_o = vbdata['roll_angle'].reshape(-1,)
	yaw_rate_o = vbdata['yaw_rate'].reshape(-1,)
	psi_angle_o = vbdata['psi_angle'].reshape(-1,)

	##vehicle model parameters
	a=1.14  # distance of c.g. from front axle (m)
	b=1.4  # distance of c.g. from rear axle  (m)
	Cf=-44000*2 # front axle cornering stiffness (N/rad)
	Cr=-47000*2 # rear axle cornering stiffness (N/rad)
	Cxf=5000*2 # front axle longitudinal stiffness (N)
	Cxr=5000*2 # rear axle longitudinal stiffness (N)
	m=1720  # the mass of the vehicle (kg)
	Iz=2420 # yaw moment of inertia (kg.m^2)

	long_vel_o = vbdata['long_vel'].reshape(-1,)
	roll_angle_o = vbdata['roll_angle'].reshape(-1,)
	yaw_rate_o = vbdata['yaw_rate'].reshape(-1,)
	psi_angle_o = vbdata['psi_angle'].reshape(-1,)
	data = np.array([long_vel_o,lat_vel_o,psi_angle_o,yaw_rate_o,lat_acc_o])
	
	##vehicle model parameters
	a=1.14  # distance of c.g. from front axle (m)
	b=1  # distance of c.g. from rear axle  (m)
	# Cf=-81722.366 # front axle cornering stiffness (N/rad)
	Cf = -88000
	# Cr=-47000*2 # rear axle cornering stiffness (N/rad)
	Cr = Cf
	# Cxf=11756.200  # front axle longitudinal stiffness (N)
	Cxf = 10000
	# Cxr=5000*2 # rear axle longitudinal stiffness (N)
	Cxr = Cxf
	m=1720  # the mass of the vehicle (kg)
	Iz=2000 # yaw moment of inertia (kg.m^2)
	# Iz = 2499.813

	Rr=0.285 # wheel radius
	Jw=1*2  # wheel roll inertia

	theta = [a,b,Cf,Cr,Cxf,Cxr,m,Iz,Rr,Jw]
	# theta = [ 1.58794615e+00,  1.62050348e-03, -6.53825484e+04, -4.35553418e+04,
 #        5.75552493e+03,  4.85291941e+03,  1.79056946e+03,  4.14564226e+02,
 #        1.12652337e+00,  1.42970238e+00]

	# theta = [ 1.58794615e+00,  1.62050348e-03, -6.53825484e+04, -4.35553418e+04,
 #    5.75552493e+03,  4.85291941e+03,  1.79056946e+03,  4.14564226e+02,
 #    0.285 , 1.42970238e+00]

	init_cond = { 'Vy' : 0, 'Vx' : 50./3.6 , 'psi' : 0, 'psi_dot' : 0, 'Y' : 0, 'X' : 0}

	start = time.time()
	mod_data = vehicle_bi(theta,time_o,st_inp_o,init_cond)
	end = time.time()
	print(f"Time taken is {end-start}")




	rows  = mod_data.shape[0]
	# mpl.style.use('plot_style.mplstyle')
	mpl.figure(1,figsize = (8,8))
	for i in range(0,rows):
		### Compare the outputs with plots to validate
		mpl.plot(time_o,data[i,:],'r',time_o,mod_data[i,:],'k')
		mpl.title('Using upper limit of 95 credibility')
		mpl.xlabel('time (s)')
		mpl.ylabel('data[i,:]')
		mpl.legend(['14dof','2dof'])
		mpl.grid()
		mpl.savefig('./images/comp_t' + str(i) + '.png',facecolor = 'w')
		mpl.show()





if __name__ == "__main__":
	main()



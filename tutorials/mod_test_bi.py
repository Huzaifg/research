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

	vbdata = sio.loadmat('vd_14dof_470.mat')
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


	# ### Compare the outputs with plots to validate
	# mpl.style.use('plot_style.mplstyle')
	# mpl.figure(1,figsize = (8,8))
	# mpl.plot(time_o,yaw_rate,'r',time_o,yaw_rate_o,'k')
	# mpl.title('yaw rate vs time')
	# mpl.xlabel('time (s)')
	# mpl.ylabel('yaw rate (rad/s)')
	# mpl.legend(['yaw rate 14dof','yaw rate bi'])
	# mpl.grid()
	# mpl.savefig('./images/comp_14bi.png',facecolor = 'w')
	# mpl.show()




if __name__ == "__main__":
    main()


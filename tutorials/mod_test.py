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
from vd_8dof_mod import vehicle_8dof


def main():

	datafile = 'vd_8dof_47_2.mat'
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

	data = np.array([roll_angle_o,lat_acc_o,yaw_rate_o,lat_vel_o,psi_angle_o,long_vel_o])

	m=1400  # Sprung mass (kg)
	Jx=900  # Sprung mass roll inertia (kg.m^2)
	Jy=2000  # Sprung mass pitch inertia (kg.m^2)
	Jz=2420  # Sprung mass yaw inertia (kg.m^2)
	a=1.14  # Distance of sprung mass c.g. from front axle (m)
	b=1.4   # Distance of sprung mass c.g. from rear axle (m)
	Jxz=90  # Sprung mass XZ product of inertia
	Jw=1    #tire/wheel roll inertia kg.m^2
	g=9.8    # acceleration of gravity 
	h=0.75   # Sprung mass c.g. height (m)
	cf=1.5   # front track width (m)
	cr=1.5   # rear track width (m)
	muf=80     #front unsprung mass (kg)
	mur=80     #rear unsprung mass (kg)
	ktf=200000   #front tire stiffness (N/m)
	ktr=200000   #rear tire stiffness (N/m)
	Cf=-44000  #front tire cornering stiffness (N/rad)
	Cr=-47000   #rear tire cornering stiffness (N/rad)
	Cxf=5000  #front tire longitudinal stiffness (N)
	Cxr=5000  #rear tire longitudinal stiffness (N)
	r0=0.285  #nominal tire radius (m)
	hrcf=0.65  #front roll center distance below sprung mass c.g.
	hrcr=0.6   #rear roll center distance below sprung mass c.g.
	krof=29000  #front roll stiffness (Nm/rad)
	kror=29000  #rear roll stiffness (Nm/rad)
	brof=3000   #front roll damping coefficient (Nm.s/rad)
	bror=3000   #rear roll damping coefficient (Nm.s/rad)


	theta = [m,Jx,Jy,Jz,a,b,Jxz,Jw,g,h,cf,cr,muf,mur,ktf,ktr,Cf,Cr,Cxf,Cxr,r0,hrcf,hrcr,krof,kror,brof,bror]
	init_cond = { 'u' : 50./3.6, 'v' : 0 , 'u_dot' : 0, 'v_dot' : 0, 'phi' : 0, 'psi' : 0, 'dphi' : 0, 'dpsi' : 0, 'wx' : 0, 'wy' : 0,
	'wz' : 0, 'wx_dot' : 0, 'wz_dot' : 0 }
	start = time.time() 
	mod_data= vehicle_8dof(theta,time_o,st_inp_o,init_cond)
	end = time.time()
	print(end-start)
	# print(yaw)
	print(mod_data.shape)
	print(mod_data)

	rows  = mod_data.shape[0]
	for i in range(0,rows):
		### Compare the outputs with plots to validate
		mpl.plot(time_o,data[i,:],'r',time_o,mod_data[i,:],'k')
		mpl.title('Using upper limit of 95 credibility')
		mpl.xlabel('time (s)')
		mpl.ylabel('data[i,:]')
		mpl.legend(['14dof','2dof'])
		mpl.grid()
		# mpl.savefig('./images/comp_t' + str(i) + '.png',facecolor = 'w')
		mpl.show()








if __name__ == "__main__":
    main()
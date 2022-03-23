import numpy as np
import scipy.io as sio
from scipy.integrate import solve_ivp
import scipy as sp
import matplotlib.pyplot as mpl
import pymc3 as pm
import arviz as az
import pandas as pd
import os
from datetime import datetime
import sys
from math import atan,cos,sin
from vd_bi_mod import vehicle_bi



"""
Requires the following command line inputs

mod_data_dof : The dof of the model which was fitted (used as black box in mcmc)
filename : The filename of the inference data file generated after mcmc. DO NOT prepend the results subdirectory or the file extension
save : True/False - Whether the plots need to be saved to ./images
data_dof - The dof of the data model that this fit is being compared to
dataFileName - The .mat file that the fit is being comparaed to - Over here DO append the file extension .mat


Returns : 
Opens the plots on the screen
If save is true, then the plots will also be saved in the ./images subdirectory 
"""

#The noise function
def add_noise(a):
	return a - np.random.normal(loc = 0., scale = abs(a.max()/20),size = a.shape) 



# Function that plots comparison plots
def plot_comp_plots(time_o,data,mod_data,title,units,mod_data_dof,data_dof,savedir,save=False):
	for i in range(0,data.shape[0]):
		mpl.figure(figsize = (10,10))
		try:
			mpl.plot(time_o,data[i,:].reshape(-1,),time_o,mod_data[i,:].reshape(-1,))
			mpl.title(title[i],fontsize = 20)
			mpl.xlabel('Time (S)',fontsize = 12)
			mpl.ylabel(title[i] + ' ' + units[i],fontsize = 12)
			mpl.legend(['Noisy Data from ' + data_dof + ' model','Mean posterior fit with ' + mod_data_dof + ' model'],fontsize = 14)
			path = 'images/'
			if(os.path.isdir(path)):
				if(save):
					mpl.savefig('./images/' + title[i]+'_'+savedir+'_comp_plot.png')
			else:
				os.mkdir(path)
				if(save):
					mpl.savefig('./images/' + title[i]+'_'+savedir+'_comp_plot.png')
			mpl.show()
		except IndexError:
			print("Possibly the title is not updated")
			raise IndexError



def main():

	if(len(sys.argv) < 5):
		print("Please provide the dof of the model which was fitted, the inference data filename, the dof of the data you want to compare to,and the data file name and whether you want to save the plots (true or false)")
		return 


	mod_data_dof = sys.argv[1]
	filename = sys.argv[2]
	savedir = filename[4:]
	save = sys.argv[5].lower() == 'true'
	data_dof = sys.argv[3]
	# The vector of data to plot which will be used as the title
	title = ['Lateral Velocity','Yaw Angle','Yaw Rate','Lateral Acceleration']
	units = ['(m/s)','(rad)','(rad/s)','(m/s^2)']

	# The data reading and formating
	dataFileName = sys.argv[4]



	print(f"The command line arguments provided are {mod_data_dof = }, {filename = }, {save = },{ data_dof= }, {dataFileName = }")



	#Read the inference data file
	idata = az.from_netcdf('./results/' + filename + ".nc")

	##vehicle model parameters
	try:
		a=idata.posterior.mean()['a'] # distance of c.g. from front axle (m)
		b=idata.posterior.mean()['b']  # distance of c.g. from rear axle  (m)
		Cf = idata.posterior.mean()['Cf']
		Cr = idata.posterior.mean()['Cr']
		Cxf = 10000
		Cxr = 10000
		m=1720  # the mass of the vehicle (kg)
		Iz=2420 # yaw moment of inertia (kg.m^2)
		Rr=0.285 # wheel radius
		Jw=1*2  # wheel roll inertia
		sigmaLat_acc = 0.38
	except KeyError:
		print("Probably some parameters you thought is sampled is not sampled")
		raise KeyError

	theta = [a,b,Cf,Cr,Cxf,Cxr,m,Iz,Rr,Jw]
	init_cond = { 'Vy' : 0, 'Vx' : 50./3.6 , 'psi' : 0, 'psi_dot' : 0, 'Y' : 0, 'X' : 0}

	vbdata = sio.loadmat(dataFileName)

	time_o = vbdata['tDash'].reshape(-1,)
	st_inp_o = vbdata['delta4'].reshape(-1,)
	# st_inp_rad = st_inp_o*np.pi/180a
	lat_acc_o = vbdata['ay1'].reshape(-1,)
	lat_vel_o = vbdata['lat_vel'].reshape(-1,)
	yaw_rate_o = vbdata['yaw_rate'].reshape(-1,)
	psi_angle_o = vbdata['psi_angle'].reshape(-1,)

	data = np.array([lat_vel_o,psi_angle_o,yaw_rate_o,lat_acc_o])
	
	# Add all the noise
	noOutputs= data.shape[0]
	for i in range(noOutputs):
		data[i,:] = add_noise(data[i,:])


	# The prediction
	mod_data = vehicle_bi(theta,time_o,st_inp_o,init_cond)
	plot_comp_plots(time_o,data,mod_data,title,units,mod_data_dof,data_dof,savedir,save)
	if(save):
		print("Voila! File written to ./images")









if __name__ == "__main__":
	main()

import numpy as np
import scipy.io as sio
from scipy.integrate import solve_ivp
import scipy as sp
import matplotlib.pyplot as mpl
import pymc as pm
import arviz as az
import pandas as pd
import os
from datetime import datetime
import sys
from math import atan,cos,sin
from vd_bi_mod import vehicle_bi
from vd_8dof_mod import vehicle_8dof


"""
Requires the following command line inputs

mod_data_dof : The dof of the model which was fitted (used as black box in mcmc)
filename : The filename of the inference data file generated after mcmc. DO NOT prepend the results subdirectory or the file extension
data_dof - The dof of the data model that this fit is being compared to
dataFileName - The .mat file that the fit is being comparaed to - Over here DO append the file extension .mat
save : True/False - Whether the plots need to be saved to ./images


Returns : 
Opens the plots on the screen
If save is true, then the plots will also be saved in the ./images subdirectory 
"""

#The noise function
#Different for long velocity as its at such a different scale and the velocity itself drops so slowly
def long_vel_noise(a):
	return a - np.random.normal(loc = 0., scale = abs(a.mean()/200),size = a.shape)

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
			# mpl.legend(['Noisy Data from ' + data_dof + ' model','Mean posterior fit with ' + mod_data_dof + ' model'],fontsize = 14)
			mpl.legend(['Noisy Data from ' + data_dof + ' model','with mean posterior fit ' + mod_data_dof + ' model'],fontsize = 14)
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
	    if(mod_data_dof[0] == '2'):
	        a=1.14 # distance of c.g. from front axle (m)
	        b=1.4  # distance of c.g. from rear axle  (m)
	        Cf = idata.posterior.mean()['Cf']
	        # Cf = -88000
	        Cr = idata.posterior.mean()['Cr']
	        # Cr = -88000
	        Cxf = 10000
	        # Cxf = idata.posterior.mean()['Cxf']
	        Cxr = 10000
	        # Cxr = idata.posterior.mean()['Cxr']
	        # m=idata.posterior.mean()['m']  # the mass of the vehicle (kg)
	        m = 1720
	        # m = idata.posterior.mean()['m']
	        Iz=idata.posterior.mean()['Iz'] # yaw moment of inertia (kg.m^2)
	        Rr=0.285 # wheel radius
	        Jw=1*2  # wheel roll inertia
	        # theta = [a,b,Cf,Cr,Cxf,Cxr,m,Iz,Rr,Jw]
	        theta = [Cf,Cr,Iz]
	        init_cond = { 'Vy' : 0, 'Vx' : 50./3.6 , 'psi' : 0, 'psi_dot' : 0, 'Y' : 0, 'X' : 0}
	    elif(mod_data_dof[0] == '8'):
	        mass = 1400
	        # Sprung mass roll inertia (kg.m^2)
	        Jx = 900  # Sprung mass roll inertia (kg.m^2)
	        # Sprung mass pitch inertia (kg.m^2)
	        Jy  = 2000
	        # Sprung mass yaw inertia (kg.m^2)
	        Jz = 2420
	        # Distance of sprung mass c.g. from front axle (m)
	        a = 1.14
	        # Distance of sprung mass c.g. from rear axle (m)
	        b = 1.4
	        # Sprung mass XZ product of inertia
	        Jxz = 90
	        #tire/wheel roll inertia kg.m^2
	        Jw = 1
	        # acceleration of gravity
	        g = 9.8
	        # Sprung mass c.g. height (m)
	        h = 0.75
	        # front track width (m)
	        cf = 1.5
	        # rear track width (m)
	        cr = 1.5
	        #front unsprung mass (kg)
	        muf = 80
	        #rear unsprung mass (kg)
	        mur = 80
	        #front tire stiffness (N/m) - Over here we are assuming that all the tires are identical to reduce the number of parameters
	        ktf = 200000
	        #rear tire stiffness (N/m) - Since rear tire is identical to front tire, we supply it as a deterministic variable
	        ktr = 200000
	        #front tire cornering stiffness (N/rad) - Over here we are assuming that all the tires are identical to reduce the number of parameters
	        Cf = idata.posterior.mean()['Cf']
	        #rear tire stiffness (N/m) - Since rear tire is identical to front tire, we supply it as a deterministic variable
	        Cr = idata.posterior.mean()['Cr']
	        #front tire longitudinal stiffness (N)
	        Cxf = 5000
	        #rear tire longitudinal stiffness (N) - Same as front tire
	        Cxr = 5000
	        #nominal tire radius (m) - Easily measurable so not sampled
	        r0 = 0.285
	        #front roll center distance below sprung mass c.g.
	        hrcf = 0.65
	        #rear roll center distance below sprung mass c.g.
	        hrcr = 0.6
	        #front roll stiffness (Nm/rad)
	        krof = 29000
	        #rear roll stiffness (Nm/rad)
	        kror = 29000
	        #front roll damping coefficient (Nm.s/rad)
	        brof = 3000
	        #rear roll damping coefficient (Nm.s/rad)
	        bror = 3000
	        # theta = [mass,Jx,Jy,Jz,a,b,Jxz,Jw,g,h,cf,cr,muf,mur,ktf,ktr,Cf,Cr,Cxf,Cxr,r0,hrcf,hrcr,krof,kror,brof,bror]
	        theta = [Cf,Cr]
	        init_cond = { 'u' : 50./3.6, 'v' : 0 , 'u_dot' : 0, 'v_dot' : 0, 'phi' : 0, 'psi' : 0, 'dphi' : 0, 'dpsi' : 0, 'wx' : 0, 'wy' : 0,
	'wz' : 0, 'wx_dot' : 0, 'wz_dot' : 0 }
	        
	except KeyError:
	        print("Probably some parameters you thought is sampled is not sampled")
	        raise KeyError






	vbdata = sio.loadmat(dataFileName)

	time_o = vbdata['tDash'].reshape(-1,)
	st_inp_o = vbdata['delta4'].reshape(-1,)
	# st_inp_rad = st_inp_o*np.pi/180a
	lat_acc_o = vbdata['ay1'].reshape(-1,)
	lat_vel_o = vbdata['lat_vel'].reshape(-1,)
	long_vel_o = vbdata['long_vel'].reshape(-1,)
	yaw_rate_o = vbdata['yaw_rate'].reshape(-1,)
	psi_angle_o = vbdata['psi_angle'].reshape(-1,)




	data = np.array([lat_vel_o,psi_angle_o,yaw_rate_o,lat_acc_o])



	# Add all the noise
	noOutputs= data.shape[0]
	for i in range(noOutputs):
		data[i,:] = add_noise(data[i,:])


	# The prediction
	if(mod_data_dof[0] == '2'):
	    mod_data = vehicle_bi(theta,time_o,st_inp_o,init_cond)
	elif(mod_data_dof[0] == '8'):
	    mod_data = vehicle_8dof(theta,time_o,st_inp_o,init_cond)



	plot_comp_plots(time_o,data,mod_data,title,units,mod_data_dof,data_dof,savedir,save)
	if(save):
		print("Voila! File written to ./images")









if __name__ == "__main__":
	main()

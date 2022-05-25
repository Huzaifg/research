import numpy as np
import scipy.io as sio
import scipy as sp
import matplotlib.pyplot as mpl
import pandas as pd


def ramp_torque(t):
	return np.where(t<1,0,5.*(t-1))
def ramp_st2(t):
	return 2.4418*np.pi/180.*(t)


time  = np.arange(0,3.7,0.01)
tor = ramp_torque(time)



# vbdata = sio.loadmat("vd_14dof_470_sin.mat")
# long_vel_o = vbdata['long_vel'].reshape(-1,)
# lat_vel_o = vbdata['lat_vel'].reshape(-1,)
# yaw_angle_o = vbdata['yaw_angle'].reshape(-1,)
# roll_angle_o = vbdata['roll_angle'].reshape(-1,)
# roll_rate_o = vbdata['roll_rate'].reshape(-1,)
# yaw_rate_o = vbdata['yaw_rate'].reshape(-1,)
# wlf_o = vbdata['wlf_'].reshape(-1,)
# wlr_o = vbdata['wlr_'].reshape(-1,)
# wrf_o = vbdata['wrf_'].reshape(-1,)
# wrr_o = vbdata['wrr_'].reshape(-1,)



data = pd.read_csv("/home/huzaifa/build_chrono/bin/DEMO_OUTPUT/WHEELED_JSON/Calibration/calib_mod_step.csv",sep=',',header='infer')
long_vel_o = np.asarray(data['vx'][700:1070])
lat_vel_o = np.asarray(data['vy'][700:1070])
yaw_angle_o = np.asarray(data['yaw'][700:1070])
roll_angle_o = np.asarray(data['roll'][700:1070])
roll_rate_o = np.asarray(data['roll_rate'][700:1070])
yaw_rate_o = np.asarray(data['yaw_rate'][700:1070])
wlf_o = np.asarray(data['wlf'][700:1070])
wlr_o = np.asarray(data['wlr'][700:1070])
wrf_o = np.asarray(data['wrf'][700:1070])
wrr_o = np.asarray(data['wrr'][700:1070])



long_vel_o = long_vel_o + np.random.normal(loc = 0., scale = 0.01,size = long_vel_o.shape)
# lat_vel_o = lat_vel_o + np.random.normal(loc = 0., scale = 0.007,size = lat_vel_o.shape)
lat_vel_o = lat_vel_o + np.random.normal(loc = 0., scale = 0.01,size = lat_vel_o.shape)
# yaw_angle_o = yaw_angle_o + np.random.normal(loc = 0., scale = 0.008,size = yaw_angle_o.shape)
yaw_angle_o = yaw_angle_o + np.random.normal(loc = 0., scale = 0.01,size = yaw_angle_o.shape)
# roll_angle_o = roll_angle_o + np.random.normal(loc = 0., scale = 0.003,size = roll_angle_o.shape)
roll_angle_o = roll_angle_o + np.random.normal(loc = 0., scale = 0.003,size = roll_angle_o.shape)
# roll_rate_o = roll_rate_o + np.random.normal(loc = 0., scale = 0.001,size = roll_rate_o.shape)
roll_rate_o = roll_rate_o + np.random.normal(loc = 0., scale = 0.0065,size = roll_rate_o.shape)
yaw_rate_o = yaw_rate_o + np.random.normal(loc = 0., scale = 0.02,size = yaw_rate_o.shape)
wlf_o = wlf_o + np.random.normal(loc = 0., scale = 0.2,size = wlf_o.shape)
wlr_o = wlr_o + np.random.normal(loc = 0., scale = 0.2,size = wlr_o.shape)
wrf_o = wrf_o + np.random.normal(loc = 0., scale = 0.2,size = wrf_o.shape)
wrr_o = wrr_o + np.random.normal(loc = 0., scale = 0.2,size = wrr_o.shape)


# data3 = np.array([long_vel_o,lat_vel_o,yaw_angle_o,roll_angle_o,roll_rate_o,yaw_rate_o,wlf_o,wlr_o,wrf_o,wrr_o])
# data = np.array([wlf_o,wlr_o,wrf_o,wrr_o])
# data = np.stack([roll_angle_o,roll_rate_o],axis=0)
# print(data.shape)

# with open('vd_14dof_exp1.npy', 'rb') as f:
# 	data1 = np.load(f)

# with open('vd_14dof_470_exp2.npy', 'rb') as f:
# 	data2 = np.load(f)

# data = np.array([data1[0,:],data2[0,:],data2[1,:],data1[1,:]])
out = np.array([lat_vel_o,roll_angle_o,roll_rate_o,yaw_rate_o])
# print(data.shape)
# print(data3.shape)
# data = data[[0,1],:]
np.save("vd_chrono_step.npy",out)
# np.save("time_14dof_470.npy",time_o)
# np.save("st_inp_14dof_ramp_470.npy",st_inp_o)

time  = np.arange(0,3.7,0.01)
for i in range(0,len(out)):
	mpl.plot(time,out[i,:])
	mpl.show()




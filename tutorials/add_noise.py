import numpy as np
import scipy.io as sio
import scipy as sp
import matplotlib.pyplot as mpl
import pandas as pd


# def ramp_torque(t):
# 	return np.where(t<1,0,5.*(t-1))
# def ramp_st2(t):
# 	return 2.4418*np.pi/180.*(t)



# tor = ramp_torque(time)



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

# n1= 700
# n2 = 1070

n1 = 100
n2 = 1250



# The time duration of the simulation
st_time = 1.
end_time = 12.5
# st_time = 0.
# end_time = 3.7

# The times for which we want the simulation
time  = np.arange(st_time,end_time,0.01)


data = pd.read_csv("test_1.csv",sep=',',header='infer')
x_o = np.asarray(data['x'][n1:n2])
y_o = np.asarray(data['y'][n1:n2])
long_vel_o = np.asarray(data['vx'][n1:n2])
lat_vel_o = np.asarray(data['vy'][n1:n2])
yaw_angle_o = np.asarray(data['yaw'][n1:n2])
roll_angle_o = np.asarray(data['roll'][n1:n2])
roll_rate_o = np.asarray(data['roll_rate'][n1:n2])
yaw_rate_o = np.asarray(data['yaw_rate'][n1:n2])
wlf_o = np.asarray(data['wlf'][n1:n2])
wlr_o = np.asarray(data['wlr'][n1:n2])
wrf_o = np.asarray(data['wrf'][n1:n2])
wrr_o = np.asarray(data['wrr'][n1:n2])



traj = np.array([x_o,y_o]) + np.random.normal(loc = 0., scale = 0.5,size = np.array([x_o,y_o]).shape)
# y_o = y_o + np.random.normal(loc = 0., scale = 0.5,size = y_o.shape)
# long_vel_o = long_vel_o + np.random.normal(loc = 0., scale = 0.2,size = long_vel_o.shape)
long_vel_o = long_vel_o + np.random.normal(loc = 0., scale = 0.1,size = long_vel_o.shape)
# lat_vel_o = lat_vel_o + np.random.normal(loc = 0., scale = 0.007,size = lat_vel_o.shape)
lat_vel_o = lat_vel_o + np.random.normal(loc = 0., scale = 0.04,size = lat_vel_o.shape)
# yaw_angle_o = yaw_angle_o + np.random.normal(loc = 0., scale = 0.008,size = yaw_angle_o.shape)
yaw_angle_o = yaw_angle_o + np.random.normal(loc = 0., scale = 0.04,size = yaw_angle_o.shape)
# roll_angle_o = roll_angle_o + np.random.normal(loc = 0., scale = 0.003,size = roll_angle_o.shape)
roll_angle_o = roll_angle_o + np.random.normal(loc = 0., scale = 0.003,size = roll_angle_o.shape)
# roll_rate_o = roll_rate_o + np.random.normal(loc = 0., scale = 0.001,size = roll_rate_o.shape)
roll_rate_o = roll_rate_o + np.random.normal(loc = 0., scale = 0.006,size = roll_rate_o.shape)
yaw_rate_o = yaw_rate_o + np.random.normal(loc = 0., scale = 0.02,size = yaw_rate_o.shape)
wlf_o = wlf_o + np.random.normal(loc = 0., scale = 1,size = wlf_o.shape)
wlr_o = wlr_o + np.random.normal(loc = 0., scale = 1,size = wlr_o.shape)
wrf_o = wrf_o + np.random.normal(loc = 0., scale = 1,size = wrf_o.shape)
wrr_o = wrr_o + np.random.normal(loc = 0., scale = 1,size = wrr_o.shape)


out = np.array([x_o,y_o,long_vel_o,lat_vel_o,yaw_angle_o,roll_angle_o,roll_rate_o,yaw_rate_o,wlf_o,wlr_o,wrf_o,wrr_o])
# out = np.array([traj[0,:],traj[1,:],long_vel_o,lat_vel_o,yaw_angle_o,roll_angle_o,roll_rate_o,yaw_rate_o,wlf_o,wlr_o,wrf_o,wrr_o])
# data = np.array([wlf_o,wlr_o,wrf_o,wrr_o])
# data = np.stack([roll_angle_o,roll_rate_o],axis=0)
# print(data.shape)

# with open('vd_chrono_ramp_1.npy', 'rb') as f:
# 	data1 = np.load(f)


# out = np.array([data1[0,:],yaw_angle_o,data1[3,:]])

# with open('vd_14dof_470_exp2.npy', 'rb') as f:
# 	data2 = np.load(f)

# data = np.array([data1[0,:],data2[0,:],data2[1,:],data1[1,:]])
# out = np.array([long_vel_o,wlf_o,wlr_o])
# out = np.array([lat_vel_o,roll_angle_o,roll_rate_o,yaw_rate_o])
# out = np.array([long_vel_o])
# print(data.shape)
# print(data3.shape)
# data = data[[0,1],:]
np.save("vd_chrono_test_1.npy",out)
# np.save("time_14dof_470.npy",time_o)
# np.save("st_inp_14dof_ramp_470.npy",st_inp_o)

for i in range(0,len(out)):
	mpl.plot(time,out[i,:])
	mpl.show()

# mpl.plot(out[0,:],out[1,:])
# mpl.show()


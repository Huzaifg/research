from vd_class import vd_2dof,vd_8dof
from vd_8dof_mod import vehicle_8dof
import numpy as np
from scipy.integrate import solve_ivp,odeint
import matplotlib.pyplot as mpl
import time
import scipy.io as sio
import scipy as sp
# import jax.numpy as np
import numpy as np

# from jax.experimental.ode import odeint
from jax import grad,vmap,jit
from jax.test_util import check_grads



def ramp_st(t):
	return np.where(t < 1, 0,1.2*np.pi/180.*(t-1))

def ramp_st_nonv(t):
	if(t<1):
		return 0
	else:
		return 1.2*np.pi/180.*(t-1)

def sin_st(t):
	if(t<1):
		return 0
	else:
		return 0.5*np.pi/180*np.sin((1/3)*2.*np.pi*(t-1))

def zero_st(t):
	return 0
def step_torque(t):
	if(t<1):
		return 0
	else:
		return 80.

def const_torque(t):
	return 0



# #The vehicle parameters
# a=1.14 # distance of c.g. from front axle (m)
# b=1.4  # distance of c.g. from rear axle  (m)
# Cxf=20000. # front axle longitudinal stiffness (N)
# Cxr=20000. # rear axle longitudinal stiffness (N)
# Cr = -90000
# m=1720.  # the mass of the vehicle (kg)
# Rr=0.285 # wheel radius
# Jw=1*2.  # wheel roll inertia
# Iz = 2420
# parameters = {'a':a,'b':b,'Cxf':Cxf,'Cxr':Cxr,'m':m,'Rr':Rr,'Jw':Jw,'Iz':Iz,'Cr':Cr}

# #The Vehicle states
# wf = 50./(3.6 * 0.285) #Angular velocity of front wheel
# wr = 50./(3.6 * 0.285) #Angular velocity of rear wheel
# Vx = 50./3.6 #Longitudanal Velocity
# Vy = 0. #Lateral velocity
# yr = 0. #Yaw rate
# state = {'Vy':Vy,'Vx':Vx}
# # state = [Vy,Vx,yr,wf,wr]

vehicle = vd_8dof()


# # print("Cf" in vehicle.params)

vehicle.set_steering(sin_st)
vehicle.set_torque(const_torque)
# vehicle.update_params(Cf=-80000,Cr=-80000)
# print(vehicle)
# start = time.time()
# trials = 50
# for i in range(0,trials):
# 	vehicle.update_params(Cr=-44000)
# stop = time.time()
# print(f"parameter update - time {(stop - start)/trials}")
# vehicle.update_params(Cr=-44000)
# print(vehicle)


time_  = np.arange(0.,4.7,0.01)

with open('vd_14dof_exp3.npy', 'rb') as f:
	targets = np.load(f)

targets = targets[[0,1],:]

# def loglike(theta):
# 	#Evaluate the model
# 	# vehicle.update_params(krof=theta[0],kror=theta[1],brof=theta[2],bror=theta[3])
# 	mod = odeint(vehicle.jax_model,list(vehicle.states.values()),time_,theta,rtol = 1.e-4,atol = 1.e-8)
# 	#Only take the relevant ones
# 	mod = np.stack([mod[3],mod[4]])
# 	#Get the sigmas from theta
# 	sigmas = np.array(theta[-(targets.shape[0]):],float).reshape(-1,1)
# 	#Evaluate negetive of log likelihood
# 	return -np.sum(np.sum((mod - targets)**2/(2.*sigmas**2))/np.linalg.norm(targets,axis = 1))




def loglike(theta,data):
	vehicle = vd_8dof()
	# # print("Cf" in vehicle.params)

	vehicle.set_steering(zero_st)
	vehicle.set_torque(step_torque)
	vehicle.update_params(Cf=-42315,Cr=-44884,krof=29902,kror=30236,brof=2073,bror=2054)
	sigmas = np.array(theta[-(data.shape[0]):]).reshape(-1,1)
	#Need to update the parameters using the update params method
	vehicle.update_params(Cxf=theta[0],Cxr=theta[1])
	mod = solve_ivp(vehicle.model,t_span=[time_[0],time_[-1]],y0 = list(vehicle.states.values()),method = 'RK45',t_eval = time_,rtol = 10e-5,atol = 10e-8)
	vehicle.reset_state()
	#Using only lateral velocity and yaw rate to compare for now
	return -np.sum(np.sum((mod['y'][[6,7],:] - data)**2/(2.*sigmas**2))/np.linalg.norm(data,axis = 1))



# theta = [5.96416216e+03, 3.60318533e+03,5918.176020642408,4948.888491020448]
vehicle.update_params(Cf=-42315,Cr=-44884,krof=29902,kror=30236,brof=2073,bror=2054)
# for i in range(0,50):
# 	print(loglike(theta,targets))
# vehicle.update_params(Cxf=theta[0],Cxr=theta[1])
# print(vehicle)
# outs = solve_ivp(vehicle.model,t_span=[time_[0],time_[-1]],y0 = list(vehicle.states.values()),method = 'RK45',t_eval = time_,rtol = 10e-4,atol = 10e-8)
# grad_loglike = grad(loglike,argnums=list(range(2)))
# 
# grad_loglike(theta,targets)
# start = time.time()
# trials = 50
# for i in range(0,trials):
# 	loglike(theta,targets)
# stop = time.time()
# print(f"grad loglike - time {(stop - start)/trials}")

# start = time.time()
# trials = 1
# for i in range(0,trials):
# 	loglike(theta,targets)
# stop = time.time()
# print(f"loglike - time {(stop - start)/trials}")

# check_grads(loglike,order=1)
# print(grad_loglike(theta))
# print(loglike(theta))


# print(vehicle.model(time))


# # print(np.array(vehicle.states.values()))


# start = time.time()
# trials = 1
# for i in range(0,trials):
# 	outs = solve_ivp(vehicle.model,t_span=[time_[0],time_[-1]],y0 = list(vehicle.states.values()),method = 'RK45',t_eval = time_,rtol = 10e-5,atol = 10e-8)
# stop = time.time()
# print(f"New model - time {(stop - start)/trials}")


# outs = odeint(vehicle.jax_model,list(vehicle.states.values()),time_,rtol = 1.e-4,atol = 1.e-8)
# print(onp.asarray(outs)[[3,4],:])

# print(targets)


# print(outs['y'][[1,5],:].shape)


# datafile = 'vd_14dof_470.mat'
# vbdata = sio.loadmat(datafile)
# lat_vel_o = vbdata['lat_vel'].reshape(-1,)
# yaw_rate_o = vbdata['yaw_rate'].reshape(-1,)
# # time_o = jnp.asarray(vbdata['tDash'].reshape(-1,),float)
# time_o = vbdata['tDash'].reshape(-1,)


# #Stack the data
# target = np.stack([lat_vel_o,yaw_rate_o],axis=0)

# print(np.linalg.norm(target,axis = 1))
# outs = vehicle.solve(package = 'odeint',t_eval = time,rtol = 10e-4,atol = 10e-8)
# # print(outs['y'].shape)

# print(np.linalg.norm(outs['y'][[1,5],:],axis = 1))
# for i in range(0,outs['y'].shape[0]):
# 	mpl.plot(time_,outs['y'][i,:])
# 	mpl.show()

# for i in range(0,outs.shape[1]):
# 	mpl.plot(time,outs[:,i])
# 	mpl.show()





# def ramp_st_2(t):
# 	return np.where(t > 1, 1.2*np.pi/180.*(t-1) ,0)

# Jx = 900
# Jy  = 2000
# Jz = 2420
# a = 1.14
# b = 1.4
# Jxz = 90
# Jw = 1
# g = 9.8
# h = 0.75
# cf = 1.5
# cr = 1.5
# muf = 80
# mur = 80
# ktf = 200000
# ktr = 200000
# Cf = -44000
# Cr = -47000
# Cxf = 5000
# Cxr = 5000
# r0 = 0.285
# hrcf = 0.65
# hrcr = 0.6
# krof = 29000
# kror = 29000
# brof = 3000
# bror = 3000

# theta = [Cf,Cr,Jz]
# time_o  = np.arange(0,4.7,0.01)
# st_inp = ramp_st_2(time_o)
# init_cond = { 'u' : 50./3.6, 'v' : 0 , 'u_dot' : 0, 'v_dot' : 0, 'phi' : 0, 'psi' : 0, 'dphi' : 0, 'dpsi' : 0, 'wx' : 0, 'wy' : 0,
# 'wz' : 0, 'wx_dot' : 0, 'wz_dot' : 0 }
# mod_data = vehicle_8dof(theta,time_o,st_inp,init_cond)
# start = time.time()
# trials = 50
# for i in range(0,trials):
# 	mod_data = vehicle_8dof(theta,time_o,st_inp,init_cond)
# stop = time.time()
# print(f"old model - time {(stop - start)/trials}")
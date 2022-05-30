from vd_class import vd_2dof,vd_8dof
from vd_8dof_mod import vehicle_8dof
import numpy as np
from scipy.integrate import solve_ivp,odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as mpl
import time
import scipy.io as sio
import scipy as sp
# import jax.numpy as np
import numpy as np
import pandas as pd
from point import Point

data = pd.read_csv("/home/huzaifa/build_chrono/bin/DEMO_OUTPUT/WHEELED_JSON/Calibration/calib_mod_simp_acc.csv",sep=',',header='infer')


def ramp_st2(t):
	return 2.4418*np.pi/180.*(t)
def ramp_st(t):
	return np.where(t < 1, 0.0104*np.pi/180.*(t),2.4418*np.pi/180.*(t-1))


def step_st(t):
	return -data['toe_in_r'][800] + 0*(t)

def ramp_st_nonv(t):
	if(t<1):
		return 0.0104*np.pi/180.*(t)
	else:
		return 2.4418*np.pi/180.*(t-1)

def sin_stv(t):
	return np.where(t < 1, 0,0.5*np.pi/180*np.sin((1/3)*2.*np.pi*(t-1)))

def sin_st(t):
	if(t<1):
		return 0
	else:
		return 0.5*np.pi/180*np.sin((1/3)*2.*np.pi*(t-1))

def zero_st(t):
	return 0 *t
def step_torque(t):
	if(t<1):
		return 0
	else:
		return 80.

def const_torque(t):
	return 50

def zero_throt(t):
	return 0 * t



n1  = 100
n2 = 850

st_time = 1.

time_  = np.arange(st_time,8.5,0.01)
# time_ = np.arange(0,3.7,0.01)


# pt1 = Point(0,-data['toe_in_avg'][n1])
# pt2 = Point(3.7,-data['toe_in_avg'][n2])
# ramp_st_3 = pt1.get_eq(pt2)

pt1 = Point(0,0)
pt2 = Point(3.7,0.2)
ramp_st_3 = pt1.get_eq(pt2)

throt1 = Point(0.5,0)
throt2 = Point(4.5,1)
ramp_throt1 = throt1.get_eq(throt2)
throt3 = Point(4.5,1)
throt4 = Point(8.5,0)
ramp_throt2 = throt3.get_eq(throt4)

def ramp_throt(t):
	return np.where(t<4.5,ramp_throt1(t),ramp_throt2(t))

#Sin steering through linear interpolation
# sin_st = interp1d(time_,-data['toe_in_l'][n1:n2],kind='linear')

# mpl.plot(sin_st(time_))
# mpl.show()

# print(ramp_st_3(time_))


# steering_values = data['toe_in'][n1:n2]

vehicle = vd_8dof()
vehicle.start_time = st_time
vehicle.set_steering(zero_st)
# vehicle.set_steering(ramp_st_3)

# vehicle.set_torque(const_torque)
vehicle.set_throttle(ramp_throt)
# vehicle.set_throttle(zero_throt)



# mpl.plot(time_,ramp_st_3(time_))
# mpl.show()

print(f"Average max steer :{data['toe_in_avg'][n2]}\n Left max steer :{data['toe_in_l'][n2]}\n Right max steer :{data['toe_in_r'][n2]} ")

vehicle.update_states(x = data['x'][n1],y=data['y'][n1],u=data['vx'][n1],v=data['vy'][n1],psi=data['yaw'][n1],phi=data['roll'][n1],wx=data['roll_rate'][n1],wz=data['yaw_rate'][n1],
	wlf = data['wlf'][n1],wlr = data['wlr'][n1],wrf = data['wrf'][n1],wrr = data['wrr'][n1])
print(vehicle)

# print(data['time'][n1])
# vehicle.xtrf = data['tiredef_rf'][n1]
# vehicle.xtlf = data['tiredef_lf'][n1]
# vehicle.xtrr = data['tiredef_rr'][n1]
# vehicle.xtlr = data['tiredef_lr'][n1]

# vehicle.xtrf = 0.018
# vehicle.xtlf = 0.018
# vehicle.xtrr = 0.018
# vehicle.xtlr = 0.018
# vehicle.update_params(m=1400*1.319,muf=80*1.319,mur=80*1.319,a= 1.6889,b =1.6889,h = 0.213,cf = 1.82,cr = 1.82,Jx = 1137.79,Jz = 4523.3,Jxz = 2.379,Cf=-50000,Cr=-50000,r0=0.47,
# 	ktf=326332,ktr=326332,krof=167062.0,kror=167062.0,brof=n168.000,bror=n168.000)
# vehicle.update_params(m=1400*1.319,muf=80*1.319,mur=80*1.319,a= 1.6889,b =1.6889,h = 0.213,cf = 1.82,cr = 1.82,Jx = 1294.91,Jz = 4290.7,Jxz = 1.7382,Cf=-50000,Cr=-50000,r0=0.47,
# 	ktf=326332,ktr=326332,krof=167062.0,kror=167062.0,brof=n168.000,bror=n168.000,hrcf = 0.6,hrcr=0.65)

#0.379
#0.327


atol_ar = [10e-6,10e-6,10e-5,10e-8,10e-9,10e-9,10e-6,10e-9,10e-5,10e-5,10e-5,10e-5]

vehicle.update_params(m=2097.85,muf=127.866,mur=129.98,a= 1.6889,b =1.6889,h = 0.713,cf = 1.82,cr = 1.82,Jx = 1289,Jz = 4519,Jxz = 3.265,Cf=39000,Cr=48000,r0=0.47,
	ktf=326332,ktr=326332,krof=31000.0,kror=31000.0,brof=3300.000,bror=3300.000,hrcf=0.379,hrcr=0.327,Jw=7,Cxf = 100000,Cxr = 100000,rr=0.0125)
# vehicle.update_states(u=data['vx'][n1])

print(vehicle)
	

start = time.time()
n = 1
for i in range(0,n):

	outs = solve_ivp(vehicle.model_tr,t_span=[time_[0],time_[-1]],y0 = list(vehicle.states.values()),method = 'RK45',t_eval = time_,
	rtol = 10e-4,atol = atol_ar)
	vehicle.reset_state()

stop = time.time()
print(f"time : {(stop-start)/n}")

start = time.time()
n = 0
for i in range(0,n):

	outs = solve_ivp(vehicle.model,t_span=[time_[0],time_[-1]],y0 = list(vehicle.states.values()),method = 'RK45',t_eval = time_,rtol = 10e-4,atol = 10e-9)
	vehicle.reset_state()

stop = time.time()
# print(f"time2 : {(stop-start)/n}")

print(f"Number of Evaluations of right hand side = {outs['nfev']}")
print(f"Number of evaluations of the Jacobian = {outs['njev']}")
print(f"Number of LU Decompositions = {outs['nlu']}")
plot = True
# mpl.plot(time_,vehicle.xtrf_ar,'k',time_,data['tiredef_rf'][n1:n2],'b')
mpl.plot(vehicle.xtrf_ar)
mpl.ylabel("Tire deflection (m)")
mpl.xlabel("solver iteration")
mpl.savefig('images/tiredef8_c',facecolor='w')
mpl.show()
mpl.plot(time_,data['tiredef_lf'][n1:n2],'b')
mpl.show()
	# mpl.savefig('images/traj_sin',facecolor='w')
if(plot):
	mpl.figure(figsize=(10,10))

	mpl.plot(time_,ramp_st_3(time_)*vehicle.max_steer)
	mpl.xlabel('Time (s)')
	mpl.ylabel('Steering angle (rad) ')
	mpl.title('Time vs Steering Angle')
	mpl.savefig('images/st_c',facecolor='w')
	mpl.show()


	# mpl.plot(time_,outs['y'][3,:])
	# mpl.xlabel('Time (s)')
	# mpl.ylabel('Lateral Velocity')
	# mpl.title('Time vs Lateral Velocity')
	# # mpl.savefig('images/sin_st',facecolor='w')
	# mpl.show()


	#Compare with the chrono model


	mpl.figure(figsize=(10,10))
	#Trajectory comparision
	mpl.plot(outs['y'][0,:],outs['y'][1,:],'k',data['x'][n1:n2],data['y'][n1:n2],'b')
	mpl.title("Vehicle Trajectory Comparision")
	mpl.xlabel("X (m)")
	mpl.ylabel("Y (m)")
	mpl.legend(['8dof','chrono'])
	mpl.savefig('images/traj_c',facecolor='w')
	mpl.show()

	mpl.figure(figsize=(10,10))
	#Lateral Velocity comparision
	mpl.plot(time_,outs['y'][3,:],'k',time_,data['vy'][n1:n2],'b')
	mpl.title("Lateral Velocity Comparision")
	mpl.xlabel("Time (s)")
	mpl.ylabel("Lateral Velocity (m/s)")
	mpl.legend(['8dof','chrono'])
	mpl.savefig('images/lat_vel_c',facecolor='w')
	mpl.show()

	mpl.figure(figsize=(10,10))
	#Longitudinal Velocity comparision
	mpl.plot(time_,outs['y'][2,:],'k',time_,data['vx'][n1:n2],'b')
	mpl.title("Longitudinal Velocity Comparision")
	mpl.xlabel("Time (s)")
	mpl.ylabel("Longitudinal Velocity (m/s)")
	mpl.legend(['8dof','chrono'])
	mpl.savefig('images/long_vel_c',facecolor='w')
	mpl.show()

	mpl.figure(figsize=(10,10))
	#Yaw Angle Comparision
	mpl.plot(time_,outs['y'][4,:],'k',time_,data['yaw'][n1:n2],'b')
	mpl.title("Yaw Angle Comparision")
	mpl.xlabel("Time (s)")
	mpl.ylabel("Yaw Angle (rad)")
	mpl.legend(['8dof','chrono'])
	mpl.savefig('images/yaw_ang_c',facecolor='w')
	mpl.show()

	mpl.figure(figsize=(10,10))
	#Roll Angle comparision
	mpl.plot(time_,outs['y'][5,:],'k',time_,data['roll'][n1:n2],'b')
	mpl.title("Roll Angle Comparision")
	mpl.xlabel("Time (s)")
	mpl.ylabel("Roll Angle (rad)")
	mpl.legend(['8dof','chrono'])
	mpl.savefig('images/roll_ang_c',facecolor='w')
	mpl.show()

	mpl.figure(figsize=(10,10))
	#Yaw Rate comparision
	mpl.plot(time_,outs['y'][7,:],'k',time_,data['yaw_rate'][n1:n2],'b')
	mpl.title("Yaw Rate comparision")
	mpl.xlabel("Time (s)")
	mpl.ylabel("Yaw Rate (rad/s)")
	mpl.legend(['8dof','chrono'])
	mpl.savefig('images/yaw_rate_c',facecolor='w')
	mpl.show()

	mpl.figure(figsize=(10,10))
	#Roll rate comparision
	mpl.plot(time_,outs['y'][6,:],'k',time_,data['roll_rate'][n1:n2],'b')
	mpl.title("Roll Rate comparision")
	mpl.xlabel("Time (s)")
	mpl.ylabel("Roll Rate (rad/s)")
	mpl.legend(['8dof','chrono'])
	mpl.savefig('images/roll_rate_c',facecolor='w')
	mpl.show()
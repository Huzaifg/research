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


def ramp_st(t):
	return np.where(t < 1, 0,1.2*np.pi/180.*(t-1))


def step_st(t):
	return np.where(t < 1, 0,-0.0000005)

def ramp_st_nonv(t):
	if(t<1):
		return 0
	else:
		return 0.5*np.pi/180.*(t-1)

def sin_stv(t):
	return np.where(t < 1, 0,0.5*np.pi/180*np.sin((1/3)*2.*np.pi*(t-1)))

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



vehicle = vd_8dof()
vehicle.set_steering(sin_st)
vehicle.set_torque(const_torque)
vehicle.update_params(Cf=-44000,Cr=-47000,krof=29902,kror=30236,brof=2073,bror=2054)


time_  = np.arange(0.,4.7,0.01)

outs = solve_ivp(vehicle.model,t_span=[time_[0],time_[-1]],y0 = list(vehicle.states.values()),method = 'RK45',t_eval = time_,rtol = 10e-4,atol = 10e-8)



mpl.plot(outs['y'][0,:],outs['y'][1,:])
mpl.xlabel('X (m)')
mpl.ylabel('Y (m)')
mpl.title('Vehicle Trajectory for ramp steer of 1.2 deg/s')
mpl.savefig('images/traj_sin',facecolor='w')
mpl.show()

mpl.plot(time_,sin_stv(time_))
mpl.xlabel('Time (s)')
mpl.ylabel('Steering angle (rad) ')
mpl.title('Time vs Steering Angle')
mpl.savefig('images/sin_st',facecolor='w')
mpl.show()

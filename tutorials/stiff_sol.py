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
import scipy.io



from scipy.integrate._ivp.base import OdeSolver
from numpy import linalg as LA

old_step = OdeSolver.step

jacobs = []
orders = []
times = []

# def new_step(self):
#     old_step(self)

#     times.append(self.t)
#     jacobs.append(self.J) 
#     # orders.append(self.order)

# OdeSolver.step = new_step



## Data
data = pd.read_csv("brake_2.csv",sep=',',header='infer')




def zero_throt(t):
	return 0 * t


def zero_st(t):
	return 0 *t

def zero_brake(t):
    return 0 * t

# n1  = 100
# n2 = 850
# n1 = 560
# n2 = 1060
# n1 = 700
# n2 = 1070
n1 = 850
n2 = 1250


# st_time = 1.0
# end_time = 8.5
# st_time = 0.
# end_time = 5.
# st_time = 0.
# end_time = 3.7
st_time = 8.5
end_time = 12.5
t_eval  = np.arange(st_time,end_time,0.01)

# print(data['time'][n1:n2])
# print(t_eval)
# print(len(t_eval))
# print(t_eval.shape)
# time_ = np.arange(0,3.7,0.01)


# pt1 = Point(0,-data['toe_in_avg'][n1])
# pt2 = Point(3.7,-data['toe_in_avg'][n2])
# ramp_st_3 = pt1.get_eq(pt2)

pt1 = Point(0,0)
pt2 = Point(3.7,0.2)
ramp_st_3 = pt1.get_eq(pt2)



throt1 = Point(0.5,0)
throt2 = Point(4.5,0.5)
ramp_throt1 = throt1.get_eq(throt2)
throt3 = Point(4.5,0.5)
throt4 = Point(8.5,0)
ramp_throt2 = throt3.get_eq(throt4)

def ramp_throt(t):
        if(t<4.5):
            return ramp_throt1(t)
        else:
            return ramp_throt2(t)


brake1 = Point(8.5,0)
brake2 = Point(12.5,0.2)
brake_tor1 = brake1.get_eq(brake2)

# mpl.plot(ramp_throt(time_))
# mpl.show()
# Constructor to initialise the model
st = {'x' : data['x'][n1]-data['x'][n1],'y':data['y'][n1]-data['y'][n1],'u':data['vx'][n1],'v':data['vy'][n1],'psi':data['yaw'][n1],
'phi':data['roll'][n1],'wx':data['roll_rate'][n1],'wz':data['yaw_rate'][n1],
	'wlf' : data['wlf'][n1],'wlr' : data['wlr'][n1],'wrf' : data['wrf'][n1],'wrr' : data['wrr'][n1]}
vehicle = vd_8dof(states = st)
# vehicle.set_steering(ramp_st_3)
vehicle.set_steering(zero_st)
steer = True

# vehicle.set_torque(const_torque)
# vehicle.set_throttle(ramp_throt)
vehicle.set_throttle(zero_throt)
# vehicle.set_torque(tor)

vehicle.set_braking(brake_tor1)


# vehicle.update_states(x = data['x'][n1],y=0.,u=data['vx'][n1],v=0.,psi=0.,phi=0.,
# wx=0.,wz=0.,
# 	wlf = data['wlf'][n1],wlr = data['wlr'][n1],wrf = data['wrf'][n1],wrr = data['wrr'][n1])

# print(vehicle)

## Array of absolute tolerances
rtol_ar = [10e-2,10e-2,10e-2,10e-2,10e-2,10e-2,10e-2,10e-2,10e-8,10e-8,10e-8,10e-8]
atol_ar = [10e-3,10e-3,10e-4,10e-4,10e-4,10e-4,10e-4,10e-4,10e-10,10e-10,10e-10,10e-10]
vehicle.update_params(m=2097.85,muf=127.866,mur=129.98,a= 1.6889,b =1.6889,h = 0.713,cf = 1.82,cr = 1.82,Jx = 1289,Jz = 4519,
Jxz = 3.265,Cf=39000,Cr=48000,r0=0.47,
	ktf=326332,ktr=326332,krof=31000.0,kror=31000.0,brof=3300.000,bror=3300.000,hrcf=0.379,hrcr=0.327,Jw=11,Cxf = 17481,Cxr = 17700,rr=0.0130)


# print(data['time'][n1],data['time'][n2])
# print(t_eval[0],t_eval[-1])

vehicle.debug = 1

start = time.time()
n = 1
for i in range(0,n):

	# outs = solve_ivp(vehicle.model_tr,t_span=[time_[0],time_[-1]],y0 = list(vehicle.states.values()),method = 'RK45',t_eval = time_,
	# rtol = rtol_ar,atol = atol_ar)
	# vehicle.reset_state()
    outs = vehicle.solve_half_impl(t_span = [t_eval[0],t_eval[-1]],t_eval = t_eval,tbar = 5e-3)
    #Reset the state if running multiple simulations
    # vehicle.reset_state(init_state=st)


stop = time.time()
print(f"time : {(stop-start)/n}")
# print(f"Number of Evaluations of right hand side = {outs['nfev']}")
# print(f"Number of evaluations of the Jacobian = {outs['njev']}")
# print(f"Number of LU Decompositions = {outs['nlu']}")

# np.set_printoptions(suppress=True)
# print(f"Solver time : {times[-1]}")
# print(f"Jacobian on the last time step : {jacobs[-1]}")
# # print(f"The order just before failure is : {orders[-1]}")
# print(f"Eigen values of the jacobian before overflow : {LA.eig(jacobs[-1])[0]}")

# np.save('/home/huzaifa/research/tutorials/eigens/bdf_jacs_1.npy',jacobs[0],allow_pickle=True)
# np.save('/home/huzaifa/research/tutorials/eigens/bdf_jacs_l.npy',jacobs[-1],allow_pickle=True)
# scipy.io.savemat('/home/huzaifa/research/tutorials/eigens/bdf_jacs_1.mat', {'jacob_1': jacobs[0]})
# scipy.io.savemat('/home/huzaifa/research/tutorials/eigens/bdf_jacs_l.mat', {'jacob_l': jacobs[-1]})

plot = True

if(plot):
    if(vehicle.debug):
        mpl.plot(vehicle.t_arr,vehicle.xtrf_ar)
        mpl.ylabel("8 dof Tire deflection (m)")
        mpl.xlabel("Time (s)")
        # mpl.savefig("images/8dof_td_hi54.png",facecolor = 'w')
        mpl.show()
        

        mpl.plot(t_eval,data['tiredef_lf'][n1:n2],'b')
        mpl.ylabel("Chrono tire deflection")
        mpl.xlabel("Time(s)")
        
        mpl.show()

        mpl.plot(vehicle.t_arr,vehicle.s_arr)
        mpl.ylabel("8 dof Longitudinal slip")
        mpl.xlabel("Time (s)")
        # mpl.savefig("images/8dof_ls_hi54.png",facecolor = 'w')
        mpl.show()

        
        
        mpl.plot(t_eval,data['long_slip'][n1:n2],'b')
        mpl.ylabel("Chrono longitudinal slip")
        mpl.xlabel("Time (S)")
        mpl.show()


        # mpl.plot(vehicle.t_arr,vehicle.dt,vehicle.t_arr,vehicle.fdt,vehicle.t_arr,vehicle.rdt,t_eval,data['sp_tor'][n1:n2])
        # mpl.ylabel("8 dof Torques")
        # mpl.legend(['drive','traction','rolling resistance','chrono spindle torque'])
        # mpl.xlabel("Time")
        # # mpl.savefig("images/8dof_tors_hi54.png",facecolor = 'w')
        # mpl.show()


        mpl.plot(vehicle.t_arr,vehicle.rdt)
        mpl.ylabel("8 dof Rolling resistance")
        mpl.xlabel("Time")
        mpl.show()

        mpl.plot(vehicle.t_arr,vehicle.flf,vehicle.t_arr,vehicle.flr,vehicle.t_arr,vehicle.frf,vehicle.t_arr,vehicle.frr)
        mpl.ylabel("8 dof Normal force")
        mpl.xlabel("Time (s)")
        mpl.legend(['lf','lr','rf','rr'])
        # mpl.savefig("images/8dof_nfs_hi54.png",facecolor = 'w')
        mpl.show()
        

    #Trajectory comparision
    mpl.figure(figsize=(10,10))
    mpl.plot(outs[:,0],outs[:,1],'k',data['x'][n1:n2]-data['x'][n1],data['y'][n1:n2]-data['y'][n1],'b')
    mpl.title("Vehicle Trajectory Comparision")
    mpl.xlabel("X (m)")
    mpl.ylabel("Y (m)")
    mpl.legend(['8dof','chrono'])
    mpl.show()


    # Wheel rotational velocity comparisions
    mpl.figure(figsize=(10,10))
    mpl.title("Wheel rotational velocity")
    mpl.plot(t_eval,data['wlf'][n1:n2],t_eval,data['wlr'][n1:n2],t_eval,outs[:,8],t_eval,outs[:,9])
    mpl.legend(['lf','lr','8 dof lf','8dof lr'])
    mpl.xlabel("Time (s)")
    mpl.ylabel("Angular velocity")
    # mpl.savefig("images/8dof_av_hi54.png",facecolor = 'w')
    mpl.show()




    if(steer):
        #Lateral Velocity comparision
        mpl.figure(figsize=(10,10))
        mpl.plot(t_eval,outs[:,3],'k',t_eval,data['vy'][n1:n2],'b')
        mpl.title("Lateral Velocity Comparision")
        mpl.xlabel("Time (s)")
        mpl.ylabel("Lateral Velocity (m/s)")
        mpl.legend(['8dof','chrono'])
        # mpl.savefig("images/8dof_lv_hi4.png",facecolor = 'w')
        mpl.show()


        #Yaw angle comparision
        mpl.figure(figsize=(10,10))
        mpl.plot(t_eval,outs[:,4],'k',t_eval,data['yaw'][n1:n2],'b')
        mpl.title("Yaw Angle Comparision")
        mpl.xlabel("Time (s)")
        mpl.ylabel("Yaw angle (rad)")
        mpl.legend(['8dof','chrono'])
        # mpl.savefig("images/8dof_lv_hi4.png",facecolor = 'w')
        mpl.show()


        #Roll angle comparision
        mpl.figure(figsize=(10,10))
        mpl.plot(t_eval,outs[:,5],'k',t_eval,data['roll'][n1:n2],'b')
        mpl.title("Roll Angle Comparision")
        mpl.xlabel("Time (s)")
        mpl.ylabel("Roll angle (rad)")
        mpl.legend(['8dof','chrono'])
        # mpl.savefig("images/8dof_lv_hi4.png",facecolor = 'w')
        mpl.show()

        #Yaw Rate comparision
        mpl.figure(figsize=(10,10))
        mpl.plot(t_eval,outs[:,7],'k',t_eval,data['yaw_rate'][n1:n2],'b')
        mpl.title("Yaw Rate Comparision")
        mpl.xlabel("Time (s)")
        mpl.ylabel("Yaw Rate (rad/s)")
        mpl.legend(['8dof','chrono'])
        # mpl.savefig("images/8dof_lv_hi4.png",facecolor = 'w')
        mpl.show()

        #Roll rate comparision
        mpl.figure(figsize=(10,10))
        mpl.plot(t_eval,outs[:,6],'k',t_eval,data['roll_rate'][n1:n2],'b')
        mpl.title("Roll rate Comparision")
        mpl.xlabel("Time (s)")
        mpl.ylabel("Roll rate (rad/s)")
        mpl.legend(['8dof','chrono'])
        # mpl.savefig("images/8dof_lv_hi4.png",facecolor = 'w')
        mpl.show()

        #Longitudinal Velocity comparision
        mpl.figure(figsize=(10,10))
        mpl.plot(t_eval,outs[:,2],'k',t_eval,data['vx'][n1:n2],'b')
        mpl.title("Longitudinal Velocity Comparision")
        mpl.xlabel("Time (s)")
        mpl.ylabel("Longitudinal Velocity (m/s)")
        mpl.legend(['8dof','chrono'])
        # mpl.savefig("images/8dof_lv_hi4.png",facecolor = 'w')
        mpl.show()


    else:
        #Just X trajectory
        mpl.figure(figsize=(10,10))
        mpl.plot(t_eval,outs[:,0],'k',t_eval,data['x'][n1:n2],'b')
        mpl.title("X comparison")
        mpl.xlabel("Time")
        mpl.ylabel("X (m)")
        mpl.legend(['8dof','chrono'])
        mpl.show()


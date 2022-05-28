
#

# BSD 3-Clause License

#

# Copyright (c) 2022 Simulation-Based Engineering Lab, University of Wisconsin - Madison

# All rights reserved.

#

# Redistribution and use in source and binary forms, with or without

# modification, are permitted provided that the following conditions are met:

#

# * Redistributions of source code must retain the above copyright notice, this

#   list of conditions and the following disclaimer.

#

# * Redistributions in binary form must reproduce the above copyright notice,

#   this list of conditions and the following disclaimer in the documentation

#   and/or other materials provided with the distribution.

#

# * Neither the name of the copyright holder nor the names of its

#   contributors may be used to endorse or promote products derived from

#   this software without specific prior written permission.

#

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"

# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE

# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE

# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE

# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL

# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR

# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER

# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,

# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE

# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.#




from vd_class import vd_2dof,vd_8dof
import matplotlib.pyplot as mpl
import time
import numpy as np
from point import Point
import pandas as pd

'''
Tutorial for a simple acceleration test where the vehicle is given a throttle from 0 to 0.5 from 0.5 s to 4.5 s and the throttle is then brought 
back down from 0.5 to 0 from 4.5 to 8.5 seconds. 
The initial state of the vehicle is obtained from a similar simulation on a chrono vehicle at 1 sec.
The simulation is started from 1 sec because the 8 dof model is incapable of starting from rest
'''


## Data - Here we use the chrono data to provide the initial states of the vehicle 
data = pd.read_csv("calib_mod_simp_acc_slow.csv",sep=',',header='infer')




# A zero steering function
def zero_st(t):
	return 0 *t



# Used for obtaining the state from the chrono vehicle
n1  = 100
n2 = 850


# The time duration of the simulation
st_time = 1.0
end_time = 8.5


# The times for which we want the simulation
time_  = np.arange(st_time,end_time,0.01)



# Now we construct the throttle function
throt1 = Point(0.5,0)
throt2 = Point(4.5,0.5)
ramp_throt1 = throt1.get_eq(throt2)
throt3 = Point(4.5,0.5)
throt4 = Point(8.5,0)
ramp_throt2 = throt3.get_eq(throt4)

def ramp_throt(t):
	return np.where(t<4.5,ramp_throt1(t),ramp_throt2(t))



# Constructor to initialise the model
vehicle = vd_8dof()

# Set the steering and the throttle functions we just created above
vehicle.set_steering(zero_st)
vehicle.set_throttle(ramp_throt)



# Update the initial state according to the chrono data file - if not provided, default states will be used
vehicle.update_states(x = data['x'][n1],y=data['y'][n1],u=data['vx'][n1],v=data['vy'][n1],psi=data['yaw'][n1],phi=data['roll'][n1],
wx=data['roll_rate'][n1],wz=data['yaw_rate'][n1],
	wlf = data['wlf'][n1],wlr = data['wlr'][n1],wrf = data['wrf'][n1],wrr = data['wrr'][n1])


# Provide the vehicle paramerters - if not provided default parametrs will be used
vehicle.update_params(m=2097.85,muf=127.866,mur=129.98,a= 1.6889,b =1.6889,h = 0.713,cf = 1.82,cr = 1.82,Jx = 1289,Jz = 4519,
Jxz = 3.265,Cf=39000,Cr=48000,r0=0.47,
	ktf=326332,ktr=326332,krof=31000.0,kror=31000.0,brof=3300.000,bror=3300.000,hrcf=0.379,hrcr=0.327,Jw=11,Cxf = 100000,Cxr = 100000,rr=0.0125)


## Array of tolerances to be used for the solver
rtol_ar = [10e-2,10e-2,10e-2,10e-2,10e-2,10e-2,10e-2,10e-2,10e-8,10e-8,10e-8,10e-8]
atol_ar = [10e-3,10e-3,10e-4,10e-4,10e-4,10e-4,10e-4,10e-4,10e-10,10e-10,10e-10,10e-10]

#To obtain additional debugging information
vehicle.debug = 1

start = time.time()
n = 1
for i in range(0,n):

	outs = vehicle.solve(t_eval = time_,method = 'RK45',rtol = rtol_ar,atol = atol_ar)
    #Reset the state if running multiple simulations
	# vehicle.reset_state()

stop = time.time()
print(f"time : {(stop-start)/n}")
print(f"Number of iterations {outs['nfev']}")

plot = True
if(plot):
    if(vehicle.debug):
        mpl.plot(vehicle.t_arr,vehicle.xtrf_ar)
        mpl.ylabel("8 dof Tire deflection (m)")
        mpl.xlabel("Time (s)")
        mpl.savefig("images/8dof_td_sci.png",facecolor = 'w')
        mpl.show()

        

        mpl.plot(time_,data['tiredef_lf'][n1:n2],'b')
        mpl.ylabel("Chrono tire deflection")
        mpl.xlabel("Time(s)")
        mpl.show()

        mpl.plot(vehicle.t_arr,vehicle.s_arr)
        mpl.ylabel("8 dof Longitudinal slip")
        mpl.xlabel("Time (s)")
        mpl.savefig("images/8dof_ls_sci.png",facecolor = 'w')
        mpl.show()


        
        
        mpl.plot(time_,data['long_slip'][n1:n2],'b')
        mpl.ylabel("Chrono longitudinal slip")
        mpl.xlabel("Time (S)")
        mpl.show()


        mpl.plot(vehicle.t_arr,vehicle.dt,vehicle.t_arr,vehicle.fdt,vehicle.t_arr,vehicle.rdt,time_,data['sp_tor'][n1:n2])
        mpl.ylabel("8 dof Torques")
        mpl.legend(['drive','traction','rolling resistance','chrono spindle torque'])
        mpl.xlabel("Time")
        mpl.savefig("images/8dof_tors_sci.png",facecolor = 'w')
        mpl.show()


        mpl.plot(vehicle.t_arr,vehicle.rdt)
        mpl.ylabel("8 dof Rolling resistance")
        mpl.xlabel("Time")
        mpl.show()

        mpl.plot(vehicle.t_arr,vehicle.flf,vehicle.t_arr,vehicle.flr,vehicle.t_arr,vehicle.frf,vehicle.t_arr,vehicle.frr)
        mpl.ylabel("8 dof Normal force")
        mpl.xlabel("Time (s)")
        mpl.legend(['lf','lr','rf','rr'])
        mpl.savefig("images/8dof_nf_sci.png",facecolor = 'w')
        mpl.show()
        

    #Trajectory comparision
    mpl.figure(figsize=(10,10))
    mpl.plot(outs['y'][0,:],outs['y'][1,:],'k',data['x'][n1:n2],data['y'][n1:n2],'b')
    mpl.title("Vehicle Trajectory Comparision")
    mpl.xlabel("X (m)")
    mpl.ylabel("Y (m)")
    mpl.legend(['8dof','chrono'])
    mpl.show()
    

    # Wheel rotational velocity comparisions
    mpl.figure(figsize=(10,10))
    mpl.title("Wheel rotational velocity")
    mpl.plot(data['time'][n1:n2],data['wlf'][n1:n2],data['time'][n1:n2],data['wlr'][n1:n2],time_,outs['y'][8,:],time_,outs['y'][10,:])
    mpl.legend(['lf','lr','8 dof lf','8dof rf'])
    mpl.xlabel("Time (s)")
    mpl.ylabel("Angular velocity")
    mpl.savefig("images/8dof_av_sci.png",facecolor = 'w')
    mpl.show()


    

    #Longitudinal Velocity comparision
    mpl.figure(figsize=(10,10))
    mpl.plot(time_,outs['y'][2,:],'k',time_,data['vx'][n1:n2],'b')
    mpl.title("Longitudinal Velocity Comparision")
    mpl.xlabel("Time (s)")
    mpl.ylabel("Longitudinal Velocity (m/s)")
    mpl.legend(['8dof','chrono'])
    mpl.savefig("images/8dof_lv_sci.png",facecolor = 'w')
    mpl.show()

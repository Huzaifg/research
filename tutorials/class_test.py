from vd_class import vd_2dof
import numpy as np
from scipy.integrate import solve_ivp




def ramp_st(t):
	return np.where(t > 1, 3.*np.pi/180.*(t-1) ,0)


#The vehicle parameters
a=1.14 # distance of c.g. from front axle (m)
b=1.4  # distance of c.g. from rear axle  (m)
Cxf=10000. # front axle longitudinal stiffness (N)
Cxr=10000. # rear axle longitudinal stiffness (N)
m=1720.  # the mass of the vehicle (kg)
Rr=0.285 # wheel radius
Jw=1*2.  # wheel roll inertia
parameters = {'a':a,'b':b,'Cxf':Cxf,'Cxr':Cxr,'m':m,'Rr':Rr,'Jw':Jw}

#The Vehicle states
wf = 50./(3.6 * 0.285) #Angular velocity of front wheel
wr = 50./(3.6 * 0.285) #Angular velocity of rear wheel
Vx = 50./3.6 #Longitudanal Velocity
Vy = 0. #Lateral velocity
yr = 0. #Yaw rate
state = {'Vy':Vy,'Vx':Vx}
# state = [Vy,Vx,yr,wf,wr]

vehicle = vd_2dof()


# print("Cf" in vehicle.params)

vehicle.set_steering(ramp_st)
vehicle.update_params(Cf=-100000,Cr=-100000)
vehicle.update_states(Vx = 20)


time  = np.arange(0,4.7,0.01)
# print(vehicle.model(time))


# print(np.array(vehicle.states.values()))
# outs = solve_ivp(vehicle.model,t_span=[time[0],time[-1]],y0 = list(vehicle.states.values()),method = 'RK45',t_eval = time,vectorized = True,rtol = 10e-3,atol = 10e-3)
outs = vehicle.solve(package = 'odeint',t_eval = time,rtol = 10e-3,atol = 10e-3)
print(outs)

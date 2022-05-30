#Created by Huzaifa Mustafa Unjhawala


import jax.numpy as jnp


def vehicle_bi(state,t,theta):


	"""
	The bicycle vehicle model that is our physical model for bayesian inference.
	This is adopted from the matlab script - https://github.com/uwsbel/projectlets/tree/master/model-repo/misc/2019/vehicles-matlab/8_DOF

	Parameters:
	----------

	theta : 
		A jax numpy of parameters of interest of the model. These are also all the parameters that are inferred from the bayesian inference. 

	tt :
		The time vector where the states are desired

	state:
		The initial state of the vehicle

	Returns:
	-------
	A jax numpy array with the ODE's of the system

	"""


	#Defining the ramp steer
	def ramp_st(t):
		return jnp.where(t > 1, 3.*jnp.pi/180.*(t-1) ,0)

	def sin_st(t):
		return jnp.where( t > 1, 0.5*jnp.pi/180*jnp.sin((1/3)*2.*jnp.pi*(t-1)),0)


	#The vehicle parameters
	a=1.14 # distance of c.g. from front axle (m)
	b=1.4  # distance of c.g. from rear axle  (m)
	Cf=theta[0] # front axle cornering stiffness (N/rad)
	Cr=theta[1] # rear axle cornering stiffness (N/rad)
	Cxf=10000. # front axle longitudinal stiffness (N)
	Cxr=10000. # rear axle longitudinal stiffness (N)
	m=1720.  # the mass of the vehicle (kg)
	Iz=theta[2] # yaw moment of inertia (kg.m^2)
	Rr=0.285 # wheel radius
	Jw=1*2.  # wheel roll inertia

	# State of the vehicle
	Vy=state[0] # lateral velocity 	
	Vx=state[1] # longitudinal velocity
	psi_dot=state[2] # yaw rate 
	wf = state[3] #Front wheel angular velocity 
	wr = state[4] #Rear wheel angular velocity



	#Some preliminaries used in the ODE's
	sf=(Rr*wf-(Vx*jnp.cos(sin_st(t))+(Vy+a*psi_dot)*jnp.sin(sin_st(t))))/jnp.abs(Vx*jnp.cos(sin_st(t))+(Vy+a*psi_dot)*jnp.sin(sin_st(t)))
	sr=(Rr*wr-Vx)/jnp.abs(Vx)
	Fxtf=Cxf*sf
	Fxtr=Cxr*sr



	#ODE's
	Vy_dot=-Vx*psi_dot+(1/m)*(Cf*((Vy+a*psi_dot)/Vx-sin_st(t))+Cr*((Vy-b*psi_dot)/Vx))
	Vx_dot=Vy*psi_dot+(sf*Cxf+sr*Cxr)/m-sin_st(t)*Cf*((Vy+a*psi_dot)/Vx-sin_st(t))/m
	dpsi_dot=1/Iz*(a*Cf*((Vy+a*psi_dot)/Vx-sin_st(t))-b*Cr*((Vy-b*psi_dot)/Vx))
	dwf=-(1/Jw)*Fxtf*Rr
	dwr=-(1/Jw)*Fxtr*Rr

	return jnp.stack([Vy_dot,Vx_dot,dpsi_dot,dwf,dwr])
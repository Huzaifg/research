import numpy as np
from math import atan,cos,sin

# import warnings
# warnings.filterwarnings("error")

def vehicle_bi(theta,tt,st_input,init_cond):
	"""
	The bicycle vehicle model that is our physical model for bayesian inference.
	This is adopted from the matlab script - https://github.com/uwsbel/projectlets/tree/master/model-repo/misc/2019/vehicles-matlab/8_DOF

	Parameters:
	----------

	theta : 
		A list of all the parameters of the model. These are also all the parameters that are inferred from the bayesian inference. 
		Need to find a way to pass a dict - list is too error prone
	tt :
		Time intervals at which the "data" is "collected"

	st_input:
		The steering input to the vehicle as a time series.

	init_cond:
		These are the initial conditions of the vehicle.

	Returns:
	-------
	mod_data :
        Matrix with size (no_of_outputs X no_of_timesteps) ---- ex. row 0 of mod_data is the longitudanal veclocity vector at each time step

	"""

	### Our model parameters - This time they are much more simple and there is no integration so hoepfully, no overflow errors
	
	a=theta[0]  # distance of c.g. from front axle (m)
	b=theta[1]  # distance of c.g. from rear axle  (m)
	Cf=theta[2] # front axle cornering stiffness (N/rad)
	Cr=theta[3] # rear axle cornering stiffness (N/rad)
	Cxf=theta[4] # front axle longitudinal stiffness (N)
	Cxr=theta[5] # rear axle longitudinal stiffness (N)
	m=theta[6]  # the mass of the vehicle (kg)
	Iz=theta[7] # yaw moment of inertia (kg.m^2)
	Rr=theta[8] # wheel radius
	Jw=theta[9] # wheel roll inertia


	# Supplied Initial Conditions

	Vy=init_cond['Vy'] # lateral velocity 	
	Vx=init_cond['Vx'] # longitudinal velocity
	psi=init_cond['psi'] # yaw anlge
	psi_dot=init_cond['psi_dot'] # yaw rate 
	Y=init_cond['Y'] # Y position in global coordinates
	X=init_cond['X'] # X position in global coordinates


	#Derived initial conditions

	wf=Vx/Rr; # front wheel rotation angular velocity
	wr=Vx/Rr; # rear wheel rotation angular velocity

	ts = tt[1] - tt[0] ##the time step
	size_tt = len(tt)
	# print(f"End time is {tt[-1]}")
	Tsim = ts*size_tt - ts #The end time
	T = ts #Since the code from matlab uses this variable for calculations



	#Steering input - from the data
	delta3 = st_input

	# Initialize the output
	Vy_,Vx_,psi_,psi_dot_,Y_,X_,lateral_acc_= (np.zeros(size_tt) for _ in range(7))


	for i,t in enumerate(tt):
		delta_r=delta3[i]
		#longitudinal slips ratio
		sf=(Rr*wf-(Vx*cos(delta_r)+(Vy+a*psi_dot)*sin(delta_r)))/abs(Vx*cos(delta_r)+(Vy+a*psi_dot)*sin(delta_r))

		sr=(Rr*wr-Vx)/abs(Vx)
		Fxtf=Cxf*sf
		Fxtr=Cxr*sr
		# the wheel rotational equation, assuming no braking torque and accelerating torque
		dwf=-(1/Jw)*Fxtf*Rr
		dwr=-(1/Jw)*Fxtr*Rr
		wf=wf+T*dwf
		wr=wr+T*dwr

		
		Vy_dot=-Vx*psi_dot+(1/m)*(Cf*((Vy+a*psi_dot)/Vx-delta_r)+Cr*((Vy-b*psi_dot)/Vx))
		Vx_dot=Vy*psi_dot+(sf*Cxf+sr*Cxr)/m-delta_r*Cf*((Vy+a*psi_dot)/Vx-delta_r)/m

		dpsi_dot=1/Iz*(a*Cf*((Vy+a*psi_dot)/Vx-delta_r)-b*Cr*((Vy-b*psi_dot)/Vx))
		Y_dot=Vx*sin(psi)+Vy*cos(psi)
		X_dot=Vx*cos(psi)-Vy*sin(psi)

		Vy=Vy+T*Vy_dot
		Vx=Vx+T*Vx_dot
		psi=psi+T*psi_dot
		psi_dot=psi_dot+T*dpsi_dot
		Y=Y+T*Y_dot
		X=X+T*X_dot

		### Store all the output responses
		Vy_[i]=Vy #Lateral Velocity
		Vx_[i]=Vx #Longitudanal Velocity
		psi_[i]=psi #Yaw angle
		psi_dot_[i]=psi_dot #Yaw rate
		Y_[i]=Y #Y position
		X_[i]=X #X position
		lateral_acc_[i]=Vy_dot+Vx*psi_dot #Lateral Acceleration




	# mod_data = np.array([Vx_,Vy_,psi_,psi_dot_,lateral_acc_])
	# mod_data = np.array([Vy_,psi_,psi_dot_,lateral_acc_])
	mod_data = np.array([Vy_,lateral_acc_])


	return mod_data







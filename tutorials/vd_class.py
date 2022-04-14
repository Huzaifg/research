import warnings
import numpy as np
import types
from scipy.integrate import solve_ivp 
from scipy.integrate import odeint

class vd_2dof:

	def __init__(self,parameters = None,states = None):
		if parameters is not None:
			if isinstance(parameters,list):
				raise Exception("Please provide a dictionary for the parameters")
			try:
				self.a= parameters['a'] # distance of c.g. from front axle (m)
			except:
				self.a = 1.14
				warnings.warn(f"Set 'a' to default value {self.a}",UserWarning)
			try:
				self.b= parameters['b']  # distance of c.g. from rear axle  (m)
			except:
				self.b = 1.4
				warnings.warn(f"Set 'b' to default value {self.b}",UserWarning)
			try:
				self.Cf= parameters['Cf']  # front axle cornering stiffness (N/rad)
			except:
				self.Cf = -88000
				warnings.warn(f"Set 'Cf' to default value {self.Cf}",UserWarning)
			try:
				self.Cr= parameters['Cr'] # rear axle cornering stiffness (N/rad)
			except:
				self.Cr = -88000
				warnings.warn(f"Set 'Cr' to default value {self.Cr}",UserWarning)
			try:
				self.Cxf= parameters['Cxf'] # front axle longitudinal stiffness (N)
			except:
				self.Cxf = 10000
				warnings.warn(f"Set 'Cxf' to default value {self.Cxf}",UserWarning)
			try:
				self.Cxr= parameters['Cxr'] # rear axle longitudinal stiffness (N)
			except:
				self.Cxr = 10000
				warnings.warn(f"Set 'Cxr' to default value {self.Cxr}",UserWarning)
			try:
				self.m= parameters['m']  # the mass of the vehicle (kg)
			except:
				self.m = 1720
				warnings.warn(f"Set 'm' to default value {self.m}",UserWarning)
			try:
				self.Iz= parameters['Iz'] # yaw moment of inertia (kg.m^2)
			except:
				self.Iz = 2420
				warnings.warn(f"Set 'Iz' to default value {self.Iz}",UserWarning)
			try:
				self.Rr = parameters['Rr'] # wheel radius
			except:
				self.Rr = 0.285 
				warnings.warn(f"Set 'Rr' to default value {self.Rr}",UserWarning)
			try:
				self.Jw= parameters['Jw']  # wheel roll inertia
			except:
				self.Jw = 2
				warnings.warn(f"Set 'Jw' to default value {self.Jw}",UserWarning)
			#A dictionary of parameters
			self.params = {"a": self.a,"b" : self.b,"Cf" : self.Cf,"Cr" : self.Cr,"Cxf": self.Cxf,"Cxr" : self.Cxr,"m" : self.m,"Iz" : self.Iz,"Rr" : self.Rr,"Jw" : self.Jw}
		else:
			self.a= 1.14# distance of c.g. from front axle (m)
			self.b= 1.4  # distance of c.g. from rear axle  (m)
			self.Cf= -88000  # front axle cornering stiffness (N/rad)
			self.Cr= -88000 # rear axle cornering stiffness (N/rad)
			self.Cxf= 10000 # front axle longitudinal stiffness (N)
			self.Cxr= 10000 # rear axle longitudinal stiffness (N)
			self.m= 1720  # the mass of the vehicle (kg)
			self.Iz= 2420 # yaw moment of inertia (kg.m^2)
			self.Rr= 0.285 # wheel radius
			self.Jw= 2  # wheel roll inertia
			#A dictionary of parameters
			self.params = {'a': self.a,'b' : self.b,'Cf' : self.Cf,'Cr' : self.Cr,'Cxf': self.Cxf,'Cxr' : self.Cxr,'m' : self.m,'Iz' : self.Iz,'Rr' : self.Rr,'Jw' : self.Jw}
			warnings.warn("Set parameters to default values" + '\n' + f"{self.params}" ,UserWarning)




		if states is not None:
			if isinstance(states,list):
				raise Exception("Please provide a dictionary for the states")

			# State of the vehicle
			try:
				self.Vy=states['Vy'] # lateral velocity 
			except:
				self.Vy = 0.
				warnings.warn(f"Set 'Vy' to default value {self.Vy}",UserWarning)
			try:
				self.Vx=states['Vx']
			except:
				self.Vx = 50./3.6
				warnings.warn(f"Set 'Vx' to default value {self.Vx}",UserWarning)
			try:
				self.psi_dot=states['psi_dot'] # yaw rate 
			except:
				self.psi_dot = 0.
				warnings.warn(f"Set 'psi_dot' to default value {self.psi_dot}",UserWarning)
			try:
				self.wf = states['wf'] #Front wheel angular velocity 
			except:
				self.wf = 50./(3.6 * 0.285)
				warnings.warn(f"Set 'wf' to default value {self.wf}",UserWarning)
			try:
				self.wr = states['wr'] #Rear wheel angular velocity
			except:
				self.wr = 50./(3.6 * 0.285)
				warnings.warn(f"Set 'wr' to default value {self.wr}",UserWarning)
			#A dictionary of states
			self.states = {'Vy' : self.Vy,'Vx' : self.Vx,'psi_dot' : self.psi_dot,'wf' : self.wf,'wr' : self.wr}
		else:
			# State of the vehicle
			self.Vy=0. # lateral velocity 	
			self.Vx=50./3.6 # longitudinal velocity
			self.psi_dot=0. # yaw rate 
			self.wf =  50./(3.6 * 0.285) #Front wheel angular velocity 
			self.wr = 50./(3.6 * 0.285) #Rear wheel angular velocity
			#A dictionary of states
			self.states = {'Vy' : self.Vy,'Vx' : self.Vx,'psi_dot' : self.psi_dot,'wf' : self.wf,'wr' : self.wr}
			warnings.warn("Set States to default values" + '\n' + f"{self.states}" ,UserWarning)

		


	#Simple print to show the parameters and states
	def __str__(self):
		return  str(self.__class__) + '\n' + "Vehicle Parameters are" + '\n' + f"{self.params}" + '\n' + "Vehicle state is" + '\n' + f"{self.states}"


	#Sets the controls for the vehicle
	def set_steering(self,steering):
		if(isinstance(steering, types.FunctionType)):
			self.steering = steering
		else:
			raise Exception(f"The controls provided should be of type {types.FunctionType}. It should also be a function of time")


	#Update the states
	def update_states(self,**kwargs):
		for key,value in kwargs.items():
			if(key not in self.states):
				raise Exception(f"{key} is not vehicle state")

			#Set the state to the respective class attribute
			setattr(self,key,value)
			#Update the states dict attribute as well
			self.states[key] = value




	#Update the parameters
	def update_params(self,**kwargs):
		for key,value in kwargs.items():
			if(key not in self.params):
				raise Exception(f"{key} is not vehicle parameter")

			#Set the parameter to the respective class attribute
			setattr(self,key,value)
			#Update the params dict of the vehicle as well
			self.params[key] = value
				


	#Returns a list of the parameters
	def get_params(self):
		return self.params


	#Returns a list of the current states
	def get_states(self):
		return self.states


	#returns the differential equations of the model - Private method for now
	def model(self,t,state):

		a,b,Cf,Cr,Cxf,Cxr,m,Iz,Rr,Jw =list(self.params.values())
		Vy,Vx,psi_dot,wf,wr = state

		#Some preliminaries used in the ODE's
		sf=(Rr*wf-(Vx*np.cos(self.steering(t))+(Vy+a*psi_dot)*np.sin(self.steering(t))))/np.abs(Vx*np.cos(self.steering(t))+(Vy+a*psi_dot)*np.sin(self.steering(t)))
		sr=(Rr*wr-Vx)/np.abs(Vx)
		Fxtf=Cxf*sf
		Fxtr=Cxr*sr



		#ODE's
		Vy_dot=-Vx*psi_dot+(1/m)*(Cf*((Vy+a*psi_dot)/Vx-self.steering(t))+Cr*((Vy-b*psi_dot)/Vx))
		Vx_dot=Vy*psi_dot+(sf*Cxf+sr*Cxr)/m-self.steering(t)*Cf*((Vy+a*psi_dot)/Vx-self.steering(t))/m
		dpsi_dot=1/Iz*(a*Cf*((Vy+a*psi_dot)/Vx-self.steering(t))-b*Cr*((Vy-b*psi_dot)/Vx))
		dwf=-(1/Jw)*Fxtf*Rr
		dwr=-(1/Jw)*Fxtr*Rr


		return np.stack([Vy_dot,Vx_dot,dpsi_dot,dwf,dwr])


	#Awrapper function for maybe a few packages - starting with solve_ivp,odeint
	def solve(self,package = 'solve_ivp',t_eval = None,**kwargs):
		try:
			self.steering
		except:
			raise Exception("Please provide steering controls for the vehicle with 'set_steering' method")

		if t_eval is None:
			raise Exception("Please provide times steps at which you want the solution to be evaluated")


		if(package ==  'solve_ivp'):
			return solve_ivp(self.model,t_span=[t_eval[0],t_eval[-1]],y0 = list(self.states.values()),vectorized = True,t_eval = t_eval,**kwargs)
		elif(package == 'odeint'):
			return odeint(self.model,y0 = list(self.states.values()),t = t_eval,tfirst = True,**kwargs)










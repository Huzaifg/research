import warnings
# import numpy as np
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


class vd_8dof:
	def __init__(self,parameters = None,states = None):
		self.g = 9.8
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
				self.Cf = -44000
				warnings.warn(f"Set 'Cf' to default value {self.Cf}",UserWarning)
			try:
				self.Cr= parameters['Cr'] # rear axle cornering stiffness (N/rad)
			except:
				self.Cr = -47000
				warnings.warn(f"Set 'Cr' to default value {self.Cr}",UserWarning)
			try:
				self.Cxf= parameters['Cxf'] # front axle longitudinal stiffness (N)
			except:
				self.Cxf = 5000
				warnings.warn(f"Set 'Cxf' to default value {self.Cxf}",UserWarning)
			try:
				self.Cxr= parameters['Cxr'] # rear axle longitudinal stiffness (N)
			except:
				self.Cxr = 5000
				warnings.warn(f"Set 'Cxr' to default value {self.Cxr}",UserWarning)
			try:
				self.m= parameters['m']  # the mass of the vehicle (kg)
			except:
				self.m = 1400
				warnings.warn(f"Set 'm' to default value {self.m}",UserWarning)
			try:
				self.Jz= parameters['Jz'] # yaw moment of inertia (kg.m^2)
			except:
				self.Jz = 2420
				warnings.warn(f"Set 'Jz' to default value {self.Jz}",UserWarning)
			try:
				self.r0 = parameters['r0'] # wheel radius
			except:
				self.r0 = 0.285 
				warnings.warn(f"Set 'r0' to default value {self.r0}",UserWarning)
			try:
				self.Jw= parameters['Jw']  # wheel roll inertia
			except:
				self.Jw = 1
				warnings.warn(f"Set 'Jw' to default value {self.Jw}",UserWarning)
			try:
				self.Jx= parameters['Jx']  # wheel roll inertia
			except:
				self.Jx = 900
				warnings.warn(f"Set 'Jx' to default value {self.Jx}",UserWarning)
			try:
				self.Jy= parameters['Jy']  # wheel roll inertia
			except:
				self.Jy= 2000
				warnings.warn(f"Set 'Jy' to default value {self.Jy}",UserWarning)
			try:
				self.Jxz= parameters['Jxz']  # wheel roll inertia
			except:
				self.Jxz = 90
				warnings.warn(f"Set 'Jxz' to default value {self.Jxz}",UserWarning)
			try:
				self.h= parameters['h']  # wheel roll inertia
			except:
				self.h = 0.75
				warnings.warn(f"Set 'h' to default value {self.h}",UserWarning)
			try:
				self.cf= parameters['cf']  # wheel roll inertia
			except:
				self.cf = 1.5
				warnings.warn(f"Set 'cf' to default value {self.cf}",UserWarning)
			try:
				self.cr= parameters['cr']  # wheel roll inertia
			except:
				self.cr = 1.5
				warnings.warn(f"Set 'cr' to default value {self.cr}",UserWarning)
			try:
				self.muf= parameters['muf']  # wheel roll inertia
			except:
				self.muf = 80
				warnings.warn(f"Set 'muf' to default value {self.muf}",UserWarning)
			try:
				self.mur= parameters['mur']  # wheel roll inertia
			except:
				self.mur = 80
				warnings.warn(f"Set 'mur' to default value {self.mur}",UserWarning)
			try:
				self.ktf= parameters['ktf']  # wheel roll inertia
			except:
				self.ktf = 200000
				warnings.warn(f"Set 'ktf' to default value {self.ktf}",UserWarning)
			try:
				self.ktr= parameters['ktr']  # wheel roll inertia
			except:
				self.ktr = 200000
				warnings.warn(f"Set 'ktr' to default value {self.ktr}",UserWarning)
			try:
				self.hrcf= parameters['hrcf']  # wheel roll inertia
			except:
				self.hrcf = 0.65
				warnings.warn(f"Set 'hrcf' to default value {self.hrcf}",UserWarning)
			try:
				self.hrcr= parameters['hrcr']  # wheel roll inertia
			except:
				self.hrcr = 0.6
				warnings.warn(f"Set 'hrcr' to default value {self.hrcr}",UserWarning)
			try:
				self.brof= parameters['brof']  # wheel roll inertia
			except:
				self.brof = 3000
				warnings.warn(f"Set 'brof' to default value {self.brof}",UserWarning)
			try:
				self.krof= parameters['krof']  # wheel roll inertia
			except:
				self.krof = 29000
				warnings.warn(f"Set 'krof' to default value {self.krof}",UserWarning)
			try:
				self.kror= parameters['kror']  # wheel roll inertia
			except:
				self.kror = 29000
				warnings.warn(f"Set 'kror' to default value {self.kror}",UserWarning)
			try:
				self.bror= parameters['bror']  # wheel roll inertia
			except:
				self.bror = 3000
				warnings.warn(f"Set 'bror' to default value {self.bror}",UserWarning)
			#A dictionary of parameters
			self.params = {'a': self.a,'b' : self.b,'Cf' : self.Cf,'Cr' : self.Cr,'Cxf': self.Cxf,'Cxr' : self.Cxr,'m' : self.m,'Jz' : self.Jz,'r0' : self.r0,'Jw' : self.Jw
			,'Jx':self.Jx,'Jy':self.Jy,'Jxz':self.Jxz,'h':self.h,'cf':self.cf,'cr':self.cr,'muf':self.muf,'mur':self.mur,'ktf':self.ktf,'ktr':self.ktr,'hrcf':self.hrcf,'hrcr':self.hrcr,
			'krof':self.krof,'kror':self.kror,'brof':self.brof,'bror':self.bror}
		else:
			self.a= 1.14# distance of c.g. from front axle (m)
			self.b= 1.4  # distance of c.g. from rear axle  (m)
			self.Cf= -44000  # front axle cornering stiffness (N/rad)
			self.Cr= -47000 # rear axle cornering stiffness (N/rad)
			self.Cxf= 5000 # front axle longitudinal stiffness (N)
			self.Cxr= 5000 # rear axle longitudinal stiffness (N)
			self.m= 1400  # the mass of the vehicle (kg)
			self.Jz= 2420 # yaw moment of inertia (kg.m^2)
			self.r0= 0.285 # wheel radius
			self.Jw= 2  # wheel roll inertia
			self.Jx = 900  # Sprung mass roll inertia (kg.m^2)
			self.Jy  = 2000
			self.Jxz = 90
			self.Jw = 1
			self.h = 0.75
			self.cf = 1.5
			self.cr = 1.5
			self.muf = 80
			self.mur = 80
			self.ktf = 200000
			self.ktr = 200000
			self.hrcf = 0.65
			self.hrcr = 0.6
			self.krof = 29000
			self.kror = 29000
			self.brof = 3000
			self.bror = 3000
			#A dictionary of parameters
			self.params = {'a': self.a,'b' : self.b,'Cf' : self.Cf,'Cr' : self.Cr,'Cxf': self.Cxf,'Cxr' : self.Cxr,'m' : self.m,'Jz' : self.Jz,'r0' : self.r0,'Jw' : self.Jw
			,'Jx':self.Jx,'Jy':self.Jy,'Jxz':self.Jxz,'h':self.h,'cf':self.cf,'cr':self.cr,'muf':self.muf,'mur':self.mur,'ktf':self.ktf,'ktr':self.ktr,'hrcf':self.hrcf,'hrcr':self.hrcr,
			'krof':self.krof,'kror':self.kror,'brof':self.brof,'bror':self.bror}

			warnings.warn("Set parameters to default values" + '\n' + f"{self.params}" ,UserWarning)




		if states is not None:
			if isinstance(states,list):
				raise Exception("Please provide a dictionary for the states")

			# State of the vehicle
			try:
				self.u=states['u'] # lateral velocity 
			except:
				self.u = 50/3.6
				warnings.warn(f"Set 'u' to default value {self.u}",UserWarning)
			try:
				self.v=states['v']
			except:
				self.v = 0.
				warnings.warn(f"Set 'v' to default value {self.v}",UserWarning)
			try:
				self.psi=states['psi'] # yaw rate 
			except:
				self.psi = 0.
				warnings.warn(f"Set 'psi' to default value {self.psi}",UserWarning)
			try:
				self.wlf = states['wlf'] #Front wheel angular velocity 
			except:
				self.wlf = self.u/self.r0
				warnings.warn(f"Set 'wlf' to default value {self.wlf}",UserWarning)
			try:
				self.wlr = states['wlr'] #Rear wheel angular velocity
			except:
				self.wlr = self.u/self.r0
				warnings.warn(f"Set 'wlr' to default value {self.wlr}",UserWarning)
			try:
				self.wrf = states['wrf'] #Rear wheel angular velocity
			except:
				self.wrf = self.u/self.r0
				warnings.warn(f"Set 'wrf' to default value {self.wrf}",UserWarning)
			try:
				self.wrr = states['wrr'] #Rear wheel angular velocity
			except:
				self.wrr = self.u/self.r0
				warnings.warn(f"Set 'wrr' to default value {self.wrr}",UserWarning)
			try:
				self.phi = states['phi'] #Rear wheel angular velocity
			except:
				self.phi = 0.
				warnings.warn(f"Set 'phi' to default value {self.phi}",UserWarning)
			try:
				self.wx = states['wx'] #Rear wheel angular velocity
			except:
				self.wx = 0.
				warnings.warn(f"Set 'wx' to default value {self.wx}",UserWarning)
			try:
				self.wz = states['wz'] #Rear wheel angular velocity
			except:
				self.wz = 0.
				warnings.warn(f"Set 'wz' to default value {self.wz}",UserWarning)
			## the initial tire compression xt
			self.xtrf=((self.m*self.g*self.b)/(2*(self.a+self.b))+self.muf*self.g)/self.ktf  # Right front
			self.xtlf=((self.m*self.g*self.b)/(2*(self.a+self.b))+self.muf*self.g)/self.ktf  #left front
			self.xtlr=((self.m*self.g*self.a)/(2*(self.a+self.b))+self.mur*self.g)/self.ktr 
			self.xtrr=((self.m*self.g*self.a)/(2*(self.a+self.b))+self.mur*self.g)/self.ktr 
			#A dictionary of states
			self.states = {'u' : self.u,'v' : self.v,'psi' : self.psi,'phi' : self.phi,'wx' : self.wx,'wz':self.wz,'wlf':self.wlf,'wlr':self.wlr,'wrf':self.wrf,
			'wrr':self.wrr}
			self.init_states = {'u' : self.u,'v' : self.v,'psi' : self.psi,'phi' : self.phi,'wx' : self.wx,'wz':self.wz,'wlf':self.wlf,'wlr':self.wlr,'wrf':self.wrf,
			'wrr':self.wrr}
		else:
			self.x = 0
			self.y= 0
			self.u=50/3.6  # the longitudinal velocity 
			self.v=0.     # the lateral velocity 
			self.phi=0.   # roll angle
			self.psi=0.
			self.wx=0.
			self.wz=0.   # yaw angular velocity
			self.wlf=self.u/self.r0  # angular velocity of left front wheel rotation 
			self.wrf=self.u/self.r0  # angular velocity of right front wheel rotation
			self.wlr=self.u/self.r0  # angular velocity of left rear wheel rotation
			self.wrr=self.u/self.r0  # angular velocity of right rear wheel rotation
			## the initial tire compression xt
			self.xtrf=((self.m*self.g*self.b)/(2*(self.a+self.b))+self.muf*self.g)/self.ktf  # Right front
			self.xtlf=((self.m*self.g*self.b)/(2*(self.a+self.b))+self.muf*self.g)/self.ktf  #left front
			self.xtlr=((self.m*self.g*self.a)/(2*(self.a+self.b))+self.mur*self.g)/self.ktr 
			self.xtrr=((self.m*self.g*self.a)/(2*(self.a+self.b))+self.mur*self.g)/self.ktr
			# print(self.xtlf,self.xtrf,self.xtlr,self.xtrr) 
			#A dictionary of states
			self.states = {'x' : self.x,'y' : self.y,'u' : self.u,'v' : self.v,'psi' : self.psi,'phi' : self.phi,'wx' : self.wx,'wz':self.wz,'wlf':self.wlf,'wlr':self.wlr,'wrf':self.wrf,
			'wrr':self.wrr}
			self.init_states = {'x' : self.x,'y' : self.y,'u' : self.u,'v' : self.v,'psi' : self.psi,'phi' : self.phi,'wx' : self.wx,'wz':self.wz,'wlf':self.wlf,'wlr':self.wlr,'wrf':self.wrf,
			'wrr':self.wrr}
			warnings.warn("Set States to default values" + '\n' + f"{self.states}" ,UserWarning)


	#Simple print to show the parameters and states
	def __str__(self):
		return  str(self.__class__) + '\n' + "Vehicle Parameters are" + '\n' + f"{self.params}" + '\n' + "Vehicle state is" + '\n' + f"{self.states}"


	#Sets the controls for the vehicle
	def set_steering(self,steering):
		if(isinstance(steering, types.FunctionType)):
			self.steering = steering
		else:
			raise Exception(f"The controls provided should be of type {types.FunctionType}. It should also be of the form f(t)")

	#Sets the controls for the vehicle
	def set_torque(self,torque):
		if(isinstance(torque, types.FunctionType)):
			self.torque = torque
		else:
			raise Exception(f"The controls provided should be of type {types.FunctionType}. It should be of the form f(t)")

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



	#Just a method to reset the state to default values or values specified when vehicle object was created
	def reset_state(self):
		self.xtrf=((self.m*self.g*self.b)/(2*(self.a+self.b))+self.muf*self.g)/self.ktf  # Right front
		self.xtlf=((self.m*self.g*self.b)/(2*(self.a+self.b))+self.muf*self.g)/self.ktf  #left front
		self.xtlr=((self.m*self.g*self.a)/(2*(self.a+self.b))+self.mur*self.g)/self.ktr 
		self.xtrr=((self.m*self.g*self.a)/(2*(self.a+self.b))+self.mur*self.g)/self.ktr
		self.state = self.init_states



	def model(self,t,state):
		a,b,Cf,Cr,Cxf,Cxr,m,Jz,r0,Jw,Jx,Jy,Jxz,h,cf,cr,muf,mur,ktf,ktr,hrcf,hrcr,krof,kror,brof,bror =list(self.params.values())
		g = 9.8
		x,y,u,v,psi,phi,wx,wz,wlf,wlr,wrf,wrr = state


		### Some calculated parameters
		hrc=(hrcf*b+hrcr*a)/(a+b) # the vertical distance from the sprung mass C.M. to the vehicle roll center.
		mt=m+2*muf+2*mur # vehicle total mass


		#Instantaneous tire radius
		Rrf=r0-self.xtrf
		Rlf=r0-self.xtlf
		Rlr=r0-self.xtlr
		Rrr=r0-self.xtrr

	   
		##position of front and rear unsprung mass
		huf=Rrf 
		hur=Rrr
		##the longitudinal and lateral velocities at the tire contact patch in coordinate frame 2
		ugrf=u+(wz*cf)/2
		vgrf=v+wz*a
		uglf=u-(wz*cf)/2
		vglf=v+wz*a
		uglr=u-(wz*cr)/2
		vglr=v-wz*b
		ugrr=u+(wz*cr)/2
		vgrr=v-wz*b
		## tire slip angle of each wheel
		delta_rf=np.arctan(vgrf/ugrf)-self.steering(t) 
		delta_lf=np.arctan(vglf/uglf)-self.steering(t) 
		delta_lr=np.arctan(vglr/uglr) 
		delta_rr=np.arctan(vgrr/ugrr) 
		##linear tire lateral force
		Fytrf=Cf*delta_rf 
		Fytlf=Cf*delta_lf 
		Fytlr=Cr*delta_lr 
		Fytrr=Cr*delta_rr 
		## longitudinal slips
		s_rf=(Rrf*wrf-(ugrf*np.cos(self.steering(t))+vgrf*np.sin(self.steering(t))))/np.abs(ugrf*np.cos(self.steering(t))+vgrf*np.sin(self.steering(t))) 
		s_lf=(Rlf*wlf-(uglf*np.cos(self.steering(t))+vglf*np.sin(self.steering(t))))/np.abs(uglf*np.cos(self.steering(t))+vglf*np.sin(self.steering(t))) 
		s_lr=(Rlr*wlr-uglr)/np.abs(uglr) 
		s_rr=(Rrr*wrr-ugrr)/np.abs(ugrr) 
		## linear tire longitudinal force 
		Fxtrf=Cxf*s_rf 
		Fxtlf=Cxf*s_lf 
		Fxtlr=Cxr*s_lr 
		Fxtrr=Cxr*s_rr 
		## the forces Fxgij obtained by resolving the longitudinal and cornering forces at the tire contact patch 
		Fxglf=Fxtlf*np.cos(self.steering(t))-Fytlf*np.sin(self.steering(t)) 
		Fxgrf=Fxtrf*np.cos(self.steering(t))-Fytrf*np.sin(self.steering(t)) 
		Fxglr=Fxtlr 
		Fxgrr=Fxtrr 
		Fyglf=Fxtlf*np.sin(self.steering(t))+Fytlf*np.cos(self.steering(t)) 
		Fygrf=Fxtrf*np.sin(self.steering(t))+Fytrf*np.cos(self.steering(t)) 
		Fyglr=Fytlr 
		Fygrr=Fytrr

		# Some other constants used in the differential equations
		dpsi=wz
		dphi=wx
		E1=-mt*wz*u+(Fyglf+Fygrf+Fyglr+Fygrr) 
		E2=(Fyglf+Fygrf)*a-(Fyglr+Fygrr)*b+(Fxgrf-Fxglf)*cf/2+(Fxgrr-Fxglr)*cr/2+(mur*b-muf*a)*wz*u
		E3=m*g*hrc*phi-(krof+kror)*phi-(brof+bror)*dphi+hrc*m*wz*u 
		A1=mur*b-muf*a 
		A2=Jx+m*hrc**2 
		A3=hrc*m 


		#Chassis Model
		u_dot = wz*v+(1/mt)*((Fxglf+Fxgrf+Fxglr+Fxgrr)+(muf*a-mur*b)*(wz)**2-2*hrc*m*wz*wx) 
		v_dot=(E1*Jxz**2-A1*A2*E2+A1*E3*Jxz+A3*E2*Jxz-A2*E1*Jz-A3*E3*Jz)/(A2*A1**2-2*A1*A3*Jxz+Jz*A3**2+mt*Jxz**2-A2*Jz*mt)
		wx_dot=(A1**2*E3-A1*A3*E2+A1*E1*Jxz-A3*E1*Jz+E2*Jxz*mt-E3*Jz*mt)/(A2*A1**2-2*A1*A3*Jxz+Jz*A3**2+mt*Jxz**2-A2*Jz*mt) 
		wz_dot=(A3**2*E2-A1*A2*E1-A1*A3*E3+A3*E1*Jxz-A2*E2*mt+E3*Jxz*mt)/(A2*A1**2-2*A1*A3*Jxz+Jz*A3**2+mt*Jxz**2-A2*Jz*mt)
		dx = u
		dy = v


		## Wheel rotational modelling
		dwlf=(1/Jw)*(self.torque(t) - Fxtlf*Rlf)
		dwrf=(1/Jw)*(self.torque(t) - Fxtrf*Rrf)
		dwlr=(1/Jw)*(self.torque(t) - Fxtlr*Rlr)
		dwrr=(1/Jw)*(self.torque(t) - Fxtrr*Rrr)



		##The normal forces at four tires are determined as in order to update the tire compression for the next time step
		Z1=(m*g*b)/(2*(a+b))+(muf*g)/2 
		Z2=((muf*huf)/cf+m*b*(h-hrcf)/(cf*(a+b)))*(v_dot+wz*u) 
		Z3=(krof*phi+brof*dphi)/cf 
		Z4=((m*h+muf*huf+mur*hur)*(u_dot-wz*v))/(2*(a+b)) 
		Fzglf=Z1-Z2-Z3-Z4 
		Fzgrf=Z1+Z2+Z3-Z4 
		Z5=(m*g*a)/(2*(a+b))+(mur*g)/2 
		Z6=((mur*hur)/cr+m*a*(h-hrcr)/(cr*(a+b)))*(v_dot+wz*u) 
		Z7=(kror*phi+bror*dphi)/cr 
		Z8=((m*h+muf*huf+mur*hur)*(u_dot-wz*v))/(2*(a+b)) 
		Fzglr=Z5-Z6-Z7+Z8 
		Fzgrr=Z5+Z6+Z7+Z8 

		self.xtlf=Fzglf/ktf 
		self.xtrf=Fzgrf/ktf 
		self.xtlr=Fzglr/ktr 
		self.xtrr=Fzgrr/ktr 


		return np.stack([dx,dy,u_dot,v_dot,dpsi,dphi,wx_dot,wz_dot,dwlf,dwlr,dwrf,dwrr])

	def jax_model(self,state,t,theta):
		a,b,Cf,Cr,Cxf,Cxr,m,Jz,r0,Jw,Jx,Jy,Jxz,h,cf,cr,muf,mur,ktf,ktr,hrcf,hrcr,krof,kror,brof,bror =list(self.params.values())
		self.update_params(Cf=theta[0],Cr=theta[1])
		print(self)
		g = 9.8
		u,v,psi,phi,wx,wz,wlf,wlr,wrf,wrr = state


		### Some calculated parameters
		hrc=(hrcf*b+hrcr*a)/(a+b) # the vertical distance from the sprung mass C.M. to the vehicle roll center.
		mt=m+2*muf+2*mur # vehicle total mass


		#Instantaneous tire radius
		Rrf=r0-self.xtrf
		Rlf=r0-self.xtlf
		Rlr=r0-self.xtlr
		Rrr=r0-self.xtrr

	   
		##position of front and rear unsprung mass
		huf=Rrf 
		hur=Rrr
		##the longitudinal and lateral velocities at the tire contact patch in coordinate frame 2
		ugrf=u+(wz*cf)/2
		vgrf=v+wz*a
		uglf=u-(wz*cf)/2
		vglf=v+wz*a
		uglr=u-(wz*cr)/2
		vglr=v-wz*b
		ugrr=u+(wz*cr)/2
		vgrr=v-wz*b
		## tire slip angle of each wheel
		delta_rf=np.arctan(vgrf/ugrf)-self.steering(t) 
		delta_lf=np.arctan(vglf/uglf)-self.steering(t) 
		delta_lr=np.arctan(vglr/uglr) 
		delta_rr=np.arctan(vgrr/ugrr) 
		##linear tire lateral force
		Fytrf=Cf*delta_rf 
		Fytlf=Cf*delta_lf 
		Fytlr=Cr*delta_lr 
		Fytrr=Cr*delta_rr 
		## longitudinal slips
		s_rf=(Rrf*wrf-(ugrf*np.cos(self.steering(t))+vgrf*np.sin(self.steering(t))))/np.abs(ugrf*np.cos(self.steering(t))+vgrf*np.sin(self.steering(t))) 
		s_lf=(Rlf*wlf-(uglf*np.cos(self.steering(t))+vglf*np.sin(self.steering(t))))/np.abs(uglf*np.cos(self.steering(t))+vglf*np.sin(self.steering(t))) 
		s_lr=(Rlr*wlr-uglr)/np.abs(uglr) 
		s_rr=(Rrr*wrr-ugrr)/np.abs(ugrr) 
		## linear tire longitudinal force 
		Fxtrf=Cxf*s_rf 
		Fxtlf=Cxf*s_lf 
		Fxtlr=Cxr*s_lr 
		Fxtrr=Cxr*s_rr 
		## the forces Fxgij obtained by resolving the longitudinal and cornering forces at the tire contact patch 
		Fxglf=Fxtlf*np.cos(self.steering(t))-Fytlf*np.sin(self.steering(t)) 
		Fxgrf=Fxtrf*np.cos(self.steering(t))-Fytrf*np.sin(self.steering(t)) 
		Fxglr=Fxtlr 
		Fxgrr=Fxtrr 
		Fyglf=Fxtlf*np.sin(self.steering(t))+Fytlf*np.cos(self.steering(t)) 
		Fygrf=Fxtrf*np.sin(self.steering(t))+Fytrf*np.cos(self.steering(t)) 
		Fyglr=Fytlr 
		Fygrr=Fytrr

		# Some other constants used in the differential equations
		dpsi=wz
		dphi=wx
		E1=-mt*wz*u+(Fyglf+Fygrf+Fyglr+Fygrr) 
		E2=(Fyglf+Fygrf)*a-(Fyglr+Fygrr)*b+(Fxgrf-Fxglf)*cf/2+(Fxgrr-Fxglr)*cr/2+(mur*b-muf*a)*wz*u
		E3=m*g*hrc*phi-(krof+kror)*phi-(brof+bror)*dphi+hrc*m*wz*u 
		A1=mur*b-muf*a 
		A2=Jx+m*hrc**2 
		A3=hrc*m 


		#Chassis Model
		u_dot = wz*v+(1/mt)*((Fxglf+Fxgrf+Fxglr+Fxgrr)+(muf*a-mur*b)*(wz)**2-2*hrc*m*wz*wx) 
		v_dot=(E1*Jxz**2-A1*A2*E2+A1*E3*Jxz+A3*E2*Jxz-A2*E1*Jz-A3*E3*Jz)/(A2*A1**2-2*A1*A3*Jxz+Jz*A3**2+mt*Jxz**2-A2*Jz*mt)
		wx_dot=(A1**2*E3-A1*A3*E2+A1*E1*Jxz-A3*E1*Jz+E2*Jxz*mt-E3*Jz*mt)/(A2*A1**2-2*A1*A3*Jxz+Jz*A3**2+mt*Jxz**2-A2*Jz*mt) 
		wz_dot=(A3**2*E2-A1*A2*E1-A1*A3*E3+A3*E1*Jxz-A2*E2*mt+E3*Jxz*mt)/(A2*A1**2-2*A1*A3*Jxz+Jz*A3**2+mt*Jxz**2-A2*Jz*mt)


		## Wheel rotational modelling
		dwlf=(1/Jw)*(self.torque(t) - Fxtlf*Rlf)
		dwrf=(1/Jw)*(self.torque(t) - Fxtrf*Rrf)
		dwlr=(1/Jw)*(self.torque(t) - Fxtlr*Rlr)
		dwrr=(1/Jw)*(self.torque(t) - Fxtrr*Rrr)




		##The normal forces at four tires are determined as in order to update the tire compression for the next time step
		Z1=(m*g*b)/(2*(a+b))+(muf*g)/2 
		Z2=((muf*huf)/cf+m*b*(h-hrcf)/(cf*(a+b)))*(v_dot+wz*u) 
		Z3=(krof*phi+brof*dphi)/cf 
		Z4=((m*h+muf*huf+mur*hur)*(u_dot-wz*v))/(2*(a+b)) 
		Fzglf=Z1-Z2-Z3-Z4 
		Fzgrf=Z1+Z2+Z3-Z4 
		Z5=(m*g*a)/(2*(a+b))+(mur*g)/2 
		Z6=((mur*hur)/cr+m*a*(h-hrcr)/(cr*(a+b)))*(v_dot+wz*u) 
		Z7=(kror*phi+bror*dphi)/cr 
		Z8=((m*h+muf*huf+mur*hur)*(u_dot-wz*v))/(2*(a+b)) 
		Fzglr=Z5-Z6-Z7+Z8 
		Fzgrr=Z5+Z6+Z7+Z8 

		self.xtlf=Fzglf/ktf 
		self.xtrf=Fzgrf/ktf 
		self.xtlr=Fzglr/ktr 
		self.xtrr=Fzgrr/ktr 


		return np.stack([u_dot,v_dot,dpsi,dphi,wx_dot,wz_dot,dwlf,dwlr,dwrf,dwrr])
		#Awrapper function for maybe a few packages - starting with solve_ivp,odeint
	def solve(self,package = 'solve_ivp',t_eval = None,**kwargs):
		try:
			self.steering
		except:
			raise Exception("Please provide steering controls for the vehicle with 'set_steering' method")

		if t_eval is None:
			raise Exception("Please provide times steps at which you want the solution to be evaluated")


		if(package ==  'solve_ivp'):
			return solve_ivp(self.model,t_span=[t_eval[0],t_eval[-1]],y0 = list(self.states.values()),vectorized = False,t_eval = t_eval,**kwargs)
		elif(package == 'odeint'):
			return odeint(self.model,y0 = list(self.states.values()),t = t_eval,tfirst = True,**kwargs)
		# elif(package == 'jax'):
		# 	return odeint(self.model,list(self.states.values()),t_eval,rtol = 1.e-4,atol = 1.e-8)







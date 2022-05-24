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
		# To parametrize the inputs
		# Pitman arm
		# self.max_steer = 0.7328647
		# self.max_steer = 1
		# Rack and pinion
		self.max_steer = 0.6525249
		#The 0.2 is from the conical gears of the chrono vehicle
		self.gear_ratio = 0.3*0.2
		self.max_torque = 1000.
		self.max_speed = 2000.


		#Trial
		self.umin = 0.5568
		self.umax = 0.9835
		self.xrel = 1.0
		self.yrel = 1.0

		self.trans_time = 0.2
		self.flf = []
		self.flr = []
		self.frf = []
		self.frr = []

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
			try:
				self.rr= parameters['rr']  # wheel roll inertia
			except:
				self.rr = 0.0125
				warnings.warn(f"Set 'rr' to default value {self.rr}",UserWarning)
			#A dictionary of parameters
			self.params = {'a': self.a,'b' : self.b,'Cf' : self.Cf,'Cr' : self.Cr,'Cxf': self.Cxf,'Cxr' : self.Cxr,'m' : self.m,'Jz' : self.Jz,'r0' : self.r0,'Jw' : self.Jw
			,'Jx':self.Jx,'Jy':self.Jy,'Jxz':self.Jxz,'h':self.h,'cf':self.cf,'cr':self.cr,'muf':self.muf,'mur':self.mur,'ktf':self.ktf,'ktr':self.ktr,'hrcf':self.hrcf,'hrcr':self.hrcr,
			'krof':self.krof,'kror':self.kror,'brof':self.brof,'bror':self.bror,'rr':self.rr}
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
			self.rr = 0.0125
			#A dictionary of parameters
			self.params = {'a': self.a,'b' : self.b,'Cf' : self.Cf,'Cr' : self.Cr,'Cxf': self.Cxf,'Cxr' : self.Cxr,'m' : self.m,'Jz' : self.Jz,'r0' : self.r0,'Jw' : self.Jw
			,'Jx':self.Jx,'Jy':self.Jy,'Jxz':self.Jxz,'h':self.h,'cf':self.cf,'cr':self.cr,'muf':self.muf,'mur':self.mur,'ktf':self.ktf,'ktr':self.ktr,'hrcf':self.hrcf,'hrcr':self.hrcr,
			'krof':self.krof,'kror':self.kror,'brof':self.brof,'bror':self.bror,'rr':self.rr}

			warnings.warn("Set parameters to default values" + '\n' + f"{self.params}" ,UserWarning)
			self.xtrf_ar = []
			self.s_arr = []
			self.t_arr = []
			self.dt = []
			self.fdt = []
			self.rdt = []



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
			


			self.Fzgrf = ((self.m*self.g*self.b)/(2*(self.a+self.b))+self.muf*self.g)
			self.Fzglf = ((self.m*self.g*self.b)/(2*(self.a+self.b))+self.muf*self.g)
			self.Fzglr = ((self.m*self.g*self.b)/(2*(self.a+self.b))+self.mur*self.g)
			self.Fzgrr = ((self.m*self.g*self.b)/(2*(self.a+self.b))+self.mur*self.g)

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


			#Vertical forces initially
			self.Fzgrf = ((self.m*self.g*self.b)/(2*(self.a+self.b))+self.muf*self.g)
			self.Fzglf = ((self.m*self.g*self.b)/(2*(self.a+self.b))+self.muf*self.g)
			self.Fzglr = ((self.m*self.g*self.b)/(2*(self.a+self.b))+self.mur*self.g)
			self.Fzgrr = ((self.m*self.g*self.b)/(2*(self.a+self.b))+self.mur*self.g)

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
		self.steering = steering





	#Sets the controls for the vehicle
	def set_torque(self,torque):
		if(isinstance(torque, types.FunctionType)):
			self.torque = torque
		else:
			raise Exception(f"The controls provided should be of type {types.FunctionType}. It should be of the form f(t)")

	#Throttle controls of the vehicle
	def set_throttle(self,throttle):
		self.throttle = throttle

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
		#Vertical forces initially
		self.Fzgrf = ((self.m*self.g*self.b)/(2*(self.a+self.b))+self.muf*self.g)
		self.Fzglf = ((self.m*self.g*self.b)/(2*(self.a+self.b))+self.muf*self.g)
		self.Fzglr = ((self.m*self.g*self.b)/(2*(self.a+self.b))+self.mur*self.g)
		self.Fzgrr = ((self.m*self.g*self.b)/(2*(self.a+self.b))+self.mur*self.g)

		self.xtrf=((self.m*self.g*self.b)/(2*(self.a+self.b))+self.muf*self.g)/self.ktf  # Right front
		self.xtlf=((self.m*self.g*self.b)/(2*(self.a+self.b))+self.muf*self.g)/self.ktf  #left front
		self.xtlr=((self.m*self.g*self.a)/(2*(self.a+self.b))+self.mur*self.g)/self.ktr 
		self.xtrr=((self.m*self.g*self.a)/(2*(self.a+self.b))+self.mur*self.g)/self.ktr
		self.state = self.init_states


	def drive_torque(self,t,w):
		motor_speed = w / self.gear_ratio
		motor_torque = self.max_torque - (motor_speed * (self.max_torque / self.max_speed))
		motor_torque = motor_torque * self.throttle(t)
		return motor_torque / self.gear_ratio

	def smooth_step(self,t,f1,t1,f2,t2):
		if(t < t1):
			return f1
		elif(t >= t2):
			return f2
		else:
			return f1 + ((f2 - f1)*((t-t1)/(t2-t1))**2 * (3 - 2*((t - t1)/(t2-t1))))


	def model(self,t,state):
		a,b,Cf,Cr,Cxf,Cxr,m,Jz,r0,Jw,Jx,Jy,Jxz,h,cf,cr,muf,mur,ktf,ktr,hrcf,hrcr,krof,kror,brof,bror,rr =list(self.params.values())
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

		# if(Rrf > 2*r0):
		# 	Rrf = r0
		# if(Rlf > 2*r0):
		# 	Rlf = r0
		# if(Rlr > 2*r0):
		# 	Rlr = r0
		# if(Rrr > 2*r0):
		# 	Rrr = r0


		# if(Rrf < 0):
		# 	Rrf = r0
		# if(Rlf < 0):
		# 	Rlf = r0
		# if(Rlr < 0):
		# 	Rlr = r0
		# if(Rrr < 0):
		# 	Rrr = r0



		# Rrf = r0
		# Rlf = r0
		# Rlr = r0
		# Rrr = r0
		# print(self.xtrf)
		self.xtrf_ar.append(self.xtlf)

	   
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
		# print(ugrf)
		# print(vgrf)	
		## tire slip angle of each wheel
		if((abs(ugrf) > 1e-4) or (abs(vgrf) > 1e-4)):
			delta_rf=np.arctan(vgrf/ugrf)-(self.steering(t)* self.max_steer)
			s_rf = max(min((Rrf*wrf-(ugrf*np.cos(self.steering(t)* self.max_steer)+vgrf*np.sin(self.steering(t)* self.max_steer)))/np.abs(ugrf*np.cos(self.steering(t)* self.max_steer)
			+vgrf*np.sin(self.steering(t)* self.max_steer)),1),-1) 
		else:
			delta_rf = self.steering(t)* self.max_steer
			s_rf = 0
		if((abs(uglf) > 1e-4) or (abs(vglf) > 1e-4)):
			delta_lf=np.arctan(vglf/uglf)-(self.steering(t)* self.max_steer)
			s_lf=max(min((Rlf*wlf-(uglf*np.cos(self.steering(t)* self.max_steer)+vglf*np.sin(self.steering(t)* self.max_steer)))/np.abs(uglf*np.cos(self.steering(t)* self.max_steer)
			+vglf*np.sin(self.steering(t)* self.max_steer)),1),-1) 
		else:
			delta_lf = self.steering(t)* self.max_steer
			s_lf = 0
		if((abs(uglr) > 1e-4) or (abs(vglr) > 1e-4)):
			delta_lr=np.arctan(vglr/uglr)
			s_lr=max(min((Rlr*wlr-uglr)/np.abs(uglr),1),-1)
		else:
			delta_lr = 0
			s_lr = 0
		if((abs(ugrr) > 1e-4) or (abs(vgrr) > 1e-4)):
			delta_rr=np.arctan(vgrr/ugrr)
			s_rr=max(min((Rrr*wrr-ugrr)/np.abs(ugrr),1),-1)
		else:
			delta_rr = 0
			s_rr = 0

		
		smth = self.smooth_step(t,0,self.start_time,1,self.start_time + self.trans_time)
		# print(smth)
		# print(s_crit_rff)
		# print(f"{delta_rf = }")
		# print(f"{s_rf = }")
		##linear tire lateral force
		Fytrf=-Cf*delta_rf*smth 
		Fytlf=-Cf*delta_lf*smth 
		Fytlr=-Cr*delta_lr*smth 
		Fytrr=-Cr*delta_rr*smth 
		## longitudinal slips

		# s_rf=(Rrf*wrf-(ugrf*np.cos(self.steering(t)* self.max_steer)+vgrf*np.sin(self.steering(t)* self.max_steer)))/np.abs(ugrf*np.cos(self.steering(t)* self.max_steer)
		# 	+vgrf*np.sin(self.steering(t)* self.max_steer)) 
		# s_lf=(Rlf*wlf-(uglf*np.cos(self.steering(t)* self.max_steer)+vglf*np.sin(self.steering(t)* self.max_steer)))/np.abs(uglf*np.cos(self.steering(t)* self.max_steer)
		# 	+vglf*np.sin(self.steering(t)* self.max_steer)) 
		# s_lr=(Rlr*wlr-uglr)/np.abs(uglr) 
		# s_rr=(Rrr*wrr-ugrr)/np.abs(ugrr) 
		## linear tire longitudinal force 
		Fxtrf=Cxf*s_rf*smth 
		Fxtlf=Cxf*s_lf*smth 
		Fxtlr=Cxr*s_lr*smth 
		Fxtrr=Cxr*s_rr*smth 
		## the forces Fxgij obtained by resolving the longitudinal and cornering forces at the tire contact patch 
		Fxglf=Fxtlf*np.cos(self.steering(t)* self.max_steer)-Fytlf*np.sin(self.steering(t)* self.max_steer) 
		Fxgrf=Fxtrf*np.cos(self.steering(t)* self.max_steer)-Fytrf*np.sin(self.steering(t)* self.max_steer) 
		Fxglr=Fxtlr 
		Fxgrr=Fxtrr 
		Fyglf=Fxtlf*np.sin(self.steering(t)* self.max_steer)+Fytlf*np.cos(self.steering(t)* self.max_steer) 
		Fygrf=Fxtrf*np.sin(self.steering(t)* self.max_steer)+Fytrf*np.cos(self.steering(t)* self.max_steer) 
		Fyglr=Fytlr 
		Fygrr=Fytrr

		# print(f"{Fxglf=}")
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
		dx = u*np.cos(psi) - v*np.sin(psi)
		dy = u*np.sin(psi) + v*np.cos(psi)






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


		#These vertical forces cannot be lesser than 0 as that means that the wheel is off the ground
		if(Fzgrf < 0):
			Fzgrf = 0
		if(Fzglf < 0):
			Fzglf = 0
		if(Fzglr < 0):
			Fzglr = 0
		if(Fzgrr < 0):
			Fzgrr = 0


		# print(f"{t = },{Fzgrf = }, {Fzglf = }")

		#Maybe replace with a better maximum tire deflection
		# if((Fzgrf/ktf) > r0):
		# 	self.xtrf = self.xtrf
		# else:
		# 	self.xtrf=Fzgrf/ktf

		# if((Fzglf/ktf) > r0):
		# 	self.xtlf = self.xtlf
		# else:
		# 	self.xtlf=Fzglf/ktf

		
		# if((Fzglr/ktf) > r0):
		# 	self.xtlr = self.xtlr
		# else:
		# 	self.xtlr=Fzglr/ktf

		# if((Fzgrr/ktf) > r0):
		# 	self.xtrr = self.xtrr
		# else:
		# 	self.xtrr=Fzgrr/ktf

		self.xtlf=Fzglf/ktf 
		self.xtrf=Fzgrf/ktf 
		self.xtlr=Fzglr/ktr 
		self.xtrr=Fzgrr/ktr 




		rolling_res_lf = -rr * np.abs(Fzglf) * np.sign(wlf)
		rolling_res_rf = -rr * np.abs(Fzgrf) * np.sign(wrf)
		rolling_res_lr = -rr * np.abs(Fzglr) * np.sign(wlr)
		rolling_res_rr = -rr * np.abs(Fzgrr) * np.sign(wrr)


		## Wheel rotational modelling
		# print(f"{Fxtlf = },{Rlf = }, {Jw = }")
		# dwlf=(1/Jw)*(self.torque(t) - Fxtlf*Rlf)
		# dwrf=(1/Jw)*(self.torque(t) - Fxtrf*Rrf)
		# dwlr=(1/Jw)*(self.torque(t) - Fxtlr*Rlr)
		# dwrr=(1/Jw)*(self.torque(t) - Fxtrr*Rrr)
		
		dwlf=(1/Jw)*(self.drive_torque(t,wlf)/4 + rolling_res_lf - Fxtlf*Rlf)
		dwrf=(1/Jw)*(self.drive_torque(t,wrf)/4 + rolling_res_rf - Fxtrf*Rrf)
		dwlr=(1/Jw)*(self.drive_torque(t,wlr)/4 + rolling_res_lr - Fxtlr*Rlr)
		dwrr=(1/Jw)*(self.drive_torque(t,wrr)/4 + rolling_res_rr - Fxtrr*Rrr)
		# print(f"{wlf=}")
		# dwlf=(1/Jw)*(self.torque(t) + rolling_res_lf - Fxtlf*Rlf)
		# dwrf=(1/Jw)*(self.torque(t) + rolling_res_rf - Fxtrf*Rrf)
		# dwlr=(1/Jw)*(self.torque(t) + rolling_res_lr - Fxtlr*Rlr)
		# dwrr=(1/Jw)*(self.torque(t) + rolling_res_rr - Fxtrr*Rrr)
		return np.stack([dx,dy,u_dot,v_dot,dpsi,dphi,wx_dot,wz_dot,dwlf,dwlr,dwrf,dwrr])

	

	def model_tr(self,t,state):
		a,b,Cf,Cr,Cxf,Cxr,m,Jz,r0,Jw,Jx,Jy,Jxz,h,cf,cr,muf,mur,ktf,ktr,hrcf,hrcr,krof,kror,brof,bror,rr =list(self.params.values())
		g = 9.8
		x,y,u,v,psi,phi,wx,wz,wlf,wlr,wrf,wrr = state

		self.t_arr.append(t)
		## Some calculated parameters
		hrc=(hrcf*b+hrcr*a)/(a+b) # the vertical distance from the sprung mass C.M. to the vehicle roll center.
		mt=m+2*muf+2*mur # vehicle total mass


		#Instantaneous tire radius
		Rrf=r0-self.xtrf
		Rlf=r0-self.xtlf
		Rlr=r0-self.xtlr
		Rrr=r0-self.xtrr
		# Rrf = r0
		# Rlf = r0
		# Rlr = r0
		# Rrr = r0
		# print(self.xtrf)
		self.xtrf_ar.append(self.xtlf)

	   
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
		
		# print(vgrf)	
		# tire slip angle of each wheel
		if((abs(ugrf) > 1e-4) or (abs(vgrf) > 1e-4)):
			# delta_rf=np.arctan(vgrf/ugrf)-(self.steering(t)* self.max_steer)
			delta_rf=np.arctan2(vgrf,np.abs(ugrf))-(self.steering(t)* self.max_steer)
			s_rf = (Rrf*wrf-(ugrf*np.cos(self.steering(t)* self.max_steer)+vgrf*np.sin(self.steering(t)* self.max_steer)))/np.abs(ugrf*np.cos(self.steering(t)* self.max_steer)
			+vgrf*np.sin(self.steering(t)* self.max_steer)) 
		else:
			delta_rf = self.steering(t)* self.max_steer
			s_rf = 0
		if((abs(uglf) > 1e-4) or (abs(vglf) > 1e-4)):
			# delta_lf=np.arctan(vglf/uglf)-(self.steering(t)* self.max_steer)
			delta_lf = np.arctan2(vglf,np.abs(uglf))-(self.steering(t)* self.max_steer)
			s_lf=(Rlf*wlf-(uglf*np.cos(self.steering(t)* self.max_steer)+vglf*np.sin(self.steering(t)* self.max_steer)))/np.abs(uglf*np.cos(self.steering(t)* self.max_steer)
			+vglf*np.sin(self.steering(t)* self.max_steer)) 
		else:
			delta_lf = self.steering(t)* self.max_steer
			s_lf = 0
		if((abs(uglr) > 1e-4) or (abs(vglr) > 1e-4)):
			# delta_lr=np.arctan(vglr/uglr)
			delta_lr=np.arctan2(vglr,np.abs(uglr))
			s_lr=(Rlr*wlr-uglr)/np.abs(uglr)
		else:
			delta_lr = 0
			s_lr = 0
		if((abs(ugrr) > 1e-4) or (abs(vgrr) > 1e-4)):
			# delta_rr=np.arctan(vgrr/ugrr)
			delta_rr = np.arctan2(vgrr,np.abs(ugrr))
			s_rr=(Rrr*wrr-ugrr)/np.abs(ugrr)
		else:
			delta_rr = 0
			s_rr = 0

		self.s_arr.append(s_lr) 
		# print(f"{s_rr = }")
		# if((abs(ugrf) > 1e-4) or (abs(vgrf) > 1e-4)):
		# 	delta_rf=np.arctan(vgrf/ugrf)-(self.steering(t)* self.max_steer)
		# 	s_rf = max(min((Rrf*wrf-(ugrf*np.cos(self.steering(t)* self.max_steer)+vgrf*np.sin(self.steering(t)* self.max_steer)))/np.abs(ugrf*np.cos(self.steering(t)* self.max_steer)
		# 	+vgrf*np.sin(self.steering(t)* self.max_steer)),1),-1) 
		# else:
		# 	delta_rf = self.steering(t)* self.max_steer
		# 	s_rf = 0
		# if((abs(uglf) > 1e-4) or (abs(vglf) > 1e-4)):
		# 	delta_lf=np.arctan(vglf/uglf)-(self.steering(t)* self.max_steer)
		# 	s_lf=max(min((Rlf*wlf-(uglf*np.cos(self.steering(t)* self.max_steer)+vglf*np.sin(self.steering(t)* self.max_steer)))/np.abs(uglf*np.cos(self.steering(t)* self.max_steer)
		# 	+vglf*np.sin(self.steering(t)* self.max_steer)),1),-1) 
		# else:
		# 	delta_lf = self.steering(t)* self.max_steer
		# 	s_lf = 0
		# if((abs(uglr) > 1e-4) or (abs(vglr) > 1e-4)):
		# 	delta_lr=np.arctan(vglr/uglr)
		# 	s_lr=max(min((Rlr*wlr-uglr)/np.abs(uglr),1),-1)
		# else:
		# 	delta_lr = 0
		# 	s_lr = 0
		# if((abs(ugrr) > 1e-4) or (abs(vgrr) > 1e-4)):
		# 	delta_rr=np.arctan(vgrr/ugrr)
		# 	s_rr=max(min((Rrr*wrr-ugrr)/np.abs(ugrr),1),-1)
		# else:
		# 	delta_rr = 0
		# 	s_rr = 0


		# smth = self.smooth_step(t,0,self.start_time,1,self.start_time + self.trans_time)
		smth = 1
		
		# print(f"{s_lr = }")
		ss_rf = min(np.sqrt(s_rf**2 + np.tan(delta_rf)**2),1.)
		ss_lf = min(np.sqrt(s_lf**2 + np.tan(delta_lf)**2),1.)
		ss_lr = min(np.sqrt(s_lr**2 + np.tan(delta_lr)**2),1.)
		ss_rr = min(np.sqrt(s_rr**2 + np.tan(delta_rr)**2),1.)

		u_rf = self.umax - (self.umax - self.umin)*ss_rf
		u_lf = self.umax - (self.umax - self.umin)*ss_lf
		u_lr = self.umax - (self.umax - self.umin)*ss_lr
		u_rr = self.umax - (self.umax - self.umin)*ss_rr


		s_crit_rf = np.abs((u_rf * self.Fzgrf) / (2 * Cxf))	
		s_crit_lf = np.abs((u_lf * self.Fzglf) / (2 * Cxf))
		s_crit_lr = np.abs((u_lr * self.Fzglr) / (2 * Cxr))
		s_crit_rr = np.abs((u_rr * self.Fzgrr) / (2 * Cxr))



		## Now all the longitudinal forces
		if(np.abs(s_rf)  < s_crit_rf):
			Fxtrf=Cxf*s_rf*smth
		else:
			Fxtrf_1 = u_rf * np.abs(self.Fzgrf)
			Fxtrf_2 = np.abs((u_rf * self.Fzgrf)**2 / (4 * s_rf * Cxf))
			Fxtrf = np.sign(s_rf)*(Fxtrf_1 - Fxtrf_2)*smth
			

		if(np.abs(s_lf)  < s_crit_lf):
			Fxtlf=Cxf*s_lf*smth
		else:
			Fxtlf_1 = u_lf * np.abs(self.Fzglf)
			Fxtlf_2 = np.abs((u_lf * self.Fzglf)**2 / (4 * s_lf * Cxf))
			Fxtlf = np.sign(s_lf)*(Fxtlf_1 - Fxtlf_2)*smth
			# print(f"{Fxtlf_1 - Fxtlf_2}")


		if(np.abs(s_lr)  < s_crit_lr):
			Fxtlr=Cxr*s_lr*smth
		else:
			Fxtlr_1 = u_lr * np.abs(self.Fzglr)
			Fxtlr_2 = np.abs((u_lr * self.Fzglr)**2 / (4 * s_lr * Cxr))
			Fxtlr = np.sign(s_lr)*(Fxtlr_1 - Fxtlr_2)*smth

		if(np.abs(s_rr)  < s_crit_rr):
			Fxtrr=Cxr*s_rr*smth
		else:
			Fxtrr_1 = u_rr * np.abs(self.Fzgrr)
			Fxtrr_2 = np.abs((u_rr * self.Fzgrr)**2 / (4 * s_rr * Cxr))
			Fxtrr = np.sign(s_rr)*(Fxtrr_1 - Fxtrr_2)*smth
		


		##First the lateral critical slip
		al_crit_rf = np.arctan((3*u_rf * np.abs(self.Fzgrf))/Cf)
		al_crit_lf = np.arctan((3*u_lf * np.abs(self.Fzglf))/Cf)
		al_crit_lr = np.arctan((3*u_lr * np.abs(self.Fzglr))/Cr)
		al_crit_rr = np.arctan((3*u_rr * np.abs(self.Fzgrr))/Cr)


		## Now the lateral forces
		if(np.abs(delta_rf) <= al_crit_rf):
			h_ = 1 - ((Cf * np.abs(np.tan(delta_rf))) / (3 * u_rf * np.abs(self.Fzgrf)))
			Fytrf = -u_rf * np.abs(self.Fzgrf) * (1-h_**3)*np.sign(delta_rf)*smth
		else:
			Fytrf = -u_rf * np.abs(self.Fzgrf) * np.sign(delta_rf)*smth

		if(np.abs(delta_lf) <= al_crit_lf):
			h_ = 1 - ((Cf * np.abs(np.tan(delta_lf))) / (3 * u_lf * np.abs(self.Fzglf)))
			Fytlf = -u_lf * np.abs(self.Fzglf) * (1-h_**3)*np.sign(delta_lf)*smth
		else:
			Fytlf = -u_lf * np.abs(self.Fzglf) * np.sign(delta_lf)*smth


		if(np.abs(delta_lr) <= al_crit_lr):
			h_ = 1 - ((Cr * np.abs(np.tan(delta_lr))) / (3 * u_lr * np.abs(self.Fzglr)))
			Fytlr = -u_lr * np.abs(self.Fzglr) * (1-h_**3)*np.sign(delta_lr)*smth
		else:
			Fytlr = -u_lr * np.abs(self.Fzglr) * np.sign(delta_lr)*smth

		if(np.abs(delta_rr) <= al_crit_rr):
			h_ = 1 - ((Cr * np.abs(np.tan(delta_rr))) / (3 * u_rr * np.abs(self.Fzgrr)))
			Fytrr = -u_rr * np.abs(self.Fzgrr) * (1-h_**3)*np.sign(delta_rr)*smth
		else:
			Fytrr = -u_rr * np.abs(self.Fzgrr) * np.sign(delta_rr)*smth

		# print(f"{Fytrr =}, {Fytlr = }, {Fytlf = }, {Fytrf = }")
		## the forces Fxgij obtained by resolving the longitudinal and cornering forces at the tire contact patch 
		Fxglf=Fxtlf*np.cos(self.steering(t)* self.max_steer)-Fytlf*np.sin(self.steering(t)* self.max_steer) 
		Fxgrf=Fxtrf*np.cos(self.steering(t)* self.max_steer)-Fytrf*np.sin(self.steering(t)* self.max_steer) 
		Fxglr=Fxtlr 
		Fxgrr=Fxtrr 
		Fyglf=Fxtlf*np.sin(self.steering(t)* self.max_steer)+Fytlf*np.cos(self.steering(t)* self.max_steer) 
		Fygrf=Fxtrf*np.sin(self.steering(t)* self.max_steer)+Fytrf*np.cos(self.steering(t)* self.max_steer) 
		Fyglr=Fytlr 
		Fygrr=Fytrr

		# print(f"{Fxglf=}")
		# Some other constants used in the differential equations
		
		
		E1=-mt*wz*u+(Fyglf+Fygrf+Fyglr+Fygrr) 
		E2=(Fyglf+Fygrf)*a-(Fyglr+Fygrr)*b+(Fxgrf-Fxglf)*cf/2+(Fxgrr-Fxglr)*cr/2+(mur*b-muf*a)*wz*u
		E3=m*g*hrc*phi-(krof+kror)*phi-(brof+bror)*wx+hrc*m*wz*u 
		A1=mur*b-muf*a 
		A2=Jx+m*hrc**2 
		A3=hrc*m 


		#Chassis Model
		u_dot = wz*v+(1/mt)*((Fxglf+Fxgrf+Fxglr+Fxgrr)+(muf*a-mur*b)*(wz)**2-2*hrc*m*wz*wx) 
		# print(f"{Fxglf+Fxgrf+Fxglr+Fxgrr}")
		v_dot=(E1*Jxz**2-A1*A2*E2+A1*E3*Jxz+A3*E2*Jxz-A2*E1*Jz-A3*E3*Jz)/(A2*A1**2-2*A1*A3*Jxz+Jz*A3**2+mt*Jxz**2-A2*Jz*mt)
		wx_dot=(A1**2*E3-A1*A3*E2+A1*E1*Jxz-A3*E1*Jz+E2*Jxz*mt-E3*Jz*mt)/(A2*A1**2-2*A1*A3*Jxz+Jz*A3**2+mt*Jxz**2-A2*Jz*mt) 
		wz_dot=(A3**2*E2-A1*A2*E1-A1*A3*E3+A3*E1*Jxz-A2*E2*mt+E3*Jxz*mt)/(A2*A1**2-2*A1*A3*Jxz+Jz*A3**2+mt*Jxz**2-A2*Jz*mt)
		dx = u*np.cos(psi) - v*np.sin(psi)
		dy = u*np.sin(psi) + v*np.cos(psi)
		dpsi=wz
		dphi=wx





		# Smoothing for the 8dof model depends on the velocity - Similar to chrono implementation
		vx_min = 0.125
		vx_max = 0.5

		smth_rr = self.smooth_step(np.abs(ugrf),0,vx_min,1,vx_max)

		rolling_res_lf = -rr * np.abs(self.Fzglf) * np.sign(wlf)*smth_rr
		rolling_res_rf = -rr * np.abs(self.Fzgrf) * np.sign(wrf)*smth_rr
		rolling_res_lr = -rr * np.abs(self.Fzglr) * np.sign(wlr)*smth_rr
		rolling_res_rr = -rr * np.abs(self.Fzgrr) * np.sign(wrr)*smth_rr

		# print(rolling_res_lf)

		## Wheel rotational modelling
		# print(f"{Fxtlf = },{Rlf = }, {Jw = }")
		# dwlf=(1/Jw)*(self.torque(t) - Fxtlf*Rlf)
		# dwrf=(1/Jw)*(self.torque(t) - Fxtrf*Rrf)
		# dwlr=(1/Jw)*(self.torque(t) - Fxtlr*Rlr)
		# dwrr=(1/Jw)*(self.torque(t) - Fxtrr*Rrr)
		# print(f"drive torque : {self.drive_torque(t,wlf)/4}, angular vel : {wlf}")
		if(s_lf > 0.25):
			a__ = 2
		dwlf=(1/Jw)*(self.drive_torque(t,wlf)/4 + rolling_res_lf - Fxtlf*Rlf)
		dwrf=(1/Jw)*(self.drive_torque(t,wrf)/4 + rolling_res_rf - Fxtrf*Rrf)
		dwlr=(1/Jw)*(self.drive_torque(t,wlr)/4 + rolling_res_lr - Fxtlr*Rlr)
		dwrr=(1/Jw)*(self.drive_torque(t,wrr)/4 + rolling_res_rr - Fxtrr*Rrr) 
		self.dt.append(self.drive_torque(t,wlr)/4)
		self.fdt.append(Fxtlf*Rlf)
		self.rdt.append(rolling_res_lf)
		# print(f"{wlf=}")
		# dwlf=(1/Jw)*(700 + rolling_res_lf - Fxtlf*Rlf)
		# dwrf=(1/Jw)*(700 + rolling_res_rf - Fxtrf*Rrf)
		# dwlr=(1/Jw)*(700 + rolling_res_lr - Fxtlr*Rlr)
		# dwrr=(1/Jw)*(700 + rolling_res_rr - Fxtrr*Rrr)



		##The normal forces at four tires are determined as in order to update the tire compression for the next time step
		Z1=(m*g*b)/(2*(a+b))+(muf*g)/2 
		Z2=((muf*huf)/cf+m*b*(h-hrcf)/(cf*(a+b)))*(v_dot+wz*u) 
		Z3=(krof*phi+brof*dphi)/cf 
		Z4=((m*h+muf*huf+mur*hur)*(u_dot-wz*v))/(2*(a+b)) 
		self.Fzglf=(Z1-Z2-Z3-Z4) 
		self.Fzgrf=(Z1+Z2+Z3-Z4) 
		Z5=(m*g*a)/(2*(a+b))+(mur*g)/2 
		Z6=((mur*hur)/cr+m*a*(h-hrcr)/(cr*(a+b)))*(v_dot+wz*u) 
		Z7=(kror*phi+bror*dphi)/cr 
		Z8=((m*h+muf*huf+mur*hur)*(u_dot-wz*v))/(2*(a+b)) 
		self.Fzglr=(Z5-Z6-Z7+Z8) 
		self.Fzgrr=(Z5+Z6+Z7+Z8)

		self.flf.append(self.Fzglf)
		self.flr.append(self.Fzglr)
		self.frf.append(self.Fzgrf)
		self.frr.append(self.Fzgrr)
		# print(f"{self.Fzglf = },{self.Fzgrf = }, {self.Fzglr = }, {self.Fzgrr = }") 
		# print(f" {dphi = }, {u_dot = },{self.Fzglr = }, {self.Fzgrr = }") 

		if(self.Fzgrf < 0):
			self.Fzgrf = 0
		if(self.Fzglf < 0):
			self.Fzglf = 0
		if(self.Fzglr < 0):
			self.Fzglr = 0
		if(self.Fzgrr < 0):
			self.Fzgrr = 0

		self.xtlf=self.Fzglf/ktf 
		self.xtrf=self.Fzgrf/ktf 
		self.xtlr=self.Fzglr/ktr 
		self.xtrr=self.Fzgrr/ktr 

		return np.stack([dx,dy,u_dot,v_dot,dpsi,dphi,wx_dot,wz_dot,dwlf,dwlr,dwrf,dwrr])

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







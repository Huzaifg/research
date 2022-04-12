
import numpy as np
from math import atan,cos,sin
from scipy.integrate import solve_ivp
# import warnings
# warnings.filterwarnings("error")


def vehicle_8dof(theta,tt,st_input,init_cond):

    """
    The 8dof vehicle model that is our physical model for bayesian inference.
    This is adopted from the matlab script - https://github.com/uwsbel/projectlets/tree/master/model-repo/misc/2019/vehicles-matlab/8_DOF

    Parameters:
    ----------

    theta : 
        A list of all the parameters of the model. These are also all the parameters that are inferred from the bayesian inference. 
        Need to find a way to pass a dict - list is too error prone - pymc3 only takes lists
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

    ### Our model parameters, some are sampled some are not

    # m=theta[0]  # Sprung mass (kg)
    # Jx=theta[1]   # Sprung mass roll inertia (kg.m^2)
    # Jy=theta[2]   # Sprung mass pitch inertia (kg.m^2)
    # Jz=theta[3]   # Sprung mass yaw inertia (kg.m^2)
    # a=theta[4]   # Distance of sprung mass c.g. from front axle (m)
    # b=theta[5]   # Distance of sprung mass c.g. from rear axle (m)
    # Jxz=theta[6]   # Sprung mass XZ product of inertia
    # Jw=theta[7]     #tire/wheel roll inertia kg.m^2
    # g=theta[8]    # acceleration of gravity 
    # h=theta[9]    # Sprung mass c.g. height (m)
    # cf=theta[10]    # front track width (m)
    # cr=theta[11]    # rear track width (m)
    # muf=theta[12]      #front unsprung mass (kg)
    # mur=theta[13]      #rear unsprung mass (kg)
    # ktf=theta[14]    #front tire stiffness (N/m)
    # ktr=theta[15]    #rear tire stiffness (N/m)
    # Cf=theta[16]   #front tire cornering stiffness (N/rad)
    # Cr=theta[17]   #rear tire cornering stiffness (N/rad)
    # Cxf=theta[18]   #front tire longitudinal stiffness (N)
    # Cxr=theta[19]   #rear tire longitudinal stiffness (N)
    # r0=theta[20]   #nominal tire radius (m)s
    # hrcf=theta[21]   #front roll center distance below sprung mass c.g.
    # hrcr=theta[22]    #rear roll center distance below sprung mass c.g.
    # krof=theta[23]   #front roll stiffness (Nm/rad)
    # kror=theta[24]   #rear roll stiffness (Nm/rad)
    # brof=theta[25]    #front roll damping coefficient (Nm.s/rad)
    # bror=theta[26]    #rear roll damping coefficient (Nm.s/rad)
    m=1400  # Sprung mass (kg)
    Jx=900  # Sprung mass roll inertia (kg.m^2)
    Jy=2000  # Sprung mass pitch inertia (kg.m^2)
    Jz=2420  # Sprung mass yaw inertia (kg.m^2)
    a=1.14  # Distance of sprung mass c.g. from front axle (m)
    b=1.4   # Distance of sprung mass c.g. from rear axle (m)
    Jxz=90  # Sprung mass XZ product of inertia
    Jw=1    #tire/wheel roll inertia kg.m^2
    g=9.8    # acceleration of gravity 
    h=0.75   # Sprung mass c.g. height (m)
    cf=1.5   # front track width (m)
    cr=1.5   # rear track width (m)
    muf=80     #front unsprung mass (kg)
    mur=80     #rear unsprung mass (kg)
    ktf=200000   #front tire stiffness (N/m)
    ktr=200000   #rear tire stiffness (N/m)
    Cf=theta[0]  #front tire cornering stiffness (N/rad)
    Cr=theta[1]  #rear tire cornering stiffness (N/rad)
    Cxf=5000  #front tire longitudinal stiffness (N)
    Cxr=5000  #rear tire longitudinal stiffness (N)
    r0=0.285  #nominal tire radius (m)
    hrcf=0.65  #front roll center distance below sprung mass c.g.
    hrcr=0.6   #rear roll center distance below sprung mass c.g.
    krof=29000  #front roll stiffness (Nm/rad)
    kror=29000  #rear roll stiffness (Nm/rad)
    brof=3000   #front roll damping coefficient (Nm.s/rad)
    bror=3000   #rear roll damping coefficient (Nm.s/rad)



    ### The initial conditions

    u=init_cond['u']  # the longitudinal velocity 
    v=init_cond['v']     # the lateral velocity 
    u_dot=init_cond['u_dot']  # the longitudinal acceleration
    v_dot=init_cond['v_dot']  # the lateral acceleration
    phi=init_cond['phi']   # roll angle
    psi=init_cond['psi']   # yaw angle
    dphi=init_cond['dphi']  # roll angular velocity
    dpsi=init_cond['dpsi']   # yaw angular velocity
    wx=init_cond['wx']   # roll angular velocity
    wy=init_cond['wy']   # pitch angular velocity
    wz=init_cond['wz']   # yaw angular velocity
    wx_dot=init_cond['wx_dot']  # roll angular acceleration
    wz_dot=init_cond['wz_dot']  # yaw angular acceleration
    wlf=u/r0  # angular velocity of left front wheel rotation 
    wrf=u/r0  # angular velocity of right front wheel rotation
    wlr=u/r0  # angular velocity of left rear wheel rotation
    wrr=u/r0  # angular velocity of right rear wheel rotation
    ## the initial tire compression xti
    xtirf=((m*g*b)/(2*(a+b))+muf*g)/ktf  # Right front
    xtilf=((m*g*b)/(2*(a+b))+muf*g)/ktf  #left front
    xtilr=((m*g*a)/(2*(a+b))+mur*g)/ktr 
    xtirr=((m*g*a)/(2*(a+b))+mur*g)/ktr 
    xtlf=xtilf 
    xtrf=xtirf 
    xtlr=xtilr 
    xtrr=xtirr 

    ### Some calculated parameters

    hrc=(hrcf*b+hrcr*a)/(a+b) # the vertical distance from the sprung mass C.M. to the vehicle roll center.
    mt=m+2*muf+2*mur # vehicle total mass

    Tsim = len(tt) ## Total simulation steps
    delt = tt[1] - tt[0] ##the delta T

    #The steering input - given to us as an input
    delta4 = st_input

    long_vel,long_acc,roll_angle,lat_acc,lat_vel,psi_angle,yaw_rate,ay = (np.zeros(Tsim) for _ in range(8))
    
    #For the first time step
    long_vel[0] = u
    long_acc[0] = u_dot
    roll_angle[0] = phi
    lat_acc[0] = v_dot
    lat_vel[0] = v
    psi_angle[0] = psi
    yaw_rate[0] = dpsi


    ### Run the simulation - scary equations 
    # for i,t in enumerate(tt):
    for i in range(0,Tsim-1):
        #The steering input at that time step
        delta=delta4[i]
        #the instantaneous tire radius
        Rrf=r0-xtrf
        Rlf=r0-xtlf
        Rlr=r0-xtlr
        Rrr=r0-xtrr
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
        delta_rf=atan(vgrf/ugrf)-delta 
        delta_lf=atan(vglf/uglf)-delta 
        delta_lr=atan(vglr/uglr) 
        delta_rr=atan(vgrr/ugrr) 
        ##linear tire lateral force
        Fytrf=Cf*delta_rf 
        Fytlf=Cf*delta_lf 
        Fytlr=Cr*delta_lr 
        Fytrr=Cr*delta_rr 
        ## longitudinal slips
        s_rf=(Rrf*wrf-(ugrf*cos(delta)+vgrf*sin(delta)))/abs(ugrf*cos(delta)+vgrf*sin(delta)) 
        s_lf=(Rlf*wlf-(uglf*cos(delta)+vglf*sin(delta)))/abs(uglf*cos(delta)+vglf*sin(delta)) 
        s_lr=(Rlr*wlr-uglr)/abs(uglr) 
        s_rr=(Rrr*wrr-ugrr)/abs(ugrr) 
        ## linear tire longitudinal force 
        Fxtrf=Cxf*s_rf 
        Fxtlf=Cxf*s_lf 
        Fxtlr=Cxr*s_lr 
        Fxtrr=Cxr*s_rr 
        ## the forces Fxgij obtained by resolving the longitudinal and cornering forces at the tire contact patch 
        Fxglf=Fxtlf*cos(delta)-Fytlf*sin(delta) 
        Fxgrf=Fxtrf*cos(delta)-Fytrf*sin(delta) 
        Fxglr=Fxtlr 
        Fxgrr=Fxtrr 
        Fyglf=Fxtlf*sin(delta)+Fytlf*cos(delta) 
        Fygrf=Fxtrf*sin(delta)+Fytrf*cos(delta) 
        Fyglr=Fytlr 
        Fygrr=Fytrr 

        ##The normal forces at four tires are determined as
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

        xtlf=Fzglf/ktf 
        xtrf=Fzgrf/ktf 
        xtlr=Fzglr/ktr 
        xtrr=Fzgrr/ktr 

        ## Skipping some storage as we are not using it anywhere anyways

        ##chassis equations
        dpsi=wz 
        dphi=wx 
        E1=-mt*wz*u+(Fyglf+Fygrf+Fyglr+Fygrr) 
        E2=(Fyglf+Fygrf)*a-(Fyglr+Fygrr)*b+(Fxgrf-Fxglf)*cf/2+(Fxgrr-Fxglr)*cr/2+(mur*b-muf*a)*wz*u
        E3=m*g*hrc*phi-(krof+kror)*phi-(brof+bror)*dphi+hrc*m*wz*u 
        A1=mur*b-muf*a 
        A2=Jx+m*hrc**2 
        A3=hrc*m 
        u_dot=wz*v+(1/mt)*((Fxglf+Fxgrf+Fxglr+Fxgrr)+(muf*a-mur*b)*(wz)**2-2*hrc*m*wz*wx) 
        v_dot=(E1*Jxz**2-A1*A2*E2+A1*E3*Jxz+A3*E2*Jxz-A2*E1*Jz-A3*E3*Jz)/(A2*A1**2-2*A1*A3*Jxz+Jz*A3**2+mt*Jxz**2-A2*Jz*mt)
        wx_dot=(A1**2*E3-A1*A3*E2+A1*E1*Jxz-A3*E1*Jz+E2*Jxz*mt-E3*Jz*mt)/(A2*A1**2-2*A1*A3*Jxz+Jz*A3**2+mt*Jxz**2-A2*Jz*mt) 
        wz_dot=(A3**2*E2-A1*A2*E1-A1*A3*E3+A3*E1*Jxz-A2*E2*mt+E3*Jxz*mt)/(A2*A1**2-2*A1*A3*Jxz+Jz*A3**2+mt*Jxz**2-A2*Jz*mt)

        ## Solving the ODEs, the solutions differ a bit from Matlab at end time
        ts = np.arange((i)*delt,(i+1)*delt,delt/10)
        sol = solve_ivp(lambda t,psi: dpsi,(ts[0],ts[-1]),[psi],t_eval = ts,method = 'RK45',rtol=1e-4, atol=1e-8)
        psi = sol.y[-1][-1]
        sol = solve_ivp(lambda t,phi: dphi,(ts[0],ts[-1]),[phi],t_eval = ts,method = 'RK45',rtol=1e-4, atol=1e-8)
        phi = sol.y[-1][-1]
        sol= solve_ivp(lambda t,u: u_dot,(ts[0],ts[-1]),[u],t_eval = ts,method = 'RK45',rtol=1e-4, atol=1e-8)
        u = sol.y[-1][-1]
        sol = solve_ivp(lambda t,v: v_dot,(ts[0],ts[-1]),[v],t_eval = ts,method = 'RK45',rtol=1e-4, atol=1e-8)
        v = sol.y[-1][-1]
        sol = solve_ivp(lambda t,wx: wx_dot,(ts[0],ts[-1]),[wx],t_eval = ts,method = 'RK45',rtol=1e-4, atol=1e-8)
        wx = sol.y[-1][-1]
        sol = solve_ivp(lambda t,wz: wz_dot,(ts[0],ts[-1]),[wz],t_eval = ts,method = 'RK45',rtol=1e-4, atol=1e-8)
        wz = sol.y[-1][-1]
        dpsi=wz
        dphi=wx
        ## Wheel rotational modelling
        dwlf=-(1/Jw)*Fxtlf*Rlf
        dwrf=-(1/Jw)*Fxtrf*Rrf
        dwlr=-(1/Jw)*Fxtlr*Rlr
        dwrr=-(1/Jw)*Fxtrr*Rrr
        ## Solving the related ODE's
        sol = solve_ivp(lambda t,wlf: dwlf,(ts[0],ts[-1]),[wlf],t_eval = ts,method = 'RK45',rtol=1e-4, atol=1e-8)
        wlf = sol.y[-1][-1]
        sol = solve_ivp(lambda t,wrf: dwrf,(ts[0],ts[-1]),[wrf],t_eval = ts,method = 'RK45',rtol=1e-4, atol=1e-8)
        wrf = sol.y[-1][-1]
        sol = solve_ivp(lambda t,wlr: dwlr,(ts[0],ts[-1]),[wlr],t_eval = ts,method = 'RK45',rtol=1e-4, atol=1e-8)
        wlr = sol.y[-1][-1]
        sol = solve_ivp(lambda t,wrr: dwrr,(ts[0],ts[-1]),[wrr],t_eval = ts,method = 'RK45',rtol=1e-4, atol=1e-8)
        wrr = sol.y[-1][-1]
        ind = i + 1
        long_vel[ind]=u
        long_acc[ind]=u_dot
        roll_angle[ind]=phi
        lat_acc[ind]=v_dot
        lat_vel[ind]=v
        psi_angle[ind]=psi
        yaw_rate[ind]=wz
        ## vehicle lateral acceleration
        ay[ind]=u*wz+v_dot

    #For now lets just return one of the vectors, we will first work on just using one of the outputs for the bayesian inference
    # mod_data = np.array([long_vel,long_acc,roll_angle,lat_acc,lat_vel,psi_angle,yaw_rate,ay])
    # mod_data = np.array([roll_angle,ay,yaw_rate,lat_vel,psi_angle,long_vel])
    mod_data = np.array([lat_vel,psi_angle,yaw_rate,ay])
    return mod_data






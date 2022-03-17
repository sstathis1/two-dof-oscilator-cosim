import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class odfOscilatorDisp:
    """
    Contains the model of a one degree of freedom linear oscilator with damping, 
    which oscilates due to a displacement on it's mass.

    @author: Stefanos Stathis
    """
    
    def __init__(self, m, k, c, cc, integration_method = 'Newmark'):
        """
        Initializes the model object.

        Inputs: 
        m : Mass (kg)
        k : Stifness (N/m)
        c : Damping coefficient (Ns/m)
        cc : Coupling damping coefficient (Ns/m)
        integration_method : How to integrate the model internally (defaults to Newmark) options: Newmark, RK45.
        """
        self.cc = cc
        self.m = m
        self.k = k
        self.c = c
        self.integration_method = integration_method
    
       
    def getInitials(self, X0):
        """
        Gets the initial state of the model

        Input: 
        X0 : Initial state of the model (x, v)
        """
        self.X0 = X0
        
    
    def getTime(self, t_eval):
        """
        Get's the simulation period of the model

        Input:
        teval : Time window (tin, tf)
        """
        self.t0 = t_eval[0]
        self.tf = t_eval[-1]
        
    
    def getStateSpaceMatrices(self, C, D):
        """
        Get's the state space matrices

        Inputs:
        C : State matrix
        D : Input matrix
        example : y(t) = C * x(t) + D * u(t)
        """
        self.C = C
        self.D = D


    def extrapolateInput(self, u1, u2, time):
        """
        Extrapolates the input 

        Inputs: 
        u1 : Vector of the displacement of the mass at previous time points.
        u2 : Vector of the velocity of the mass at previous time points.
        time: Vector of the corresponding previous time points.
        """
        self.input1 = interp1d(time, u1, kind=len(time)-1, fill_value='extrapolate')
        self.input2 = interp1d(time, u2, kind=len(time)-1, fill_value='extrapolate')
        
        
    def interpolateInput(self, u1, u2, time):
        """
        Interpolates the input 

        Inputs: 
        u1 : Vector of the displacement of the mass at previous time points.
        u2 : Vector of the velocity of the mass at previous time points.
        time: Vector of the corresponding previous time points.
        """
        self.input1 = interp1d(time, u1, kind=len(time)-1, bounds_error=False, fill_value=u1[-1])
        self.input2 = interp1d(time, u2, kind=len(time)-1, bounds_error=False, fill_value=u2[-1])
        
        
    def ode(self, t, x):
        """
        The ordinary differential equation we want to solve

        Inputs:
        t : time (s)
        x : state [x (m), v (m/s)]
        
        Returns:
        The solution of each differential equation.
        (x(t), v(t))
        """
        return([x[1], -(2*self.k)/self.m*x[0] -(self.c+self.cc)/self.m*x[1] 
               + self.k/self.m*self.input1(t) + self.cc/self.m*self.input2(t)]) 
    
    
    def solve(self, h=1e-3):
        """
        Solves the differential equation starting from initial conditions X0 for the time window specified from teval.

        Input:
        h : micro step size (defaults to 0.001) should be at least 0.1 * Macro step size.

        Returns:
        The solution of the differential equation at tfinal specified in the teval input.
        (x(tf), v(tf))
        
        """
        if self.integration_method == 'Newmark':
            self.micro_steps = int((self.tf - self.t0)/h)
            t = np.linspace(self.t0, self.tf, self.micro_steps+1)
            x_prev = self.X0[0]
            xdot_prev = self.X0[1]
            xddot_prev = (-(2*self.k)/self.m*x_prev -(self.c+self.cc)/self.m*xdot_prev 
                         + self.k/self.m*self.input1(self.t0) 
                         + self.cc/self.m*self.input2(self.t0))
            a = 1/4
            b = 1/2
            tmp1 = (1 + 2*self.k*h**2*a/self.m + (self.cc+self.c)/self.m*h*b)
            tmp2 = (2*self.k*h/self.m + (self.cc+self.c)/self.m)
            tmp3 = (2*self.k/self.m*h**2*(0.5-a) +(self.cc+self.c)/self.m*h*(1-b))
            for i in range(len(t)-1):
                xddot = ((-2*self.k/(self.m*tmp1)*x_prev - tmp2/tmp1*xdot_prev  
                        - tmp3/tmp1*xddot_prev) 
                        +self.k*self.input1(t[i+1])/(self.m*tmp1)
                        +self.cc*self.input2(t[i+1])/(self.m*tmp1))
                xdot = xdot_prev + h*((1-b)*xddot_prev + b*xddot)
                x = x_prev + h*xdot_prev + h**2*((0.5-a)*xddot_prev +a*xddot)
                x_prev = x
                xdot_prev = xdot
                xddot_prev = xddot
            states = np.array([[x], [xdot]]).reshape(2, 1)
            self.input = np.array([self.input1(self.tf), self.input2(self.tf)]).reshape(2, 1)
            out = np.dot(self.C, states) + np.dot(self.D, self.input)
        else:
            sol = solve_ivp(self.ode, [self.t0, self.tf], self.X0, 
                            method=self.integration_method, max_step=h)
            states = np.array([[sol.y[0][-1]], [sol.y[1][-1]]]).reshape(2, 1)
            self.input = np.array([self.input1(self.tf), self.input2(self.tf)]).reshape(2, 1)
            out = np.dot(self.C, states) + np.dot(self.D, self.input)
        return(states, out.reshape(-1, 1))
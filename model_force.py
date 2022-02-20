"""
Created on Fri Nov  5 18:38:10 2021

@author: MrStevenn007
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class odfOscilatorForce:
    
    
    def __init__(self, m, k, c, cc, integration_method = 'Newmark'):
        self.cc = cc
        self.m = m
        self.k = k
        self.c = c
        self.integration_method = integration_method
    
       
    def getInitials(self, X0):
        self.X0 = X0
        
    
    def getTime(self, t_eval):
        self.t0 = t_eval[0]
        self.tf = t_eval[-1]
    

    def getStateSpaceMatrices(self, C, D):
        self.C = C
        self.D = D


    def extrapolateInput(self, u, time):
        self.input = interp1d(time, u, kind=len(time)-1, fill_value='extrapolate')


    def ode(self, t, x):
        return([x[1], -self.k/self.m*x[0] - self.c/self.m*x[1] + self.input(t)/self.m])  


    def solve(self, h=1e-3):
        if self.integration_method == 'Newmark':
            self.micro_steps = int((self.tf - self.t0)/h)
            t = np.linspace(self.t0, self.tf, self.micro_steps+1)
            x_prev = self.X0[0]
            xdot_prev = self.X0[1]
            xddot_prev = (-self.k/self.m*x_prev - self.c/self.m*xdot_prev 
                          + self.input(self.t0)/self.m)
            a = 1/4
            b = 1/2
            for i in range(len(t)-1):
                tmp1 = (1 + self.k*h**2*a/self.m + self.c/self.m*h*b)
                tmp2 = (self.k*h/self.m + self.c/self.m)
                tmp3 = (self.k/self.m*h**2*(0.5-a) + self.c/self.m*h*(1-b))
                xddot = (-self.k/(self.m*tmp1)*x_prev - tmp2/tmp1*xdot_prev  
                        - tmp3/tmp1*xddot_prev) + self.input(t[i+1])/(self.m*tmp1)
                xdot = xdot_prev + h*((1-b)*xddot_prev + b*xddot)
                x = x_prev + h*xdot_prev + h**2*((0.5-a)*xddot_prev +a*xddot)
                x_prev = x
                xdot_prev = xdot
                xddot_prev = xddot
            states = np.array([[x], [xdot]]).reshape(2, 1)
            out = np.dot(self.C, states) + np.dot(self.D, self.input(self.tf))
        else:
            sol = solve_ivp(self.ode, [self.t0, self.tf], self.X0, 
                            method=self.integration_method, max_step=h)
            states = np.array([[sol.y[0][-1]], [sol.y[1][-1]]]).reshape(2, 1)
            out = np.dot(self.C, states) + np.dot(self.D, self.input(self.tf))
        return(states, out.reshape(-1, 1))
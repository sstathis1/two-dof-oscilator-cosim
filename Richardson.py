import numpy as np
import matplotlib.pyplot as plt
from  model_disp import odfOscilatorDisp
from model_force import odfOscilatorForce
import sys
from math import cos, sin, sqrt, exp					


class Orchestrator():
    """
    A master algorithm for explicit co-simulation of 2 slave models. This model does not contain an automatic step size control
    technique. It uses Richardson extrapolation technique to compute the local error in each fixed time step.

    @author: Stefanos Stathis
    """    
    
    def __init__(self, H, k, tf, kc, cc, cosiMethod = 'Jacobi'):
        """ 
        Initializes the Master object
        
        Inputs: 
        H : Macro Step (s)
        k : Polynomial degree
        tf : Final time of simulation (s)
        kc : Coupling stifness (N/m)
        cc : Coupling damping coefficient (Ns/m)
        cosiMethod : Comunication method for the co-simulation (defaults to Parallel Jacobi)
        """
        self.stepDuration = H
        self.polyDegree = k
        self.endTime = tf
        self.kc = kc
        self.cc = cc
        self.cosiMethod = cosiMethod
        # The macro step sizes are actually H / 2.
        self.macroSteps = int(2*self.endTime/self.stepDuration)
        self.Z1 = np.zeros([2, self.macroSteps+1])
        self.Z2 = np.zeros([2, self.macroSteps+1])							  									  
        self.time = np.linspace(0, self.endTime, self.macroSteps+1)
        self.currentMacro = 0
        self.doubleStepCheck = False
        self.tmp = 0
        # Error estimation matrices
        self.ESTY1 = np.zeros([1, int(self.macroSteps/2)+1])
        self.ESTY2 = np.zeros([1, int(self.macroSteps/2)+1])
    
    
    def setModel1(self, m, k, c, oscMethod, integrationMethod, h=1e-3):
        """
        Creates an object for the first model
        
        Input:
        m : mass (kg)
        k : sifness (Nm)
        c : damping coefficient (Nm/s)
        oscMethod : oscilation method -- 'Force' / 'Disp'
        integrationMethod : what algorithm to use to integrate the model -- "Newmark" / "RK45"
        h : micro step for Newmark
        """
        self.microStep1 = h
        self.oscMethod1 = oscMethod
        if self.oscMethod1 == 'Force':
            self.S1 = odfOscilatorForce(m, k, c, self.cc, integrationMethod)
            self.U1 = np.zeros((1, self.macroSteps+1))
        else:
            self.S1 = odfOscilatorDisp(m, k, c, self.cc, integrationMethod)
            self.U1 = np.zeros((2, self.macroSteps+1))
        
    
    def setModel2(self, m, k, c, oscMethod, integrationMethod, h=1e-3):
        """
        Creates an object for the second model
        
        Input:
        m : mass (kg)
        k : sifness (Nm)
        c : damping coefficient (Nm/s)
        oscMethod : oscilation method -- 'Force' / 'Disp'
        integrationMethod : what algorithm to use to integrate the model -- "Newmark" / "RK45"
        h : micro step for Newmark
        """
        self.microStep2 = h
        self.oscMethod2 = oscMethod
        if self.oscMethod2 == 'Force':
            self.S2 = odfOscilatorForce(m, k, c, self.cc, integrationMethod)
            self.U2 = np.zeros((1, self.macroSteps+1))
        else:
            self.S2 = odfOscilatorDisp(m, k, c, self.cc, integrationMethod)
            self.U2 = np.zeros((2, self.macroSteps+1))
    
    
    def setStateSpaceMatrices(self):
        """Creates the state space matrices for the 2 subsystems"""
        if self.oscMethod1 == 'Force':
            self.C2 = np.array([self.kc, self.cc])
            self.D2 = np.array([-self.kc, -self.cc])
            self.Y2 = np.zeros([1, self.macroSteps+1])
        else:
            self.C2 = np.eye(2)
            self.D2 = np.zeros([len(self.U2), len(self.U2)])
            self.Y2 = np.zeros([2, self.macroSteps+1])

        if self.oscMethod2 == 'Force':
            self.C1 = np.array([-self.kc, -self.cc])
            self.D1 = np.array([self.kc, self.cc])
            self.Y1 = np.zeros([1, self.macroSteps+1])
        else:
            self.C1 = np.eye(2)
            self.D1 = np.zeros([len(self.U1), len(self.U1)])
            self.Y1 = np.zeros([2, self.macroSteps+1])

        if self.oscMethod1 == 'Force' and self.oscMethod2 == 'Force':
            self.C1 = np.zeros(2)
            self.C2 = np.zeros(2)
            self.D1 = np.array(-1)
            self.D2 = np.array(-1)
        self.S1.getStateSpaceMatrices(self.C1, self.D1)
        self.S2.getStateSpaceMatrices(self.C2, self.D2)


    def setStates(self, state1, state2):
        """Saves the current state of the 2 subsystems"""
        self.Z1[:, self.currentMacro:self.currentMacro+1] = state1
        self.Z2[:, self.currentMacro:self.currentMacro+1] = state2			


    def setOutputs(self, output1, output2):
        """Saves the outputs of the 2 subsystems"""
        self.Y1[:, self.currentMacro:self.currentMacro+1] = output1
        self.Y2[:, self.currentMacro:self.currentMacro+1] = output2
        
    
    def setInput1(self):
        """Computes the input to the first subsystem"""
        if self.oscMethod2 == 'Force' and self.oscMethod1 == 'Force':
            self.U1[0, self.currentMacro:self.currentMacro+1] =(
                self.kc*(self.Z2[0, self.currentMacro] - self.Z1[0, self.currentMacro])
                + self.cc*(self.Z2[1, self.currentMacro] - self.Z1[1, self.currentMacro]))
        elif self.oscMethod1 == 'Disp':
            self.U1[0, self.currentMacro:self.currentMacro+1] = self.Y2[0, self.currentMacro:self.currentMacro+1]
            self.U1[1, self.currentMacro:self.currentMacro+1] = self.Y2[1, self.currentMacro:self.currentMacro+1]
        else:
            self.U1[0, self.currentMacro:self.currentMacro+1] = self.Y2[0, self.currentMacro:self.currentMacro+1]
        
        
    def setInput2(self):
        """Computes the input to the second subsystem"""
        if self.oscMethod2 == 'Force' and self.oscMethod1 == 'Force':
            self.U2[0, self.currentMacro:self.currentMacro+1] =-(
                self.kc*(self.Z2[0, self.currentMacro] - self.Z1[0, self.currentMacro])
                + self.cc*(self.Z2[1, self.currentMacro] - self.Z1[1, self.currentMacro]))
        elif self.oscMethod2 == 'Disp':
            self.U2[0, self.currentMacro:self.currentMacro+1] = self.Y1[0, self.currentMacro:self.currentMacro+1]
            self.U2[1, self.currentMacro:self.currentMacro+1] = self.Y1[1, self.currentMacro:self.currentMacro+1]
        else:
            self.U2[0, self.currentMacro:self.currentMacro+1] = self.Y1[0, self.currentMacro:self.currentMacro+1]
        
        
    def sortModels(self): 
        """Sorts the models for the Gauss Seidel method"""      
        if self.oscMethod1 == 'Disp' and self.oscMethod2 == 'Force':
            self.firstModel = self.S2
            self.firstMethod = self.oscMethod2
            self.secondModel = self.S1
            self.secondMethod = self.oscMethod1
            self.secondInput = self.U1
            self.secondMicroStep = self.microStep1
            self.secondOutput = self.Z1
        else:
            self.firstModel = self.S1
            self.firstMethod = self.oscMethod1
            self.secondModel = self.S2
            self.secondMethod = self.oscMethod2
            self.secondInput = self.U2
            self.secondMicroStep = self.microStep2
            self.secondOutput = self.Z2
        

    def giveMacroTime(self):
        """ Returns the current simulation time [Tn, Tn+1]"""
        return(self.time[self.currentMacro:self.currentMacro+2])
    
    
    def beginSimulation(self, z1, z2, y1, y2):	
        """ 
        Simulates the given Orchestrator object and returns the estimated errors of both outputs

        Inputs:
        z1 : initial state 1 [x, v]
        z2 : initial state 2 [x, v]
        y1 : initial output 1 [x, v] or f
        y2 : initial output 2 [x, v] or f

        Outputs:
        ESTY1 : error estimation of y1
        ESTY2 : error estimation of y2

        Alternatively Outputs:
        absoluteError1 : The absolute global error of the position of the first model.
        absoluteError2 : The absolute global error of the position of the second model.
        """	   
        self.setStateSpaceMatrices()
        self.setStates(z1, z2)
        self.setOutputs(y1, y2)
        self.setInput1()
        self.setInput2()
        self.sortModels()
        while self.currentMacro < self.macroSteps:
            for i in range(2):
                if self.doubleStepCheck == False:
                    self.z1Double, self.z2Double, self.y1Double, self.y2Double = self.doubleStep()
                    self.doubleStepCheck = True
                self.advanceStep()
            self.tmp += 1
            self.setErrorEstimate()
            self.doubleStepCheck = False
        self.analyticalSolution()
        self.calculateError()
        #return (self.absoluteError1, self.absoluteError2)
        return (self.ESTY1, self.ESTY2)
        
    
    def doubleStep(self):
        """Performs a double step for use in Richardson extrapolation and returns the states and the outputs that were computed"""
        if self.cosiMethod == 'Jacobi':
            z1Double, y1Double = self.simulateModel(self.S1, self.oscMethod1)
            z2Double, y2Double = self.simulateModel(self.S2, self.oscMethod2)
        else:
            if self.oscMethod1 == 'Force' and self.oscMethod2 == 'Force':
                z1Double, y1Double = self.simulateModel(self.firstModel, self.firstMethod)
                u, time = self.inputPredictionDoubleStep(self.secondModel, self.secondMethod)
                u = np.append(u, y1Double[0, :])
                time = np.append(time, self.time[self.currentMacro+2])
                u = np.delete(u, 0)
                time = np.delete(time, 0)
                self.secondModel.getTime(self.time[self.currentMacro:self.currentMacro+3:2])
                self.secondModel.getInitials(self.secondOutput[:, self.currentMacro])
                self.secondModel.extrapolateInput(u, time)
                z2Double, y2Double = self.secondModel.solve(self.secondMicroStep)
            else:
                z1Double, y1Double = self.simulateModel(self.firstModel, self.firstMethod)
                u1, u2, time = self.inputPredictionDoubleStep(self.secondModel, 
                                                            self.secondMethod)
                u1 = np.append(u1, y1Double[0, :])
                u2 = np.append(u2, y1Double[1, :])
                time = np.append(time, self.time[self.currentMacro+2])
                u1 = np.delete(u1, 0)
                u2 = np.delete(u2, 0)
                time = np.delete(time, 0)
                self.secondModel.getTime(self.time[self.currentMacro:self.currentMacro+3:2])
                self.secondModel.getInitials(self.secondOutput[:, self.currentMacro])
                self.secondModel.extrapolateInput(u1, u2, time)
                z2Double, y2Double = self.secondModel.solve(self.secondMicroStep)
        return(z1Double, z2Double, y1Double, y2Double)   

    
    def advanceStep(self):
        """Advances the time by 1 iteration and returns the states and outputs that were computed"""
        if self.cosiMethod == 'Jacobi':
            state1, output1 = self.simulateModel(self.S1, self.oscMethod1)
            state2, output2 = self.simulateModel(self.S2, self.oscMethod2)
            self.currentMacro += 1
            self.setStates(state1, state2)
            self.setOutputs(output1, output2)
            self.setInput1()
            self.setInput2()
        else:
            if self.oscMethod1 == 'Force' and self.oscMethod2 == 'Force':
                state1, output1 = self.simulateModel(self.firstModel, self.firstMethod)
                u, time = self.inputPrediction(self.secondModel, self.secondMethod)
                u = np.append(u, output1[0, :])
                time = np.append(time, self.time[self.currentMacro+1])
                u = np.delete(u, 0)
                time = np.delete(time, 0)
                self.secondModel.getTime(self.giveMacroTime())
                self.secondModel.getInitials(self.secondOutput[:, self.currentMacro])
                self.secondModel.extrapolateInput(u, time)
                state2, output2 = self.secondModel.solve(self.secondMicroStep)
            else:
                state1, output1 = self.simulateModel(self.firstModel, self.firstMethod)
                u1, u2, time = self.inputPrediction(self.secondModel, self.secondMethod)
                u1 = np.append(u1, output1[0, :])
                u2 = np.append(u2, output1[1, :])
                time = np.append(time, self.time[self.currentMacro+1])
                u1 = np.delete(u1, 0)
                u2 = np.delete(u2, 0)
                time = np.delete(time, 0)
                self.secondModel.getTime(self.giveMacroTime())
                self.secondModel.getInitials(self.secondOutput[:, self.currentMacro])
                self.secondModel.extrapolateInput(u1, u2, time)
                state2, output2 = self.secondModel.solve(self.secondMicroStep)
            self.currentMacro += 1
            self.setStates(state1, state2)
            self.setOutputs(output1, output2)
            self.setInput1()
            self.setInput2()
            

    def simulateModel(self, model, oscMethod):
        """Calls on the model to simulate it for a given time window using it's oscilation method"""
        if self.doubleStepCheck == True:
            model.getTime(self.giveMacroTime())
        else:
            model.getTime(self.time[self.currentMacro:self.currentMacro+3:2])
        if oscMethod == 'Force':
            if self.doubleStepCheck == True:
                u, time = self.inputPrediction(model, oscMethod)
            else:
                u, time = self.inputPredictionDoubleStep(model, oscMethod)
            model.extrapolateInput(u, time)
        else:
            if self.doubleStepCheck == True:
                u1, u2, time = self.inputPrediction(model, oscMethod)
            else:
                u1, u2, time = self.inputPredictionDoubleStep(model, oscMethod)
            model.extrapolateInput(u1, u2, time)
        if model == self.S1:
            model.getInitials(self.Z1[:, self.currentMacro])
            state, out = model.solve(self.microStep1)
        else:
            model.getInitials(self.Z2[:, self.currentMacro])    
            state, out = model.solve(self.microStep2)
        return(state, out)


    def inputPredictionDoubleStep(self, model, oscMethod):
        """
        Takes as input a model object and it's oscilation method and returns the values 
        that are to be used for extrapolation or interpolation of the input.
        
        Outputs:
        u : vector containing the values of input
        time : vector containing the time values
        """
        if self.currentMacro < 2*self.polyDegree:
            if model == self.S1:
                u = self.U1[:, 0:self.currentMacro+1:2]
            else:
                u = self.U2[:, 0:self.currentMacro+1:2]
            time = self.time[0:self.currentMacro+1:2]
        else:
            if model == self.S1:
                u = self.U1[:, self.currentMacro-2*self.polyDegree:self.currentMacro+1:2]
            else:
                u = self.U2[:, self.currentMacro-2*self.polyDegree:self.currentMacro+1:2]
            time = self.time[self.currentMacro-2*self.polyDegree:self.currentMacro+1:2]
        if oscMethod == 'Force':
            return(u.reshape(np.shape(u)[1]), time)
        else:
            return(u[0, :], u[1, :], time)


    def inputPrediction(self, model, oscMethod):
        """
        Takes as input a model object and it's oscilation method and returns the values 
        that are to be used for extrapolation or interpolation of the input.
        
        Outputs:
        u : vector containing the values of input
        time : vector containing the time values
        """
        if self.currentMacro < 2*self.polyDegree:
            if model == self.S1:
                u = self.U1[:, self.currentMacro-self.tmp:self.currentMacro+1]
            else:
                u = self.U2[:, self.currentMacro-self.tmp:self.currentMacro+1]
            time = self.time[self.currentMacro-self.tmp:self.currentMacro+1]
        else:
            if model == self.S1:
                u = self.U1[:, self.currentMacro-self.polyDegree:self.currentMacro+1]
            else:
                u = self.U2[:, self.currentMacro-self.polyDegree:self.currentMacro+1]
            time = self.time[self.currentMacro-self.polyDegree:self.currentMacro+1]
        if oscMethod == 'Force':
            return(u.reshape(np.shape(u)[1]), time)
        else:
            return(u[0, :], u[1, :], time)


    def analyticalSolution(self):
        """Computes and saves the analytical solution of the problem based on mathematical formulas"""
        k = self.S1.k
        m = self.S1.m
        c = self.S1.c
        w1 = sqrt(k/m)   # Πρώτη Ιδιοσυχνότητα
        w2 = sqrt(3*k/m) # Δεύτερη Ιδιοσυχνότητα
        t0 = sqrt(m/2)*np.array([self.Z1[0, 0] + self.Z2[0, 0], self.Z1[0, 0] - self.Z2[0, 0]])
        t0_dot = sqrt(m/2)*np.array([self.Z1[1, 0] + self.Z2[1, 0], self.Z1[1, 0] - self.Z2[1, 0]])
        z1 = 0.05
        z2 = 0.05 * sqrt(3)
        wd1 = w1*sqrt(1-z1**2)
        wd2 = w2*sqrt(1-z2**2)
        t1 = np.zeros((len(self.time), 1))
        t2 = np.zeros((len(self.time), 1))
        t1Dot = np.zeros((len(self.time), 1))
        t2Dot = np.zeros((len(self.time), 1))
        for i in range(len(self.time)):
            t1[i] = ((t0[0]*cos(wd1*self.time[i])
                    +((t0_dot[0]+z1*w1*t0[0])/wd1)
                    *sin(wd1*self.time[i]))*exp(-z1*w1*self.time[i]))
            t1Dot[i] = ((t0[0]*cos(wd1*self.time[i])
                         +((t0_dot[0]+z1*w1*t0[0])/wd1)
                         *sin(wd1*self.time[i]))*exp(-z1*w1*self.time[i])*(-z1*w1)
                        -t0[0]*wd1*sin(wd1*self.time[i])
                        +((t0_dot[0]+z1*w1*t0[0])/wd1)*wd1*cos(wd1*self.time[i])
                        *exp(-z1*w1*self.time[i]))
            t2[i] = ((t0[1]*cos(wd2*self.time[i])
                    +((t0_dot[1]+z2*w2*t0[1])/wd2)
                    *sin(wd2*self.time[i]))*exp(-z2*w2*self.time[i]))
            t2Dot[i] = ((t0[1]*cos(wd2*self.time[i])
                         +((t0_dot[1]+z2*w2*t0[1])/wd2)
                         *sin(wd2*self.time[i]))*exp(-z2*w2*self.time[i])*(-z2*w2)
                        -t0[1]*wd2*sin(wd2*self.time[i])
                        +((t0_dot[1]+z2*w2*t0[1])/wd2)*wd2*cos(wd2*self.time[i])
                        *exp(-z2*w2*self.time[i]))
        self.x1Analytical = 1/sqrt(2*m)*(t1 + t2).reshape(-1)
        self.v1Analytical = 1/sqrt(2*m)*(t1Dot + t2Dot).reshape(-1)
        self.x2Analytical = 1/sqrt(2*m)*(t1 - t2).reshape(-1)
        self.v2Analytical = 1/sqrt(2*m)*(t1Dot - t2Dot).reshape(-1)
        self.Z1Analytical = np.array([self.x1Analytical, self.v1Analytical])
        self.Z2Analytical = np.array([self.x2Analytical, self.v2Analytical])
        
        
    def calculateError(self):
        """
        Calculates the absolute global error for the positions of the numerical solution in respect to the analytical solution 
        and returns it
        """
        self.absoluteError1 = np.abs((self.Z1-self.Z1Analytical))
        self.absoluteError2 = np.abs((self.Z2-self.Z2Analytical))
        return(self.absoluteError1, self.absoluteError2)


    def setErrorEstimate(self):
        """Calculates and saves the local error estimation using richardson extrapolation"""
        self.ESTY1[0, self.tmp] = (2**(self.polyDegree+1)/(2**(self.polyDegree+1)-1)
                                    *np.linalg.norm(self.y1Double
                                        -self.Y1[:, self.currentMacro].reshape(-1,1)))
        self.ESTY2[0, self.tmp] = (2**(self.polyDegree+1)/(2**(self.polyDegree+1)-1)
                                    *np.linalg.norm(self.y2Double
                                    -self.Y2[:, self.currentMacro].reshape(-1,1)))



    def plotOutputs(self):
        """Plots the numerical positions x1num and x2num and compares them with the analytical positions x1anal, x2anal"""
        plt.figure(figsize=(14,8))
        plt.title('Απόκριση Διβάθμιου Ταλαντωτή μέσω Άμμεσης Συν-Προσομοίωσης' 
                  f' {self.cosiMethod} , {self.oscMethod1} - {self.oscMethod2}, k = {self.polyDegree}')
        plt.plot(self.time, self.Z1[0, :], label='$x_{1}$')
        plt.plot(self.time, self.Z2[0, :], label='$x_{2}$')
        plt.plot(self.time, self.x1Analytical, '--', label='$x_{1,analytical}$')
        plt.plot(self.time, self.x2Analytical, '--', label='$x_{2,analytical}$')
        plt.xlabel('time (sec)')
        plt.ylabel('x(t) (m)')
        plt.grid()
        plt.xlim(xmin=0, xmax=self.endTime)
        plt.legend()
        plt.show()   
        
        
    def plotLocalError(self):
        """Plots the local error estimations for y1, y2 that were computed"""
        plt.figure(figsize=(14,8))
        plt.title(f'Τοπικό σφάλμα άμεσης συν-προσομοίωσης με βήμα {self.stepDuration}' 
                  f' {self.cosiMethod}')
        plt.plot(self.time[0:self.macroSteps+1:2], self.ESTY1[0, :], 
                 label='$Richardson Extrapolation Error x_{1}$')
        plt.plot(self.time[0:self.macroSteps+1:2], self.ESTY2[0, :], 
                 label='$Richardson Extrapolation Error x_{2}$')
        plt.xlabel('time (sec)')
        plt.ylabel('Error(t) (-)')
        plt.grid()
        plt.xlim(xmin=0, xmax=self.endTime)
        plt.legend()
        plt.show()   
        
    def plotGlobalError(self):
        """
        Plots the absolute global error of the forces if oscilation method is 'Force' or of the positions if oscilation method 
        is 'Disp
        '"""
        plt.figure(figsize=(14,8))
        plt.title(f'Ολικό σφάλμα άμεσης συν-προσομοίωσης με σταθερό βήμα και μέθοδο' 
                  f' {self.cosiMethod}')
        plt.plot(self.time, self.absoluteError1[0, :], label='Global Absolute Error $x_1$')
        plt.plot(self.time, self.absoluteError2[0, :], label='Global Absolute Error $x_2$')
        plt.xlabel('time (sec)')
        plt.ylabel('Global Absolute Error(t)')
        plt.grid()
        plt.xlim(xmin=0, xmax=self.endTime)
        plt.legend()
        plt.show()
"""
Created on Mon Nov  8 18:08:55 2021

@author: MrStevenn007
"""

import numpy as np
import matplotlib.pyplot as plt
from  model_disp import odfOscilatorDisp
from model_force import odfOscilatorForce
import sys
from math import cos, sin, sqrt, exp

class Orchestrator():
    
    
    def __init__(self, time, k, tf, kc, cc, cosiMethod = 'Jacobi'):
        self.time = time
        self.polyDegree = k
        self.endTime = tf
        self.kc = kc
        self.cc = cc
        self.cosiMethod = cosiMethod
        self.macroSteps = len(self.time)-1
        self.Z1 = np.zeros([2, self.macroSteps+1])
        self.Z2 = np.zeros([2, self.macroSteps+1])
        self.localErrorY1 = np.zeros([1, self.macroSteps+1])
        self.localErrorY2 = np.zeros([1, self.macroSteps+1])
        self.currentMacro = 0
        
    
    def setModel1(self, m, k, c, oscMethod, integrationMethod, h=1e-3):
        self.microStep1 = h
        self.oscMethod1 = oscMethod
        if self.oscMethod1 == 'Force':
            self.S1 = odfOscilatorForce(m, k, c, self.cc, integrationMethod)
            self.U1 = np.zeros((1, self.macroSteps+1))
        else:
            self.S1 = odfOscilatorDisp(m, k, c, self.cc, integrationMethod)
            self.U1 = np.zeros((2, self.macroSteps+1))
        
    
    def setModel2(self, m, k, c, oscMethod, integrationMethod, h=1e-3):
        self.microStep2 = h
        self.oscMethod2 = oscMethod
        if self.oscMethod2 == 'Force':
            self.S2 = odfOscilatorForce(m, k, c, self.cc, integrationMethod)
            self.U2 = np.zeros((1, self.macroSteps+1))
        else:
            self.S2 = odfOscilatorDisp(m, k, c, self.cc, integrationMethod)
            self.U2 = np.zeros((2, self.macroSteps+1))
    

    def setStateSpaceMatrices(self):
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
        self.Z1[:, self.currentMacro:self.currentMacro+1] = state1
        self.Z2[:, self.currentMacro:self.currentMacro+1] = state2
        

    def setOutputs(self, output1, output2):
        self.Y1[:, self.currentMacro:self.currentMacro+1] = output1
        self.Y2[:, self.currentMacro:self.currentMacro+1] = output2
        
    
    def setInput1(self):
        if self.oscMethod1 == 'Force':
            self.U1[0, self.currentMacro:self.currentMacro+1] =(
                self.kc*(self.Z2[0, self.currentMacro] - self.Z1[0, self.currentMacro])
                + self.cc*(self.Z2[1, self.currentMacro] - self.Z1[1, self.currentMacro]))
        else:
            self.U1[0, self.currentMacro:self.currentMacro+1] = self.Z2[0, self.currentMacro:self.currentMacro+1]
            self.U1[1, self.currentMacro:self.currentMacro+1] = self.Z2[1, self.currentMacro:self.currentMacro+1]
        
        
    def setInput2(self):
        if self.oscMethod2 == 'Force':
            self.U2[0, self.currentMacro:self.currentMacro+1] =-(
                self.kc*(self.Z2[0, self.currentMacro] - self.Z1[0, self.currentMacro])
                + self.cc*(self.Z2[1, self.currentMacro] - self.Z1[1, self.currentMacro]))
        else:
            self.U2[0, self.currentMacro:self.currentMacro+1] = self.Z1[0, self.currentMacro:self.currentMacro+1]
            self.U2[1, self.currentMacro:self.currentMacro+1] = self.Z1[1, self.currentMacro:self.currentMacro+1]
        
        
    def sortModels(self):      
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
        return(self.time[self.currentMacro:self.currentMacro+2])
    
    
    def beginSimulation(self, z1, z2, y1, y2):
        self.setStateSpaceMatrices()
        self.setStates(z1, z2)
        self.setOutputs(y1, y2)
        self.analyticalSolution()
        self.calculateError(y1, y2)
        self.sortModels()
        while self.time[self.currentMacro] <= self.endTime:
            self.advanceStep()
        #self.plotLocalError()
        return self.localErrorY1, self.localErrorY2


    def advanceStep(self):
        if self.cosiMethod == 'Jacobi':
            self.setInput1()
            self.setInput2()
            state1, output1 = self.simulateModel(self.S1, self.oscMethod1)
            state2, output2 = self.simulateModel(self.S2, self.oscMethod2)
            self.currentMacro += 1
            self.calculateError(output1, output2)
            self.setStates(self.Z1Analytical[:, self.currentMacro:self.currentMacro+1], 
                            self.Z2Analytical[:, self.currentMacro:self.currentMacro+1])
            self.setOutputs(output1, output2)
        else:
            if self.oscMethod1 == 'Force' and self.oscMethod2 == 'Force':
                self.setInput1()
                self.setInput2()
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
                self.setInput1()
                self.setInput2()
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
            self.calculateError(output1, output2)
            self.setStates(self.Z1Analytical[:, self.currentMacro:self.currentMacro+1], 
                           self.Z2Analytical[:, self.currentMacro:self.currentMacro+1])
            self.setOutputs(output1, output2)
            

    def simulateModel(self, model, oscMethod):
        model.getTime(self.giveMacroTime())
        if oscMethod == 'Force':
            u, time = self.inputPrediction(model, oscMethod)
            model.extrapolateInput(u, time)
        else:
            u1, u2, time = self.inputPrediction(model, oscMethod)
            model.extrapolateInput(u1, u2, time)
        if model == self.S1:
            model.getInitials(self.Z1[:, self.currentMacro])
            sol = model.solve(self.microStep1)
        else:
            model.getInitials(self.Z2[:, self.currentMacro])    
            sol = model.solve(self.microStep2)
        return(sol)


    def inputPrediction(self, model, oscMethod):
        if self.currentMacro < self.polyDegree:
            if model == self.S1:
                u = self.U1[:, 0:self.currentMacro+1]
            else:
                u = self.U2[:, 0:self.currentMacro+1]
            time = self.time[0:self.currentMacro+1]
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
        k = self.S1.k
        m = self.S1.m
        c = self.S1.c
        w1 = sqrt(k/m)   # Πρώτη Ιδιοσυχνότητα
        w2 = sqrt(3*k/m) # Δεύτερη Ιδιοσυχνότητα
        t0 = sqrt(m/2)*np.array([self.Z1[0, 0] + self.Z2[0, 0], self.Z1[0, 0] - self.Z2[0, 0]])
        t0_dot = sqrt(m/2)*np.array([self.Z1[1, 0] + self.Z2[1, 0], self.Z1[1, 0] - self.Z2[1, 0]])
        z1 = c/(2*m*w1)
        z2 = 1/sqrt(3)
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
                        + (-t0[0]*wd1*sin(wd1*self.time[i])
                        +(t0_dot[0]+z1*w1*t0[0])*cos(wd1*self.time[i]))
                        *exp(-z1*w1*self.time[i]))
            t2[i] = ((t0[1]*cos(wd2*self.time[i])
                    +((t0_dot[1]+z2*w2*t0[1])/wd2)
                    *sin(wd2*self.time[i]))*exp(-z2*w2*self.time[i]))
            t2Dot[i] = ((t0[1]*cos(wd2*self.time[i])
                         +((t0_dot[1]+z2*w2*t0[1])/wd2)
                         *sin(wd2*self.time[i]))*exp(-z2*w2*self.time[i])*(-z2*w2)
                        + (-t0[1]*wd2*sin(wd2*self.time[i])
                        +(t0_dot[1]+z2*w2*t0[1])*cos(wd2*self.time[i]))
                        *exp(-z2*w2*self.time[i]))
        self.x1Analytical = 1/sqrt(2*m)*(t1 + t2).reshape(-1)
        self.v1Analytical = 1/sqrt(2*m)*(t1Dot + t2Dot).reshape(-1)
        self.x2Analytical = 1/sqrt(2*m)*(t1 - t2).reshape(-1)
        self.v2Analytical = 1/sqrt(2*m)*(t1Dot - t2Dot).reshape(-1)
        self.Z1Analytical = np.array([self.x1Analytical, self.v1Analytical])
        self.Z2Analytical = np.array([self.x2Analytical, self.v2Analytical])
        self.couplingForce = (self.kc*(self.x2Analytical - self.x1Analytical)
                             +self.cc*(self.v2Analytical - self.v1Analytical))
        
        
    def calculateError(self, y1, y2):
        if self.oscMethod1 == 'Disp':
            self.localErrorY2[:, self.currentMacro] = np.linalg.norm(y2[:, 0] 
                                                      - self.Z2Analytical[:, self.currentMacro])
        else:
            self.localErrorY2[:, self.currentMacro] = np.linalg.norm(y2
                                                      - self.couplingForce[self.currentMacro])
        if self.oscMethod2 == 'Disp':
            self.localErrorY1[:, self.currentMacro] = np.linalg.norm(y1[:, 0] 
                                                      - self.Z1Analytical[:, self.currentMacro])
        else:
            self.localErrorY1[:, self.currentMacro] = np.linalg.norm(y1
                                                      + self.couplingForce[self.currentMacro])


    def plotOutputs(self):
        plt.figure(figsize=(14,8))
        plt.title('Απόκριση Διβάθμιου Ταλαντωτή μέσω Άμμεσης Συν-Προσομοίωσης' 
                  f' {self.cosiMethod}')
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
        plt.figure(figsize=(14,8))
        plt.title(f'Σχετικό σφάλμα άμεσης συν-προσομοίωσης με βήμα {self.stepDuration}' 
                  f' {self.cosiMethod}')
        plt.plot(self.time, self.localErrorY1[0, :], label='$Local Error 1$')
        plt.plot(self.time, self.localErrorY2[0, :], label='$Local Error 2$')
        plt.xlabel('time (sec)')
        plt.ylabel('RelativeError(t) (-)')
        plt.grid()
        plt.xlim(xmin=0, xmax=self.endTime)
        plt.legend()
        plt.show()   
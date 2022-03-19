"""
Created on Tue Nov  2 17:58:39 2021

@author: MrStevenn007
"""

from Orchestrator_2DoF_Richardson_1 import Orchestrator as OrchestratorRichardson
from Orchestrator_2DoF_local_error import Orchestrator
import numpy as np
import matplotlib.pyplot as plt

# Import Data of 2-DOF Linear Oscilator
m = 10 # (kg)
k = 10 # (N/m)
c = 10   # (Nsec/m)
cc = 5   # (Nsec/m)

# Import Initial Conditions for the Simulation
x10 = 0  # (m)
x20 = 0  # (m)
v10 = 1 # (m/sec)
v20 = 0  # (m/sec)
lc0 = k*(x20 - x10) + cc*(v20 - v10) # (N) Coupling Force at t = 0

# Initial Condition vectors
initial1 = np.array([[x10], [v10]])
initial2 = np.array([[x20], [v20]])

# Set Co-Simulation parameters
t0 = 0 # Initial Time of Simulation
tf = 12 # Final Time of Simulation
H = 1e-2 # Macro-Step of Co-Sim
polyDegree = 1 # Polynomial Interpolation degree
h = 1 # Micro-Step of Numerical Integration
Model1Method = 'Disp' # Oscilation Method of Model 1   
Model2Method = 'Disp' # Oscilation Method of Model 2
CoSimMethod = 'Gauss' # Co-Simulation Method for coupling variables

if Model1Method == 'Force':
    y2 = lc0
else:
    y2 = initial2

if Model2Method == 'Force':
    y1 = -lc0
else:
    y1 = initial1

# Initialize the Co-Simulation for the analytical local error
Co_Sim = Orchestrator(H, polyDegree, tf, k, cc, CoSimMethod)

# Create the 2 Subsystem models
Co_Sim.setModel1(m, k, c, Model1Method, 'RK45', h) # First Subsystem
Co_Sim.setModel2(m, k, c, Model2Method, 'RK45', h) # Second Subsystem

# Begin the Co-Simulation for the analytical local error
(localErrorX1, localErrorX2, localErrorV1,
 localErrorV2, localErrorY1, localErrorY2) = Co_Sim.beginSimulation(initial1, initial2, y1, y2)
 
print("Finished local error simulation.\n")

# Initialize the Co-Simulation for the Richardson Extrapolation Error Estimate
Co_SimRichardson = OrchestratorRichardson(H, polyDegree, tf, k, cc, CoSimMethod)

# Create the 2 Subsystem models
Co_SimRichardson.setModel1(m, k, c, Model1Method, 'RK45', h) # First Subsystem
Co_SimRichardson.setModel2(m, k, c, Model2Method, 'RK45', h) # Second Subsystem

# # Begin the Co-Simulation for the richardson extrapolation local error
(ERichY1, ERichY2) = Co_SimRichardson.beginSimulation(initial1, initial2, y1, y2)

print("Finished Richardson extrapolation simulation.\n")

# Plot Local Error of outputs
if polyDegree == 0:
    start_time = 0
else:
    start_time = 10

plt.figure(figsize=(14,8))
plt.title(f'Τοπικό Σφαλμα εξόδων λόγω συν-προσομοίωσης: {CoSimMethod}, {Model1Method} - {Model2Method}, k = {polyDegree}')
plt.plot(Co_Sim.time[start_time:], localErrorY1[0, start_time:], label='Local Error $y_1$')
plt.plot(Co_Sim.time[start_time:], localErrorY2[0, start_time:], label='Local Error $y_2$')
plt.plot(Co_Sim.time[start_time:], ERichY1[0, start_time:], '--', 
          label='Richardson Extrapolation Estimation $Y_1$')
plt.plot(Co_Sim.time[start_time:], ERichY2[0, start_time:], '--',
          label='Richardson Extrapolation Estimation $Y_2$')
plt.xlabel('time (sec)')
plt.ylabel('Local Error y')
plt.grid()
plt.xlim(xmin=0, xmax=tf)
plt.legend()
plt.show()   

print("FINISHED SIMULATION\n")

# # Plot Local Error of position
# plt.figure(figsize=(14,8))
# plt.title('Τοπικό Σφαλμα θέσεων λόγω συν-προσομοίωσης')
# plt.plot(Co_Sim.time, localErrorX1[:, 0], label='Local Error $x_1$')
# plt.plot(Co_Sim.time, localErrorX2[:, 0], label='Local Error $x_2$')
# plt.plot(Co_Sim.time, ERichX1[0, :], '--', 
#           label='Richardson Extrapolation Estimation $x_1$')
# plt.plot(Co_Sim.time, ERichX2[0, :], '--',
#           label='Richardson Extrapolation Estimation $x_2$')
# plt.xlabel('time (sec)')
# plt.ylabel('Local Error x (m)')
# plt.grid()
# plt.xlim(xmin=0, xmax=tf)
# plt.legend()
# plt.show() 

# # Plot Local Error of velocities
# plt.figure(figsize=(14,8))
# plt.title('Τοπικό Σφαλμα ταχυτήτων λόγω συν-προσομοίωσης')
# plt.plot(Co_Sim.time, localErrorV1[:, 0], label='Local Error $v_1$')
# plt.plot(Co_Sim.time, localErrorV2[:, 0], label='Local Error $v_2$')
# plt.plot(Co_Sim.time, ERichV1[0, :], '--', 
#           label='Richardson Extrapolation Estimation $v_1$')
# plt.plot(Co_Sim.time, ERichV2[0, :], '--',
#           label='Richardson Extrapolation Estimation $v_2$')
# plt.xlabel('time (sec)')
# plt.ylabel('Local Error v (m/sec)')
# plt.grid()
# plt.xlim(xmin=0, xmax=tf)
# plt.legend()
# plt.show()  
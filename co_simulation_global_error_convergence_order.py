"""
Created on Tue Nov  2 17:58:39 2021

@author: MrStevenn007
"""

from Orchestrator_2DoF_Richardson_1 import Orchestrator
import numpy as np
import matplotlib.pyplot as plt

# Import Data of 2-DOF Linear Oscilator
m = 10 # (kg)
k = 10 # (N/m)
c = 10 # (Nsec/m)
cc = 5 # (Nsec/m)
M = np.array([[m, 0], [0, m]])
K = np.array([[k+k, -k], [-k, k+k]])

# Import Initial Conditions for the Simulation
x10 = 0 # (m)
x20 = 0 # (m)
v10 = 1 # (m/sec)
v20 = 0 # (m/sec)
lc0 = k*(x20 - x10) + cc*(v20 - v10) # (N) Coupling Force at t = 0
initial1 = np.array([[x10], [v10]])
initial2 = np.array([[x20], [v20]])

# Set Co-Simulation parameters
t0 = 0 # Initial Time of Simulation
tf = 5 # Final Time of Simulation
polyDegree = 0 # Polynomial Interpolation degree
cosimMethod = 'Gauss'
MethodModel1 = 'Disp'
MethodModel2 = 'Disp'

if MethodModel1 == 'Force':
    y2 = lc0
else:
    y2 = initial2

if MethodModel2 == 'Force':
    y1 = -lc0
else:
    y1 = initial1

h = np.array([0.5, 1e-1, 0.5e-1, 1e-2, 0.5e-2, 1e-3])

errorX1 = np.zeros([len(h), 1])
errorV1 = np.zeros([len(h), 1])
errorX2 = np.zeros([len(h), 1])
errorV2 = np.zeros([len(h), 1])

errorX = np.zeros([len(h), 1])
errorV = np.zeros([len(h), 1])

for i in range(len(h)):
    Co_Sim = Orchestrator(h[i], polyDegree, tf, k, cc, cosimMethod)
    Co_Sim.setModel1(m, k, c, MethodModel1, 'RK45', 1) # First Subsystem
    Co_Sim.setModel2(m, k, c, MethodModel2, 'RK45', 1) # Second Subsystem
    absoluteError1, absoluteError2 = Co_Sim.beginSimulation(initial1, initial2, y1, y2)
    rmsErrorX1 = np.sqrt(np.mean(absoluteError1[0, :]**2))
    rmsErrorX2 = np.sqrt(np.mean(absoluteError2[0, :]**2))
    rmsErrorV1 = np.sqrt(np.mean(absoluteError1[1, :]**2))
    rmsErrorV2 = np.sqrt(np.mean(absoluteError2[1, :]**2))
    errorX1[i, 0] = rmsErrorX1 
    errorX2[i, 0] = rmsErrorX2
    errorV1[i, 0] = rmsErrorV1
    errorV2[i, 0] = rmsErrorV2
    # rmsX = np.array([[rmsErrorX1], [rmsErrorX2]]).reshape(2, 1)
    # rmsV = np.array([[rmsErrorV1], [rmsErrorV2]]).reshape(2, 1)
    # errorX[i, 0] = np.dot(rmsX.T, np.dot(np.eye(2), rmsX))
    # errorV[i, 0] = np.dot(rmsV.T, np.dot(np.eye(2), rmsV))
    print(f"Finished simulation of h = {h[i]}\n")
    
y1 = h**(polyDegree+1)
y2 = h**(polyDegree+2)

plt.figure(figsize=(14,8))
plt.title(f'Rms Global Error - H, {cosimMethod}, {MethodModel1} - {MethodModel2}, polydegree = {polyDegree}')
plt.plot(h, errorX1[:, 0], '*-', label='Τοπικό Σφάλμα Θέσεων - $x_1$')
plt.plot(h, errorV1[:, 0], '*-', label='Τοπικό Σφάλμα Ταχυτήτων - $v_1$')
plt.plot(h, errorX2[:, 0], '*-', label='Τοπικό Σφάλμα Θέσεων - $x_2$')
plt.plot(h, errorV2[:, 0], '*-', label='Τοπικό Σφάλμα Ταχυτήτων - $v_2$')
plt.plot(h, y1, '--', label='$Ο(H^{k+1})$')
plt.plot(h, y2, '--', label='$Ο(H^{k+2})$')
plt.xlabel('Macro Step Size - h (log)')
plt.ylabel('Global Error $e_{rms}$ (log)')
plt.grid()
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()

print("FINISHED SIMULATION\n")
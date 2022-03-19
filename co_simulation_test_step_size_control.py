"""
Created on Tue Nov  2 17:58:39 2021

@author: MrStevenn007
"""

from Orchestrator_2DoF_Richardson1_Step_Size_Control import Orchestrator
from Orchestrator_2DoF_Richardson_1 import Orchestrator as Orchestrator_constantStep
import numpy as np
import time

# Import Data of 2-DOF Linear Oscilator
m = 10 # (kg)
k = 10 # (N/m)
c = 10   # (Nsec/m)
cc = 5   # (Nsec/m)

# Import Initial Conditions for the Simulation
x10 = 0  # (m)
x20 = 0  # (m)
v10 = 10 # (m/sec)
v20 = 0  # (m/sec)
lc0 = k*(x20 - x10) + cc*(v20 - v10) # (N) Coupling Force at t = 0
initial1 = np.array([[x10], [v10]])
initial2 = np.array([[x20], [v20]])

# Set Co-Simulation parameters
t0 = 0 # Initial Time of Simulation
tf = 12 # Final Time of Simulation
H = 1e-2 # Macro-Step of Co-Sim
polyDegree = 2 # Polynomial Interpolation degree
h = 1 # Micro-Step of Numerical Integration

Model1Method = 'Disp'
Model2Method = 'Disp'
CoSimMethod = 'Jacobi'

if Model1Method == 'Force':
    y2 = lc0
else:
    y2 = initial2

if Model2Method == 'Force':
    y1 = -lc0
else:
    y1 = initial1

start_time = time.perf_counter()

# Initialize the Co-Simulation
Co_Sim = Orchestrator(H, polyDegree, tf, k, cc, CoSimMethod)

# Create the 2 Subsystem models
Co_Sim.setModel1(m, k, c, Model1Method, 'RK45', h) # First Subsystem
Co_Sim.setModel2(m, k, c, Model2Method, 'RK45', h) # Second Subsystem

# Begin the Co-Simulation
Co_Sim.beginSimulation(initial1, initial2, y1, y2)

rmsErrorX1 = np.sqrt(np.mean(Co_Sim.absoluteError1[0, :]**2))
rmsErrorX2 = np.sqrt(np.mean(Co_Sim.absoluteError2[0, :]**2))
print(f'\nRms error of position 1 is : {rmsErrorX1} with adaptive step')
print(f'\nRms error of position 2 is : {rmsErrorX2} with adaptive step')

end_time1 = time.perf_counter()
print(f'\nAdaptive Step Simulation finished in {end_time1 - start_time} second(s)')

# Initialize the Co-Simulation
Co_Sim_constantStep = Orchestrator_constantStep(H, polyDegree, tf, k, cc, CoSimMethod)

# Create the 2 Subsystem models
Co_Sim_constantStep.setModel1(m, k, c, Model1Method, 'RK45', h) # First Subsystem
Co_Sim_constantStep.setModel2(m, k, c, Model2Method, 'RK45', h) # Second Subsystem

# Begin the Co-Simulation
Co_Sim_constantStep.beginSimulation(initial1, initial2, y1, y2)

rmsErrorX1 = np.sqrt(np.mean(Co_Sim_constantStep.absoluteError1[0, :]**2))
rmsErrorX2 = np.sqrt(np.mean(Co_Sim_constantStep.absoluteError2[0, :]**2)) 
print(f'\nRms error of position 1 is : {rmsErrorX1} with fixed step')
print(f'\nRms error of position 2 is : {rmsErrorX2} with fixed step')

end_time2 = time.perf_counter()
print(f'\nFixed Step Simulation finished in {end_time2 - end_time1} seconds')
"""
Created on Tue Nov  2 17:58:39 2021

@author: MrStevenn007
"""

from Richardson import Orchestrator
import numpy as np
import time

start_time = time.perf_counter()

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
initial1 = np.array([[x10], [v10]])
initial2 = np.array([[x20], [v20]])

# Set Co-Simulation parameters
t0 = 0 # Initial Time of Simulation
tf = 12 # Final Time of Simulation
H = 1e-2 # Macro-Step of Co-Sim
polyDegree = 0 # Polynomial Interpolation degree
h = 1 # Micro-Step of Numerical Integration
Model1Method = 'Force' # Oscilation Method of Model 1   
Model2Method = 'Force' # Oscilation Method of Model 2
CoSimMethod = 'Gauss' # Co-Simulation Method for coupling variables

if Model1Method == 'Force':
    y2 = lc0
else:
    y2 = initial2

if Model2Method == 'Force':
    y1 = -lc0
else:
    y1 = initial1

# Initialize the Co-Simulation
Co_Sim = Orchestrator(H, polyDegree, tf, k, cc, CoSimMethod)

# Create the 2 Subsystem models
Co_Sim.setModel1(m, k, c, Model1Method, 'RK45', h) # First Subsystem
Co_Sim.setModel2(m, k, c, Model2Method, 'RK45', h) # Second Subsystem

# Begin the Co-Simulation
Co_Sim.beginSimulation(initial1, initial2, y1, y2)

end_time = time.perf_counter()
print(f"Process finished in : {end_time-start_time} second(s)")

Co_Sim.plotOutputs()
Co_Sim.plotLocalError()
Co_Sim.plotStepSize()
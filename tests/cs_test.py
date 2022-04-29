"""
Contains a test script that executes an explicit co-simulation of a linear 2dof oscilator and plots the position response, 
as well as the global and local errors.

@author: Stefanos Stathis
"""

import context
from sample.richardson import Orchestrator
import numpy as np
import time

start_time = time.perf_counter()

# Specify the data of the 2-DOF linear oscilator
m = 1    # (kg)
k = 100  # (N/m)

# Specify a Rayleigh damping. C = β * Κ where β = c / k = 1 / (10 * ω1)
c = 0.1 * (k * m)**0.5   # (Nsec/m)
cc = c   # (Nsec/m)

# Specify the initial conditions for the simulation
x10 = 0  # (m)
x20 = 1  # (m)
v10 = 1  # (m/sec)
v20 = 0  # (m/sec)

# Coupling Force at t = 0 (N)
lc0 = k*(x20 - x10) + cc*(v20 - v10)

initial1 = np.array([[x10], [v10]])
initial2 = np.array([[x20], [v20]])

# Final time
tf = 12

# Macro step
H = 1e-3

# Interpolation / Extrapolation degree
polyDegree = 1

micro_steps = 5

# Oscilation method of models
Model1Method = 'Disp'
Model2Method = 'Disp'

# Solver to use
solver_first = "Newmark"
solver_second = "RK45"

# Co-simulation comunication method to use
CoSimMethod = 'Gauss'

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
print("Succesfully initialized the orchestrator object")

# Create the 2 Subsystem models
Co_Sim.setModel1(m, k, c, Model1Method, solver_first, micro_steps)
Co_Sim.setModel2(m, k, c, Model2Method, solver_second, micro_steps)
print("Succesfully created the 2 slave models")

# Begin the Co-Simulation
print("Begining of simulation...")
Co_Sim.beginSimulation(initial1, initial2, y1, y2)

end_time = time.perf_counter()
print(f"Co-Simulation finished correctly in : {end_time-start_time} second(s)")

Co_Sim.plotPositions()
Co_Sim.plotVelocities()
Co_Sim.plotLocalError()
Co_Sim.plotGlobalError()
Co_Sim.plotStepSize()
"""
Contains a test script for ploting a compare between the local error due to co-simulation
and the error estimated through richardson extrapolation technique.

@author: Stefanos Stathis
"""

from Richardson import Orchestrator as master_richardson
from Local_Error import Orchestrator as master_local
import matplotlib.pyplot as plt
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

# Interpolation / Extrapolation degree
polyDegree = 0

# Macro step
H = 1e-2

micro_steps = 5

# Oscilation method of models
Model1Method = "Disp"
Model2Method = "Disp"

# Solver to use
solver_first = "Newmark"
solver_second = "RK45"

# Co-simulation comunication method to use
CoSimMethod = "Jacobi"

if Model1Method == "Force":
    y2 = lc0
else:
    y2 = initial2

if Model2Method == "Force":
    y1 = -lc0
else:
    y1 = initial1

# Initialize the Co-Simulation for the analytical local error
print("Begining of simulation for local error...")
Co_Sim = master_local(H, polyDegree, tf, k, cc, CoSimMethod)

# Create the 2 Subsystem models
Co_Sim.setModel1(m, k, c, Model1Method, solver_first, micro_steps) # First Subsystem
Co_Sim.setModel2(m, k, c, Model2Method, solver_second, micro_steps) # Second Subsystem

# Begin the Co-Simulation for the analytical local error
sim_start_local = time.perf_counter()
(localErrorX1, localErrorX2, localErrorV1,
localErrorV2, localErrorY1, localErrorY2) = Co_Sim.beginSimulation(initial1, initial2, y1, y2)
sim_end_local = time.perf_counter()
print(f"Succesfully finished local error simulation in {sim_end_local - sim_start_local} second(s)")

# Initialize the Co-Simulation for the Richardson Extrapolation Error Estimate
print("Begining of simulation for richardson estimation...")
Co_SimRichardson = master_richardson(H, polyDegree, tf, k, cc, CoSimMethod)

# Create the 2 Subsystem models
Co_SimRichardson.setModel1(m, k, c, Model1Method, solver_first, micro_steps) # First Subsystem
Co_SimRichardson.setModel2(m, k, c, Model2Method, solver_second, micro_steps) # Second Subsystem

# Begin the Co-Simulation for the richardson extrapolation local error
sim_start_richardson = time.perf_counter()
ERichY1, ERichY2 = Co_SimRichardson.beginSimulation(initial1, initial2, y1, y2)
sim_end_richardson = time.perf_counter()
print(f"Succesfully finished richardson estimation simulation in {sim_end_richardson - sim_start_richardson} second(s)")

end_time = time.perf_counter()
print(f"Succesfully finished simulations in {end_time - start_time} second(s)")

# Plot Local Error of outputs
plt.figure(figsize=(14,8))
plt.title(f'Τοπικό Σφαλμα εξόδων λόγω συν-προσομοίωσης: {CoSimMethod}, {Model1Method} - {Model2Method}, k = {polyDegree}')
plt.plot(Co_Sim.time[10:], localErrorY1[0, 10:], label='Local Error $y_1$')
plt.plot(Co_Sim.time[10:], localErrorY2[0, 10:], label='Local Error $y_2$')
plt.plot(Co_Sim.time[10:], ERichY1[0, 10:], '--', 
          label='Richardson Extrapolation Estimation $Y_1$')
plt.plot(Co_Sim.time[10:], ERichY2[0, 10:], '--',
          label='Richardson Extrapolation Estimation $Y_2$')
plt.xlabel('time (s)')
plt.ylabel('Local Error y')
plt.grid()
plt.xlim(xmin=0, xmax=tf)
plt.legend()
plt.show()

# Plot Local Error of position
plt.figure(figsize=(14,8))
plt.title('Τοπικό Σφαλμα θέσεων λόγω συν-προσομοίωσης')
plt.plot(Co_Sim.time[10:], localErrorX1[10:, 0], label='Local Error $x_1$')
plt.plot(Co_Sim.time[10:], localErrorX2[10:, 0], label='Local Error $x_2$')
plt.xlabel('time (s)')
plt.ylabel('Local Error x (m)')
plt.grid()
plt.xlim(xmin=0, xmax=tf)
plt.legend()
plt.show() 

# Plot Local Error of velocities
plt.figure(figsize=(14,8))
plt.title('Τοπικό Σφαλμα ταχυτήτων λόγω συν-προσομοίωσης')
plt.plot(Co_Sim.time[10:], localErrorV1[10:, 0], label='Local Error $v_1$')
plt.plot(Co_Sim.time[10:], localErrorV2[10:, 0], label='Local Error $v_2$')
plt.xlabel('time (s)')
plt.ylabel('Local Error v (m/s)')
plt.grid()
plt.xlim(xmin=0, xmax=tf)
plt.legend()
plt.show()  
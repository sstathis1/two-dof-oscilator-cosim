"""
Contains a test script for ploting the global and local error convergence orders
of an explicit co-simulation of a linear 2dof oscilator. 

@author: Stefanos Stathis
"""

from Richardson import Orchestrator
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
polyDegree = 1

micro_steps = 5

# Oscilation method of models
Model1Method = "Disp"
Model2Method = "Disp"

# Solver to use
solver_first = "Newmark"
solver_second = "RK45"

# Co-simulation comunication method to use
CoSimMethod = "Gauss"

if Model1Method == "Force":
    y2 = lc0
else:
    y2 = initial2

if Model2Method == "Force":
    y1 = -lc0
else:
    y1 = initial1

# List of macro steps
h = np.array([1e-2, 0.5e-2, 1e-3])

# Initialization of error arrays
errorX1 = np.zeros([len(h), 1])
errorV1 = np.zeros([len(h), 1])
errorX2 = np.zeros([len(h), 1])
errorV2 = np.zeros([len(h), 1])

# Loop through macro step list
for i in range(len(h)):
    Co_Sim = Orchestrator(h[i], polyDegree, tf, k, cc, CoSimMethod)
    Co_Sim.setModel1(m, k, c, Model1Method, solver_first, micro_steps) # First Subsystem
    Co_Sim.setModel2(m, k, c, Model2Method, solver_second, micro_steps) # Second Subsystem
    sim_start = time.perf_counter()
    print(f"Begining simulation with H = {h[i]}...")
    absoluteError1, absoluteError2 = Co_Sim.beginSimulation(initial1, initial2, y1, y2)
    sim_finish = time.perf_counter()
    print(f"Finished simulation of H = {h[i]} in {sim_finish - sim_start} second(s)")
    rmsErrorX1 = np.sqrt(np.mean(absoluteError1[0, 10::]**2))
    rmsErrorX2 = np.sqrt(np.mean(absoluteError2[0, 10::]**2))
    rmsErrorV1 = np.sqrt(np.mean(absoluteError1[1, 10::]**2))
    rmsErrorV2 = np.sqrt(np.mean(absoluteError2[1, 10::]**2))
    errorX1[i, 0] = rmsErrorX1 
    errorX2[i, 0] = rmsErrorX2
    errorV1[i, 0] = rmsErrorV1
    errorV2[i, 0] = rmsErrorV2
    print(f"Succesfully saved rms errors, moving to next simulation...")
   
y1 = h**(polyDegree + 1)
y2 = h**(polyDegree + 2)

end_time = time.perf_counter()
print(f"Co-Simulations finished correctly in : {end_time-start_time} second(s)")

# Plot error convergence orders
plt.figure(figsize=(14,8))
plt.title(f"Rms Global Error - H, {CoSimMethod}, {Model1Method} - {Model2Method}, polydegree = {polyDegree}")
plt.plot(h, errorX1[:, 0], '*-', label='Τοπικό Σφάλμα Θέσεων - $x_1$')
plt.plot(h, errorV1[:, 0], '*-', label='Τοπικό Σφάλμα Ταχυτήτων - $v_1$')
plt.plot(h, errorX2[:, 0], '*-', label='Τοπικό Σφάλμα Θέσεων - $x_2$')
plt.plot(h, errorV2[:, 0], '*-', label='Τοπικό Σφάλμα Ταχυτήτων - $v_2$')
plt.plot(h, y1, '--', label='$Ο(H^{k+1})$')
plt.plot(h, y2, '--', label='$Ο(H^{k+2})$')
plt.xlabel("Macro Step Size - h (log)")
plt.ylabel("Global Error $e_{rms}$ (log)")
plt.grid()
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()
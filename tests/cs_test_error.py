"""
Contains a test script for ploting the global and local error convergence orders
of an explicit co-simulation of a linear 2dof oscilator. 

@author: Stefanos Stathis
"""

import context
from sample.local import Orchestrator
import matplotlib.pyplot as plt
import numpy as np
import time

start_time = time.perf_counter()

# Specify mode to run -1 : global errors -2: local errors
mode = "local"

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
errorX1 = np.zeros([3*len(h), 1])
errorV1 = np.zeros([3*len(h), 1])
errorX2 = np.zeros([3*len(h), 1])
errorV2 = np.zeros([3*len(h), 1])

# Loop through macro step list
j = 0
for i in range(len(h)):
    for polyDegree in [0, 1, 2]:
        Co_Sim = Orchestrator(h[i], polyDegree, tf, k, cc, CoSimMethod)
        Co_Sim.setModel1(m, k, c, Model1Method, solver_first, micro_steps) # First Subsystem
        Co_Sim.setModel2(m, k, c, Model2Method, solver_second, micro_steps) # Second Subsystem
        print(f"Begining simulation with H = {h[i]}...")
        sim_start = time.perf_counter()
        if mode == "global":
            absoluteError1, absoluteError2 = Co_Sim.beginSimulation(initial1, initial2, y1, y2)
            sim_finish = time.perf_counter()
            print(f"Finished simulation of H = {h[i]} in {sim_finish - sim_start} second(s)")
            rmsErrorX1 = np.sqrt(np.mean(absoluteError1[0, 5::]**2))
            rmsErrorX2 = np.sqrt(np.mean(absoluteError2[0, 5::]**2))
            rmsErrorV1 = np.sqrt(np.mean(absoluteError1[1, 5::]**2))
            rmsErrorV2 = np.sqrt(np.mean(absoluteError2[1, 5::]**2))
        else:
            (local_error_X1, local_error_X2, local_error_V1, 
            local_error_V2, local_error_Y1, local_error_Y2) = Co_Sim.beginSimulation(initial1, initial2, y1, y2)
            sim_finish = time.perf_counter()
            print(f"Finished simulation of H = {h[i]} in {sim_finish - sim_start} second(s)")
            rmsErrorX1 = np.sqrt(np.mean(local_error_X1[5::]**2))
            rmsErrorX2 = np.sqrt(np.mean(local_error_X2[5::]**2))
            rmsErrorV1 = np.sqrt(np.mean(local_error_V1[5::]**2))
            rmsErrorV2 = np.sqrt(np.mean(local_error_V2[5::]**2))
        errorX1[j, 0] = rmsErrorX1 
        errorX2[j, 0] = rmsErrorX2
        errorV1[j, 0] = rmsErrorV1
        errorV2[j, 0] = rmsErrorV2
        j += 1
        print(f"Succesfully saved rms errors, moving to next simulation...")
        print()
   
y1 = h**(1)
y2 = h**(2)
y3 = h**(3)
y4 = h**(4)
y5 = h**(5)

end_time = time.perf_counter()
print(f"Co-Simulations finished correctly in : {end_time-start_time} second(s)")

# Plot global error convergence orders
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE, weight = 'bold')          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE, labelweight = 'bold')     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE, labelweight = 'bold')    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig, axs = plt.subplots(1, 2, dpi=150)
fig.suptitle(f"Τάξη τοπικών σφαλμάτων μέσω: {CoSimMethod}, {Model1Method} - {Model2Method}", fontweight='bold')

# Position subplots
axs[0].set_title("θέσεων $x_1$", fontweight='bold')
axs[0].plot(h, errorX2[0::3, 0], '*-', linewidth=3.0)
axs[0].plot(h, errorX2[1::3, 0], '*-', linewidth=3.0)
axs[0].plot(h, errorX2[2::3, 0], '*-', linewidth=3.0)
axs[0].plot(h, y3, '--', color="C4", linewidth=3.0)
axs[0].plot(h, y4, '--', color="C5", linewidth=3.0)
l7,= axs[0].plot(h, y5, '--', color="C6", linewidth=3.0)
axs[0].set(xlabel = "Βήμα επικοινωνίας H (log)", ylabel="Τοπικό σφάλμα $le^x_{rms}$ (log)")
axs[0].grid()
axs[0].set_xscale("log")
axs[0].set_yscale("log")

# Velocity subplots
axs[1].set_title(f"ταχυτήτων $v_1$", fontweight='bold')
l1,= axs[1].plot(h, errorV2[0::3, 0], '*-', linewidth=3.0)
l2,= axs[1].plot(h, errorV2[1::3, 0], '*-', linewidth=3.0)
l3,= axs[1].plot(h, errorV2[2::3, 0], '*-', linewidth=3.0)
l4,= axs[1].plot(h, y2, '--', linewidth=3.0)
l5,= axs[1].plot(h, y3, '--', linewidth=3.0)
l6,= axs[1].plot(h, y4, '--', linewidth=3.0)
axs[1].set(xlabel = "Βήμα επικοινωνίας H (log)", ylabel="Τοπικό σφάλμα $le^v_{rms}$ (log)")
axs[1].grid()
axs[1].set_yscale("log")
axs[1].set_xscale("log")
plt.legend([l1, l2, l3, l4, l5, l6, l7], ["k=0", "k=1", "k=2", "$O(H^2)$", "$O(H^3)$", "$O(H^4)$", "$O(H^5)$"], 
    bbox_to_anchor=(1,1), loc="upper left")
plt.show()
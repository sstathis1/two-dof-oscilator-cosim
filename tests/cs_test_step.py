"""
Contains a test script that compares the execution times and the local and global errors of a fixed step and 
an addaptive step explicit co-simulation.

@author: Stefanos Stathis
"""

import context
from sample.richardson_pi import Orchestrator as Orch_addaptive
from sample.richardson import Orchestrator as Orch_fixed
import matplotlib.pyplot as plt
import numpy as np
import time

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
polyDegree = 2

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

# Fixed Step simulation
print("Begining co-simulation with fixed step...")
start_time_fixed = time.perf_counter()

# Initialize the Co-Simulation
Co_Sim_constantStep = Orch_fixed(H, polyDegree, tf, k, cc, CoSimMethod)

# Create the 2 Subsystem models
Co_Sim_constantStep.setModel1(m, k, c, Model1Method, solver_first, micro_steps) # First Subsystem
Co_Sim_constantStep.setModel2(m, k, c, Model2Method, solver_second, micro_steps) # Second Subsystem

# Begin the Co-Simulation
Co_Sim_constantStep.beginSimulation(initial1, initial2, y1, y2)

end_time_fixed = time.perf_counter()
print(f'\nFixed Step Simulation finished in {end_time_fixed - start_time_fixed} seconds')

rmsErrorX1 = np.sqrt(np.mean(Co_Sim_constantStep.absoluteError1[0, 10::]**2))
rmsErrorX2 = np.sqrt(np.mean(Co_Sim_constantStep.absoluteError2[0, 10::]**2)) 
print(f'\nRms error of position 1 is : {rmsErrorX1} with fixed step')
print(f'\nRms error of position 2 is : {rmsErrorX2} with fixed step')
print(f"\nMaximum local error y = {np.max(Co_Sim_constantStep.ESTY1[0, 10::])}")

# Addaptive Step simulation
print("--------------------------------------------------------------------")
print("Begining co-simulation with adaptive step...")
start_time_addaptive = time.perf_counter()

# Initialize the Co-Simulation
Co_Sim = Orch_addaptive(H, polyDegree, tf, k, cc, CoSimMethod)

# Create the 2 Subsystem models
Co_Sim.setModel1(m, k, c, Model1Method, solver_first, micro_steps) # First Subsystem
Co_Sim.setModel2(m, k, c, Model2Method, solver_second, micro_steps) # Second Subsystem

# Begin the Co-Simulation
Co_Sim.beginSimulation(initial1, initial2, y1, y2)

end_time_addaptive = time.perf_counter()
print(f'\nAdaptive Step Simulation finished in {end_time_addaptive - start_time_addaptive} second(s)')

rmsErrorX1 = np.sqrt(np.mean(Co_Sim.absoluteError1[0, 10::]**2))
rmsErrorX2 = np.sqrt(np.mean(Co_Sim.absoluteError2[0, 10::]**2))
print(f'\nRms error of position 1 is : {rmsErrorX1} with adaptive step')
print(f'\nRms error of position 2 is : {rmsErrorX2} with adaptive step')

# Plot Local and Global Error
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE, weight = 'bold')          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE, labelweight = 'bold')     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE, labelweight = 'bold')    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig, axs = plt.subplots(1, 2, dpi=125)
fig.suptitle(f"Σύγκριση χρόνου επίλυσης - ολικών σφαλμάτων συν-προσομοίωσης με σταθερό και μεταβλητό βήμα, k = {polyDegree}", 
    fontweight='bold')

# Fixed Step plot
axs[0].set_title(f'Ολικό σφάλμα θέσεων με σταθερό βήμα', fontweight="bold")
axs[0].plot(Co_Sim_constantStep.time[10::], Co_Sim_constantStep.absoluteError1[0, 10::], 
    label='$e^{x_{1}}$', linewidth = 3.0)
axs[0].plot(Co_Sim_constantStep.time[10::], Co_Sim_constantStep.absoluteError2[0, 10::], 
    label='$e^{x_{2}}$', linewidth = 3.0)
axs[0].set(xlabel="time (s)", ylabel="$e^x$")
axs[0].grid()
axs[0].set_xlim([0, tf])
axs[0].legend()
axs[0].text(7, np.max(Co_Sim_constantStep.absoluteError1[0, 10::]) / 2, 
    f"t(fixed) = {(end_time_fixed - start_time_fixed):.2f} (s)", ha="center", va="center",  
    fontsize=14, fontweight='bold', bbox=dict(facecolor = 'orange', boxstyle="square"))
axs[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

# Adaptive Step plot
axs[1].set_title(f'Ολικό σφάλμα θέσεων με μεταβλητό βήμα', fontweight="bold")
axs[1].plot(Co_Sim.time[10::], Co_Sim.absoluteError1[0, 10::], label='$e^{x_{1}}$', linewidth = 3.0)
axs[1].plot(Co_Sim.time[10::], Co_Sim.absoluteError2[0, 10::], label='$e^{x_{2}}$', linewidth = 3.0)
axs[1].set(xlabel="time (s)", ylabel="$e^x$")
axs[1].grid()
axs[1].set_xlim([0, tf])
axs[1].legend()
axs[1].text(7, np.max(Co_Sim.absoluteError1[0, 10::]) / 2, 
    f"t(adaptive) = {(end_time_addaptive - start_time_addaptive):.2f} (s)", ha="center", va="center",  
    fontsize=14, fontweight='bold', bbox=dict(facecolor = 'orange', boxstyle="square"))
axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

plt.show()
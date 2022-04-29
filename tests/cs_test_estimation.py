"""
Contains a test script for ploting a compare between the local error due to co-simulation
and the error estimated through richardson extrapolation technique.

@author: Stefanos Stathis
"""

import context
from sample.richardson import Orchestrator as master_richardson
from sample.richardson_pi import Orchestrator as master_richardson_PI
from sample.local import Orchestrator as master_local
from sample.local_pi import Orchestrator as master_local_PI
import matplotlib.pyplot as plt
import numpy as np
import time

def main():

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
    polyDegree = 2

    # Macro step
    H = 1e-3

    micro_steps = 5

    # Oscilation method of models
    Model1Method = "Force"
    Model2Method = "Force"

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

    # Run fixed step simulations    
    fixed_step(m, k, c, H, polyDegree, tf, cc, CoSimMethod, Model1Method, Model2Method, 
    solver_first, solver_second, micro_steps, initial1, initial2, y1, y2)
    
    # Run addaptive step simulations
    print()
    addaptive_step(m, k, c, H, polyDegree, tf, cc, CoSimMethod, Model1Method, Model2Method, 
    solver_first, solver_second, micro_steps, initial1, initial2, y1, y2)

    # Print total time of simulation
    end_time = time.perf_counter()
    print()
    print(f"Total time of simulations: {end_time - start_time} second(s)")


def fixed_step(m, k, c, H, polyDegree, tf, cc, CoSimMethod, Model1Method, Model2Method, 
    solver_first, solver_second, micro_steps, initial1, initial2, y1, y2):
    """
    Does a fixed step simulation for the local error then 
    does a fixed step simulation for richardson estimation and compares them.
    """
    # Plot Local Error of outputs
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
    fig.suptitle(f"Τοπικό Σφαλμα εξόδων λόγω συν-προσομοίωσης: {CoSimMethod}, {Model1Method} - {Model2Method}", 
        fontweight='bold') 

    i = 0
    start_time = time.perf_counter()
    for polyDegree in [0, 1]:
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
        print(f"Succesfully finished fixed step simulations in {end_time - start_time} second(s)")
   
        axs[i].set_title(f"Εκτίμηση τοπικών σφαλμάτων για k = {i}", fontweight='bold')
        axs[i].plot(Co_Sim.time[10:], localErrorY1[0, 10:], label='$le^{y_1}$', linewidth=2.0)
        axs[i].plot(Co_Sim.time[10:], localErrorY2[0, 10:], label='$le^{y_2}$', linewidth=2.0)
        axs[i].plot(Co_Sim.time[10:], ERichY1[0, 10:], '--', 
                label='$EST_{Y_1}$', linewidth=2.0)
        axs[i].plot(Co_Sim.time[10:], ERichY2[0, 10:], '--',
                label='$EST_{Y_2}$', linewidth=2.0)
        axs[i].set(xlabel='time (s)', ylabel="$le^y$")
        axs[i].grid()
        axs[i].set_xlim([0, tf])

        i += 1
    plt.show()


def addaptive_step(m, k, c, H, polyDegree, tf, cc, CoSimMethod, Model1Method, Model2Method, 
    solver_first, solver_second, micro_steps, initial1, initial2, y1, y2):
    """
    Does an addaptive step simulation using richardson and PI controller then,
    does an addaptive step simulation for local error and compares them.
    """
    # Plot Local Error of outputs
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

    fig, axs = plt.subplots(2, 2, dpi=125)
    fig.suptitle(f"Τοπικό Σφαλμα εξόδων μεταβλητού βήματος: {CoSimMethod}, {Model1Method} - {Model2Method}", 
        fontweight='bold') 
    axs[0, 0].set_title("Εκτίμηση τοπικών σφαλμάτων", fontweight='bold')
    axs[0, 1].set_title("Απόκριση Μεταβλητού Βήματος άμεσης συν-προσομοίωσης", fontweight='bold')

    i = 0
    start_time = time.perf_counter()

    for polyDegree in [0, 1]:
        # Initialize the Co-Simulation for the analytical local error
        print("Begining of simulation, richardson adaptive step...")
        Co_SimRichardson = master_richardson_PI(H, polyDegree, tf, k, cc, CoSimMethod)

        # Create the 2 Subsystem models
        Co_SimRichardson.setModel1(m, k, c, Model1Method, solver_first, micro_steps) # First Subsystem
        Co_SimRichardson.setModel2(m, k, c, Model2Method, solver_second, micro_steps) # Second Subsystem

        # Begin the Co-Simulation for the analytical local error
        sim_start = time.perf_counter()
        ERichY1, ERichY2 = Co_SimRichardson.beginSimulation(initial1, initial2, y1, y2)
        sim_end = time.perf_counter()
        print(f"Succesfully finished richardson estimation simulation in {sim_end - sim_start} second(s)")
        t = Co_SimRichardson.time[0::2]

        # Initialize the Co-Simulation for the Richardson Extrapolation Error Estimate
        print("Begining of simulation, local error adaptive step...")
        Co_SimAnal = master_local_PI(t, polyDegree, tf, k, cc, CoSimMethod)

        # Create the 2 Subsystem models
        Co_SimAnal.setModel1(m, k, c, Model1Method, solver_first, micro_steps) # First Subsystem
        Co_SimAnal.setModel2(m, k, c, Model2Method, solver_second, micro_steps) # Second Subsystem

        # Begin the Co-Simulation for the richardson extrapolation local error
        sim_start = time.perf_counter()
        localErrorY1, localErrorY2 = Co_SimAnal.beginSimulation(initial1, initial2, y1, y2)
        sim_end = time.perf_counter()
        print(f"Succesfully finished local error simulation in {sim_end - sim_start} second(s)")

        end_time = time.perf_counter()
        print(f"Succesfully finished addaptive step simulations in {end_time - start_time} second(s)")

        # Plot Local Error of outputs
        axs[i, 0].plot(t[30::], localErrorY1[0, 30::], label='$le^{y_1}$', linewidth=2.0)
        axs[i, 0].plot(t[30::], localErrorY2[0, 30::], label='$le^{y_2}$', linewidth=2.0)
        axs[i, 0].plot(t[30::], ERichY1[30::], '--', label = '$EST_{Y_1}$', linewidth=2.0)
        axs[i, 0].plot(t[30::], ERichY2[30::], '--',  label = '$EST_{Y_2}$', linewidth=2.0)
        axs[i, 0].set(xlabel='time (s)', ylabel="$le^y$")
        axs[i, 0].grid()
        axs[i, 0].set_xlim([0, tf])
        axs[0, 0].legend(loc='lower right')

       # Plot Step Size
        axs[i, 1].text(1.5, np.max(Co_SimRichardson.Hn) / 2, f"k = {i}", ha="center", 
            va="center",  fontsize=16, fontweight='bold', 
            bbox=dict(facecolor = 'orange', boxstyle="square"))
        axs[i, 1].plot(t, Co_SimRichardson.Hn, linewidth=2.0)
        axs[i, 1].set(xlabel = 'time (s)', ylabel = '$Η_n$ (s)')
        axs[i, 1].grid()
        axs[i, 1].set_xlim([0, tf])
        i += 1
    plt.show()


if __name__ == "__main__":
    main()    
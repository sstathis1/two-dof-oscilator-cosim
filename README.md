# MyThesis-Project1
Explicit methods for the co-simulation of a linear two-degrees of freedom oscilator.

## Main scripts
Contain a master class for simulating 2 slave models.
* **Richardson.py** : Uses richardson extrapolation to estimate the local error.
* **Richardson_PI.py** : Uses richardson extrapolation with a PI controller for automatic step size selection.
* **Local_Error.py** : Uses the analytical solution for precise initial conditions in each step for the approximation of the local error due to co-simulation.
* **Local_Error_PI.py** : Uses the analytical solution for precise initial conditions in each step for the approximation of the local error due to co-simulation, works for addaptive step.

## Model scripts
Contain the solver for a model of a one-degree of freedom linear oscilator.
* **model_disp.py** : Solver for a displacement excitated mass with spring-damper.
* **model_force.py** : Solver for a force excitated mass with spring-damper.

## Test scripts
Contain some test scripts for initialization of the master objects and for ploting relevant data.
* **cs_test.py** : Used for doing a co-simulation with any of the main scripts. Plots displacements, velocities, local errors (opt: step size)
* **cs_test_error.py** : Used for doing a number of simulations to plot the convergence order for the local and global errors. (error - H)
* **cs_test_estimation.py** : Used for showcasing how richardson estimation compares to the local error due to co-simulation.
* **cs_test_step.py** : Used for showcasing how addaptive step time-to-run compare to fixed step time-to-run.

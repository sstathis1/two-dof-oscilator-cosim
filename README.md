# MyThesis-Project1
Explicit methods for the co-simulation of a linear two-degrees of freedom oscilator.

## Main scripts
Contain a master class for simulating 2 slave models.
* __Richardson.py__ : Uses richardson extrapolation to estimate the local error.
* __Richardson_PI.py__ : Uses richardson extrapolation with a PI controller for automatic step size selection.
* __Local_Error.py__ : Uses the analytical solution for precise initial conditions in each step for the approximation of the local error due to co-simulation.
* __Local_Error_PI.py__ : Uses the analytical solution for precise initial conditions in each step for the approximation of the local error due to co-simulation, works for addaptive step.

# MyThesis-Project1
Explicit methods for the co-simulation of a linear two-degrees of freedom oscilator.

## Main scripts
Contain a master class for simulating 2 slave models.
* Richardson.py : Uses richardson extrapolation to estimate the local error.
* Richardson_PI.py : Uses richardson extrapolation with a PI controller for automatic step size selection.
* Local_Error.py : Uses the analytical solution for precise initial conditions in each step for the approximation of the local error due to co-simulation.
* Local_Error_PI.py : Uses the analytical solution for precise initial conditions in each step for the approximation of the local error due to co-simulation, works for addaptive step.

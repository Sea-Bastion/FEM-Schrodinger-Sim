# Finite Element Method Schrodinger Simulation

This Project is based on the Galerkin method of solving differencial equations. A weak formulation of the Schrodinger equation is used with a set of basis functions to break the continuous differancial equation into a finite linear equation. This simulation uses a linear or "tent" basis function which essencially makes the space a linear interpolation between all sample points.

Unlike the Finite Distance Method I've used in my other simulations this method takes into accound the space between sample points rather then completely ignorming it by having the basis functions included in the differencial equation being solve, aka the weak formulation.


## Script Files

# GradientMtxGen / OverlapMtxGen / PotentialMtxGen

these three scripts all server the similar purpose of generating matricies based off the basis function which weight the connections between samples points in different ways. These scripts have been faily optimized with them running in under 30 seconds on my hardware. These matricies need to be generated based off of your space mesh and are saved into the resources folder as .npz files.


# GenEigenSates

This script is likely to take the longest time of the bunch. this goes thought and generates the eigen values of your system based off of what is essencially the weak formulation Hamiltonian Matrix. This takes a while to do often a few hours and it takes up a lot of space. if your choose you can limit the amount of eigen vectors generated which will lower the precision of the simulation while allowing you to increase the resolution with the ram saved.

# GenEvolution

This script uses the Eigen States generated using the last script as well as an initial condition and generates the evolution of the superposition over time. This uses the unity function with the eigen values found as the energies to evolve each eigen state in time and acheve a simulation of the time dependent schrodinger equation.

# RenderEvolution / ViewEvolution

These scripts are used to view or render respectively the evolution data generated from the last script. Often times the data can be slow to render and using the view script will generally give you a slow framerate.

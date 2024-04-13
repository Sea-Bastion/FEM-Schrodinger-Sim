#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 23:48:00 2023

    This script requires some matricies to be dense and so limits the detail of the system

@author: sebas
"""

from scipy import sparse
import scipy.sparse.linalg
from scipy.sparse import load_npz
import numpy as np
import scipy as sp
import trimesh
import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt

Mesh = trimesh.load_mesh("resources/square.2.stl", "stl")


PotentialMatrix = load_npz("resources/PotentialMatrix.npz")
GradientMatrix = load_npz("resources/GradientMatrix.npz")
OverlapMatrix = load_npz("resources/OverlapMatrix.npz")


# move from CPU to GPU
# PotentialMatrix = sparse.csr_matrix(PotentialMatrix)
# GradientMatrix = sparse.csr_matrix(GradientMatrix)
# OverlapMatrix = sparse.csr_matrix(OverlapMatrix)


# InitFunc = lambda x,y: np.exp(- 8*(x**2+y**2) )


# Initial = np.array([ InitFunc(i[0], i[1]) for i in Mesh.vertices ])


h_bar = 1
ElectronMass = 1



EigCount = 4000
alpha = -h_bar**2 / (2* ElectronMass)

HamiltonianMatrix = alpha * GradientMatrix + PotentialMatrix
# OverlapFactored = sparse.linalg.factorized(OverlapMatrix)

# def OperatorFunction(x):
#     return OverlapFactored(HamiltonianMatrix @ x)


# OperatorMatrix = sparse.linalg.LinearOperator(OverlapMatrix.shape, matvec=OperatorFunction)


print("computing EigenValues")

# EigVal, EigVec = sp.linalg.eigh( RHS , b=LHS, overwrite_a=True, overwrite_b=True, driver="gv" )
EigVal, EigVec = sparse.linalg.eigsh(HamiltonianMatrix, M=OverlapMatrix, k=EigCount, which='SA')
# EigVal, EigVec = sparse.linalg.eigsh(OperatorMatrix, k=EigCount, which='SA')
# del RHS, LHS
print("Saving EigenValues")





ExportVars = {
    "EigVal": EigVal,
    "EigVec": EigVec,
    "Mesh": Mesh
}


with open("resources/EigenStateData.pickle", 'wb') as outfile:
    pickle.dump(ExportVars, outfile)


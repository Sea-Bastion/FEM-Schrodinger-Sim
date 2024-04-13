#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 21:14:17 2023

@author: sebas
"""

import numpy as np
import scipy as sp
import pickle


DeltaTime = 1/10000
FrameCount = 2000
EigenDataPath = "resources/EigenStateData.pickle"
EvoOutPath = "resources/EvolutionData.pickle"

with open(EigenDataPath, 'rb') as infile:
    EigenData = pickle.load(infile)
    

EigVal = EigenData["EigVal"]
EigVec = EigenData["EigVec"]
Mesh = EigenData["Mesh"]



def InitFunc(x,y):
    dx = 0.2
    k = np.array([0, 60])
    r = np.array([x,y]) 
    r -= np.array([0, -0.5])
    
    a = 2* dx**2
    
    return 1/(a * np.sqrt(np.pi) ) * np.exp(-np.dot(r,r)/(a**2) + 1j * np.dot(r,k) )


print("Approximating for Init superposition")
Initial = np.array([ InitFunc(i[0], i[1]) for i in Mesh.vertices ])
InitState, Resid, _, _ = sp.linalg.lstsq( EigVec, Initial, cond=None, overwrite_b=True)
print("Residual 2-norm: {Err}".format(Err=Resid))
# InitState = sp.linalg.solve( EigVec, Initial, overwrite_b=True)

print("Setting up Time Itterated states")
Times = np.linspace(0, DeltaTime*(FrameCount-1), FrameCount)
TimeOp = np.exp(-1j*np.outer(EigVal, Times))

States = np.vstack(FrameCount*[InitState]).T * TimeOp

print("Solving for Evolved States")
EvolutionData = EigVec @ States
del EigVec
EvolutionData = EvolutionData.T

ExportVars = { 
    "EvolutionData": EvolutionData,
    "Mesh":          Mesh
}

with open(EvoOutPath, 'wb') as outfile:
    pickle.dump(ExportVars, outfile)




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 08:33:20 2024

@author: sebas
"""


import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import animation
import pickle



EvolutionPath = "resources/EvolutionData.pickle"

with open(EvolutionPath, 'rb') as infile:
    EvolutionData = pickle.load(infile)
    

States = EvolutionData["EvolutionData"]
States = np.abs(States)
Mesh = EvolutionData["Mesh"]
FrameCount = States.shape[0]


# make matplotlib triangulation object for plots
verts = Mesh.vertices
Triangles = mpl.tri.Triangulation(verts[:,0], verts[:,1], Mesh.faces)


# setup plot for simulation and plot network over output
fig = plt.figure(frameon=False, figsize=(6,6))
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
ax.set_xlim((-1,1))
ax.set_ylim((-1,1))

ScaleMax = States.max()
ScaleMin = States.min()
StatePlot = ax.tripcolor(Triangles, States[0], cmap='magma', vmax=ScaleMax, vmin=ScaleMin, shading="gouraud")


# ax.triplot(Triangles, zorder=5, color='black', lw=0.1)



def Animate(frame):
    global StatePlot
    
    
    # StatePlot = ax.tripcolor(Triangles, States[frame], vmax=ScaleMax, vmin=ScaleMin, cmap='Spectral')
    StatePlot.set_array(States[frame])
    print("rendering frame: {frameNum} out of {total}".format(frameNum=frame, total=FrameCount))

    
    return StatePlot

anim = animation.FuncAnimation(fig, Animate, frames=FrameCount, repeat=True, interval=20)

plt.show()
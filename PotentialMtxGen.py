#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:14:36 2024

@author: sebas
"""

import numpy as np
from scipy import sparse 
import scipy as sp
import trimesh 
from multiprocessing import Pool
from tqdm import tqdm
from matplotlib import pyplot as plt
import warnings

warnings.simplefilter('ignore', sp.integrate.IntegrationWarning)


# Load in mesh and get Mesh properties needed
Mesh = trimesh.load_mesh("resources/square.2.stl", "stl")
VertCount = Mesh.vertices.shape[0]
FaceAreas = Mesh.area_faces
Faces = Mesh.faces
VertFaces = Mesh.vertex_faces
Verts = Mesh.vertices


# Generate list of all connections including self 
Neighbors = np.array(Mesh.edges_unique)
SelfConnect = np.array( 2*[range( VertCount )] ).T
Neighbors = np.append(Neighbors, SelfConnect, axis=0)
del SelfConnect
ConnectionCount = Neighbors.shape[0]


def Potential(r):
    if (-0.9 < r[0] < 0.9 and -0.9 < r[1] < 0.9) and ( np.abs(r[0]) > 0.04 ):
        return 0
    else:
        return 1
        




# #display potential map
# DispSize = 200
# PotSample = [[ Potential([x,y]) for x in np.linspace(-1, 1, DispSize) ] for y in np.linspace(-1, 1, DispSize) ]

# plt.imshow(PotSample)
# plt.show()

def ProcessID(ID):
    AID = Neighbors[ID,0]
    BID = Neighbors[ID,1]
    
        
    CommonFaces = np.intersect1d(VertFaces[AID,:], VertFaces[BID,:])
    CommonFaces = CommonFaces[CommonFaces >= 0]
    
    FinalValue = 0
    
    
    for FaceID in CommonFaces:
          
        VertID = Faces[FaceID]
        XY_Matrix = np.array(Verts[VertID])
        
        
        #--------------------------------Transform Space for Integration------------------- 
        
        #get edge lengths
        EdgeMtx = np.array([[0,1], [1,2], [2,0]]) #list of Vertex edges based off XY_Matrix
        EdgeLengths = np.array([np.linalg.norm(r[0] - r[1]) for r in XY_Matrix[EdgeMtx]]) #Length of edges
        
        #gets the coords of the verts that belong to the longest edge
        LongEdgeVerts = XY_Matrix[EdgeMtx[np.argmax(EdgeLengths)]] #edge ID longest > vert ID > vert coords
        
        # this takes the verts of the longest edge and sorts by the x value
        SortedCoordVerts = LongEdgeVerts[np.argsort(LongEdgeVerts[:,0])]
        
        # set smaller x value point to origin
        Origin = np.copy(SortedCoordVerts[0,:])
        XY_Matrix -= Origin
        SortedCoordVerts -= Origin
        
        # find angle of longest line and rotate space to set longest line to x axis
        Rotation = -np.arctan2(SortedCoordVerts[1,1], SortedCoordVerts[1,0])
        RotMtx = sp.spatial.transform.Rotation.from_rotvec([0, 0, Rotation]).as_matrix()
        RevRotMtx = sp.spatial.transform.Rotation.from_rotvec([0, 0, -Rotation]).as_matrix()
        
        #apply Rotation Matrix and flip triangle up if pointing down
        XY_Matrix = np.abs( (RotMtx @ XY_Matrix.T).T )
        
        #space should now be translated so that the longest edge is on the x axis
        #and all points are in the first quadrent (all positive coords)
        
        
        # -------------------------------------Integration---------------------------
        
        # linear regression 
        XY_Matrix[:,-1] = [1,1,1]
        CoeffA = np.linalg.solve(XY_Matrix, VertID == AID)
        CoeffB = np.linalg.solve(XY_Matrix, VertID == BID)
        
        # y axis boundry equations
        HeightPoint = XY_Matrix[np.argmax(XY_Matrix[:,1])][:2]
        EndPoint = XY_Matrix[np.argmax(XY_Matrix[:,0])][:2]
        Slopes = np.array([ HeightPoint[1]/HeightPoint[0], 
                           (HeightPoint[1] - EndPoint[1])/(HeightPoint[0] - EndPoint[0]) ])
        Intercepts = np.array([0, EndPoint[1] - Slopes[1] * EndPoint[0]])
        
        Bound1 = lambda x: Slopes[0]*x + Intercepts[0]
        Bound2 = lambda x: Slopes[1]*x + Intercepts[1]
        
        # x axis bounds
        x1 = HeightPoint[0]
        x2 = EndPoint[0]
        
        #------------Use Scipy quad function-------------
        
        #potential uses a global xy while the interpolations use a local xy
        Integrand = lambda y,x:  CoeffA.dot([x,y,1]) * CoeffB.dot([x,y,1]) * Potential( (RevRotMtx @ [x, y, 1]) + Origin )
        
        FinalValue += sp.integrate.dblquad(Integrand,  0, x1, 0, Bound1)[0]
        FinalValue += sp.integrate.dblquad(Integrand, x1, x2, 0, Bound2)[0]
        
        
    
    
    
    
    return FinalValue, ID




OutputValues = np.zeros(ConnectionCount)

with Pool(10) as p:
    for v,i in tqdm(p.imap_unordered(ProcessID, range(ConnectionCount)), total=ConnectionCount):
        OutputValues[i] = v
        
OutputValues *= 1e9 #---------------------------------Multiply Values for particle in box

print("\nMaking Sparse Matrix")
OutputMatrix = sparse.csr_matrix( ( OutputValues, ( Neighbors[:,0], Neighbors[:,1] ) ), shape=(VertCount, VertCount) )


OutputMatrix += OutputMatrix.T - sparse.diags(OutputMatrix.diagonal())

sparse.save_npz("resources/PotentialMatrix.npz", OutputMatrix)


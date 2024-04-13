#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 04:36:32 2023

@author: sebas
"""

import numpy as np
from scipy import sparse 
import scipy as sp
import trimesh 
from multiprocessing import Pool
from tqdm import tqdm


# Load in mesh and get Mesh properties needed
Mesh = trimesh.load_mesh("resources/square.2.stl", "stl")
VertCount = Mesh.vertices.shape[0]
Faces = Mesh.faces
Verts = Mesh.vertices
VertFaces = Mesh.vertex_faces
#getting these vars are actually functions so it should be done ahead of time
#so they don't have to be regenerated every loop


# Generate list of all connections including self 
Neighbors = np.array(Mesh.edges_unique)
SelfConnect = np.array( 2*[range( VertCount )] ).T
Neighbors = np.append(Neighbors, SelfConnect, axis=0)
del SelfConnect
ConnectionCount = Neighbors.shape[0]


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
        XY_Matrix -= SortedCoordVerts[0,:]
        SortedCoordVerts -= SortedCoordVerts[0,:]
        
        # find angle of longest line and rotate space to set longest line to x axis
        Rotation = -np.arctan2(SortedCoordVerts[1,1], SortedCoordVerts[1,0])
        RotMtx = sp.spatial.transform.Rotation.from_rotvec([0, 0, Rotation]).as_matrix()
        
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
        
        # x axis bounds
        x1 = HeightPoint[0]
        x2 = EndPoint[0]
        
        #------------Use Precaluated Integral equation-------------
        Z = np.zeros(2)
        BoundsMtx = np.array([
            [Slopes,        Slopes**3 /3,               Z,          Slopes**2 /2,           Z,                      Z],
            [Intercepts,    Intercepts * Slopes**2,     Z,          Intercepts * Slopes,    Slopes**2 /2,           Slopes],
            [Z,             Intercepts**2 * Slopes,     Slopes,     Intercepts**2 /2,       Intercepts * Slopes,    Intercepts],
            [Z,             Intercepts**3 /3,           Intercepts, Z,                      Intercepts**2 /2,       Z]
        ]).transpose((2,0,1))
        
        
        RegCoeffVec = np.hstack(( CoeffA * CoeffB, 
                                [ np.dot( CoeffA[[0,1]], np.flip(CoeffB[[0,1]]) ),
                                  np.dot( CoeffA[[1,2]], np.flip(CoeffB[[1,2]]) ),
                                  np.dot( CoeffA[[0,2]], np.flip(CoeffB[[0,2]]) )]) )
        
        
        XVector = np.array([
            [x1**i / i for i in range(4,0,-1)],
            [x2**i / i for i in range(4,0,-1)],
        ])
        XVector[1] -= XVector[0]
        
        SectionInteg = np.sum([ np.dot(BoundsMtx[i] @ RegCoeffVec, XVector[i]) for i in [0,1] ])
        FinalValue += SectionInteg
        
        # you just have to trust me I spent a long time expanding the plane equations, 
        # Integrating them and then simplifying the end equation
        
        # the triangle is lined up and then split into 2 right triangles which are integrated over
        # with a line as the upper y limit
    
    
    return FinalValue, ID


OutputValues = np.zeros(ConnectionCount)

with Pool(10) as p:
    for v,i in tqdm(p.imap_unordered(ProcessID, range(ConnectionCount)), total=ConnectionCount):
        OutputValues[i] = v

print("Making Sparse Matrix")
OutputMatrix = sparse.csr_matrix( ( OutputValues, ( Neighbors[:,0], Neighbors[:,1] ) ), shape=(VertCount, VertCount) )


OutputMatrix += OutputMatrix.T - sparse.diags(OutputMatrix.diagonal())

sparse.save_npz("resources/OverlapMatrix.npz", OutputMatrix)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 19:58:43 2023

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


def ProcessID(ID):
    AID = Neighbors[ID,0]
    BID = Neighbors[ID,1]
    
    CommonFaces = np.intersect1d(VertFaces[AID,:], VertFaces[BID,:])
    CommonFaces = CommonFaces[CommonFaces >= 0]
    
    FinalValue = 0
    
    for FaceID in CommonFaces:
        VertID = Faces[FaceID]
        
        # set up matrix os x and y values and replace z columb with a 1 columb
        # to allow for non 0 z intercepts
        XY_Matrix = np.array(Verts[VertID])
        XY_Matrix[:,-1] = [1,1,1]
        
        # do a linear regression on the face to find slope vector
        CoeffA = np.linalg.solve(XY_Matrix, VertID == AID)
        CoeffB = np.linalg.solve(XY_Matrix, VertID == BID)
        
        #calulate dot product of gradients
        GradDot = np.dot(CoeffA[:2], CoeffB[:2])
        FinalValue += -GradDot * FaceAreas[FaceID]
    
    
    
    
    return FinalValue, ID




OutputValues = np.zeros(ConnectionCount)

with Pool(10) as p:
    for v,i in tqdm(p.imap_unordered(ProcessID, range(ConnectionCount)), total=ConnectionCount):
        OutputValues[i] = v

print("\nMaking Sparse Matrix")
OutputMatrix = sparse.csr_matrix( ( OutputValues, ( Neighbors[:,0], Neighbors[:,1] ) ), shape=(VertCount, VertCount) )


OutputMatrix += OutputMatrix.T - sparse.diags(OutputMatrix.diagonal())

sparse.save_npz("resources/GradientMatrix.npz", OutputMatrix)

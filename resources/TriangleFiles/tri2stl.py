
import numpy as np
import re
import argparse
import trimesh

parser = argparse.ArgumentParser(description="turn .ele and .node file pairs into .stl files")
parser.add_argument('prefix', metavar='P', type=str, help='prefix for files to be parsed')

args = parser.parse_args()

with open(args.prefix + ".node", 'r') as infile:
	NodeString = infile.read()

with open(args.prefix + ".ele", 'r') as infile:
	TriString = infile.read()

pattern = re.compile("([-\d\.e]+)\s+([-\d\.e]+)\s+([-\d\.e]+)\s+([-\d\.e]+)\n")


Verts = np.array( pattern.findall(NodeString), dtype=float )
Verts = np.delete(Verts, 0, 0)
Verts = np.delete(Verts, 0, 1)
Verts[:,-1] = np.zeros(Verts.shape[0])


Triangles = np.array( pattern.findall(TriString) , dtype=float )
Triangles = np.delete(Triangles, 0, 1)
Triangles -= 1

Mesh = trimesh.Trimesh(Verts, Triangles)

STLData = trimesh.exchange.stl.export_stl(Mesh)

with open(args.prefix + ".stl", 'wb') as outfile:
 	outfile.write(STLData)

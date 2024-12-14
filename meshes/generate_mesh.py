import sys, os
import dolfin as df
import numpy as np
import chaospy as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
from os import path
from dolfin import *
# from kl import *
# from tools import *
import argparse


pt0 = Point(0.0,0.0)
pt1 = Point(3.67,1.0)
mesh = RectangleMesh.create([pt0,pt1], [200,100], CellType.Type.quadrilateral)
V_CG1 = FunctionSpace(mesh, "CG", 1)
V_DG0 = FunctionSpace(mesh, "DG", 0)
V_VecDG0 = VectorFunctionSpace(mesh, "DG", 0)

pp = Function(V_CG1)  # perm at nodes
pe = Function(V_DG0)  # perm at cells

#p_array_points = p_array_points.reshape((npoints,2))
V = FunctionSpace(mesh, 'Lagrange', 1)
p = V.tabulate_dof_coordinates().reshape((-1, 2)).transpose()
np.savetxt('my_points_cluster.txt', p)

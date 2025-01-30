import sys, os
import dolfin as df
import numpy as np
import chaospy as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
from os import path
import time
from dolfin import *
import argparse
import scipy.sparse.linalg as spla
from scipy import sparse
from scipy.spatial import distance

def print_matrix_sample(label, matrix, sample_size=5):
    print(f"\n{label} (amostra {sample_size}x{sample_size}):")
    sample = matrix[:sample_size, :sample_size] if isinstance(matrix, np.ndarray) else matrix.toarray()[:sample_size, :sample_size]
    print(sample)


def createmeshs(n_kl, l_kl, sigma_kl):
    
    pt0 = Point(0.0,0.0)
    pt1 = Point(3.67,1.0)

    mesh = RectangleMesh.create([pt0,pt1], [55,15], CellType.Type.quadrilateral)    

    hdf = HDF5File(mesh.mpi_comm(), "output.h5", "w")
    hdf.write(mesh, "/mesh")

    elem_inds = [cell.index() for cell in cells(mesh)]
    facet_inds = [facet.index() for facet in facets(mesh)]
    point_inds = [vert.index() for vert in vertices(mesh)]
    nelems = len(elem_inds)
    npoints = len(point_inds)
    print('Number of elements:', nelems)
    print('Number of nodes:', npoints)

    # dados de perm
    d = np.ones((15,55))
    # d = np.loadtxt(perm_arq)
    d2m2 = 9.869233e-13
    # convert unit and transform to log10
    d = np.log10(d2m2 * d)

    # permeability field
    perm = d.flatten()
    perm_at_points = np.zeros(npoints)

    V_CG1 = FunctionSpace(mesh, "CG", 1)
    V_DG0 = FunctionSpace(mesh, "DG", 0)
    V_VecDG0 = VectorFunctionSpace(mesh, "DG", 0)

    pp = Function(V_CG1)  # perm at nodes
    pe = Function(V_DG0)  # perm at cells

    pe.vector().set_local(perm)    
    pp = project(pe, V_CG1)
    perm_at_points = pp.vector().get_local()

    # print('Permeability field:', np.shape(d))
    # print('Permeabilities:', perm_at_points)

    #output files and options 
    fileO = XDMFFile(mesh.mpi_comm(), f"output_new.xdmf")
    fileO.parameters['rewrite_function_mesh']= False
    fileO.parameters["functions_share_mesh"] = True
    fileO.parameters["flush_output"] = True
    fileO.write(pp,0)

    # ---------------------------------------------------------------------------------------------
    # Karhunen-Loeve Expansion
    # ---------------------------------------------------------------------------------------------

    V = FunctionSpace(mesh, 'Lagrange', 1)
    p = V.tabulate_dof_coordinates()
    np.savetxt('my_points.txt', p.reshape((-1, 2)).transpose())
    print(np.shape(p))
    p = p.transpose()    
    print(np.shape(p))

    # Laplace term
    u = TrialFunction(V)
    v = TestFunction(V)
    a_M = u*v*dx

    # Compute matrices
    M = assemble(a_M)

    row, col, data = as_backend_type(M).mat().getValuesCSR()
    M_csr = sparse.csr_matrix((data,col,row))

    # Converter os deslocamentos de linha (row) em índices de linha correspondentes
    row_indices = np.repeat(np.arange(len(row) - 1), np.diff(row))
    assert len(row_indices) == len(col) == len(data), "Dimensões inconsistentes nos dados da matriz"

    # Salvar a matriz no formato triplo (linha, coluna, valor)
    np.savetxt(
        "matrix_M.txt",
        np.column_stack((row_indices, col, data)),
        fmt="%d %d %.18e"
    )

    print("Computing Covariance matrix")
    start_time = time.time()

    npts = np.shape(p)[1]
    num_elements = int((npts * (npts + 1)) / 2 - npts)  # Tamanho do vetor para a parte triangular inferior
    Qvec = np.zeros(num_elements)
    Q = np.zeros((npts, npts))

    sigmakl2 = sigma_kl ** 2
    lkl2 = l_kl ** 2
    
    # Calcular valores da diagonal principal = correlação de um ponto com ele mesmo
    Q_diag = sigmakl2

    # Montagem da lista auxiliar com os calculos do triangulo inferior
    coordp1 = np.zeros(num_elements)
    coordp2 = np.zeros(num_elements)
    countcoord = 0
    for p1 in range(1, npts):
        for p2 in range(p1):
            coordp1[countcoord] = p1
            coordp2[countcoord] = p2
            countcoord += 1
    assert countcoord == num_elements

    for i in range(num_elements):
        p_i = [p[0, int(coordp1[i])], p[1, int(coordp1[i])]]
        p_j = [p[0, int(coordp2[i])], p[1, int(coordp2[i])]]
        Qvec[i] = sigmakl2 * exp(-distance.euclidean(p_i, p_j) / (2 * lkl2))

    # Abrir o Qvec na forma matricial
    for i in range(npts):
        Q[i, i] = Q_diag

    countcoord = 0
    for i in range(1, npts):
        for j in range(i):
            Q[i, j] = Qvec[countcoord]
            Q[j, i] = Q[i, j]
            countcoord += 1

    Q = sparse.csr_matrix(Q)

    np.savetxt("matrix_Q_py.txt", Q.toarray(), fmt="%.18e")
    print_matrix_sample("Matriz Q:", Q)

    # Solve eigenvalue problem

    print("Solving eigenvalue problem")
    print(f"Q type: {type(Q)}, shape: {getattr(Q, 'shape', 'undefined')}")
    print(f"M_csr type: {type(M_csr)}, shape: {getattr(M_csr, 'shape', 'undefined')}")

    Q = Q.toarray() if not isinstance(Q, np.ndarray) else Q
    M_csr = M_csr.toarray() if not isinstance(M_csr, np.ndarray) else M_csr

    plt.clf()
    plt.imshow(M_csr, cmap='jet', origin='upper')
    plt.colorbar()
    plt.savefig('mass_matrix.png', dpi=300)
    plt.close()

    auxm = np.dot(Q, M_csr)
    print(f"auxm type: {type(auxm)}, shape: {getattr(auxm, 'shape', 'undefined')}")

    Tmat = np.dot(np.transpose(M_csr), auxm)
    print(f"Tmat type: {type(Tmat)}, shape: {getattr(Tmat, 'shape', 'undefined')}")

    np.savetxt("matrix_T_py.txt", Tmat, fmt="%.18e")
    print_matrix_sample("Matriz T:", Tmat)

    lamb, phi = spla.eigs(Tmat, M=M_csr, k=n_kl, which="LR")
    lamb, phi = lamb.real, phi.real

    print(np.shape(lamb), np.shape(phi))

    np.savetxt('eigenvalues_py.txt', lamb)
    np.savetxt('eigenfunctions_py.txt', phi)

    runtime = time.time() - start_time
    print(f'Ellapsed time: {runtime}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n',     default=8,   help='number of terms in the KL expansion (nkl)')
    parser.add_argument('-l',     default=0.5, help='characteristic length (lkl)')
    parser.add_argument('-sigma', default=0.5, help='sigma KL')
    args = parser.parse_args()

    if not os.path.exists('figs'):
        os.mkdir('figs')
        print(f"Directory '{'figs'}' created.")
    else:
        print(f"Directory '{'figs'}' already exists.")

    if not os.path.exists('outputs'):
        os.mkdir('outputs')
        print(f"Directory '{'outputs'}' created.")
    else:
        print(f"Directory '{'outputs'}' already exists.")

    n_kl = int(args.n)
    l_kl = float(args.l)
    sigma_kl = float(args.sigma)

    createmeshs(n_kl, l_kl, sigma_kl)

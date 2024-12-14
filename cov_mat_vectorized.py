import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
import time
from math import exp
import sys


def CovarianceMatrix(p, sigma_kl, l_kl):

    start_time = time.time()
    npts = np.shape(p)[1]
    num_elements = int((npts * (npts + 1)) / 2 - npts)  # Tamanho do vetor para a parte triangular inferior

    # Q = np.zeros((npts, npts))
    Qvec = np.zeros(num_elements)

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

    # # Abrir o Qvec na forma matricial
    # for i in range(npts):
    #     Q[i, i] = Q_diag

    # countcoord = 0
    # for i in range(1, npts):
    #     for j in range(i):
    #         Q[i, j] = Qvec[countcoord]
    #         Q[j, i] = Q[i, j]
    #         countcoord += 1

    np.savetxt(f'covmat_vectorized_{npts}_python.txt', Qvec)
    runtime = time.time() - start_time
    print(f'Ellapsed time: {runtime}')


if __name__ == '__main__':

    p = np.loadtxt('/home/lamap/Documents/CAD/my_points.txt')
    sigma_kl = 0.5
    l_kl = 0.5
    CovarianceMatrix(p, sigma_kl, l_kl)

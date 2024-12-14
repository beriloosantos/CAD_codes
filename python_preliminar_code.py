import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
import time
from math import exp


def CovarianceMatrix(p, sigma_kl, l_kl):

    start_time = time.time()
    npts = np.shape(p)[1]
    Q = np.zeros((npts, npts))
    sigmakl2 = sigma_kl ** 2
    lkl2 = l_kl ** 2
    lpts = range(npts)
    for i in tqdm(lpts, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        for j in range(npts):
            if i <= j:
                p_i = [p[0, i], p[1, i]]
                p_j = [p[0, j], p[1, j]]
                Q[i, j] = sigmakl2 * exp(-distance.euclidean(p_i, p_j) / (2 * lkl2))
            else:
                Q[i, j] = Q[j, i]

    np.savetxt(f'covmat_{npts}_python.txt', Q)
    runtime = time.time() - start_time
    print(f'Ellapsed time: {runtime}')


if __name__ == '__main__':

    p = np.loadtxt('my_points_test.txt')
    sigma_kl = 0.5
    l_kl = 0.5

    CovarianceMatrix(p, sigma_kl, l_kl)

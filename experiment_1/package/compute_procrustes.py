import scipy.linalg as la
import numpy as np

def compute_procrustes(A,epsilon=1e-6):
    d = np.ones(A.shape[1])
    V_old = np.zeros(A.shape)
    V = np.ones(A.shape)
    compteur = 0
    li = []
    while np.linalg.norm(A - V @ np.diag(d)) > epsilon and compteur < 100:
        V_old = V
        V,P = la.polar(A @ np.diag(d))
        d = np.sum(A*V,axis=0)
        compteur += 1
    return V @ np.diag(d)
        
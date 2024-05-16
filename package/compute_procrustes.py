import scipy.linalg as la
import numpy as np

def compute_procrustes(A,epsilon=1e-6):
    d = np.ones(A.shape[1])
    V_old = np.zeros(A.shape)
    V = np.ones(A.shape)
    compteur = 0
    li = []
    while np.linalg.norm(A - V @ np.diag(d)) > epsilon and compteur < 100 and np.linalg.norm(V - V_old) > epsilon:
        V_old = V
        V,P = la.polar(A @ np.diag(d))
        d = np.sum(A*V,axis=0)
        compteur += 1
    return V @ np.diag(d)

def compute_orthonormal_projection(A,epsilon=1e-6):
    u,s,vh = np.linalg.svd(A)       
    return u @ vh


def compute_independent(A):
    M = np.zeros(A.shape)
    A_sq = A**2
    for i in range(A.shape[1]):
        max_idx = np.argmax(np.abs(A_sq[:,i]))
        M[max_idx,i] = A[max_idx,i]
    return M
        
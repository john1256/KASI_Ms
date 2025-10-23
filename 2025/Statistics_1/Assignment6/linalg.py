import numpy as np
def cholesky_arr(cov):
    from numpy.linalg import LinAlgError
    L = np.zeros_like(cov)
    for i in range(cov.shape[0]):
        for j in range(i+1):
            if i == j:
                L[i][j] = np.sqrt(cov[j][j] - np.sum(L[j][:j]**2))
                if L[i][j] <= 0:
                    raise LinAlgError("Matrix is not positive definite")
            else:
                L[i][j] = (cov[i][j] - np.sum(L[i][:j]*L[j][:j])) / L[j][j]
    return L

def LUdecompose(arr):
    n = arr.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i][i] = 1
        for j in range(i, n):
            if i ==0:
                U[i][j] = arr[i][j]
            else:
                U[i][j] = arr[i][j] - np.sum(L[i,:i] * U[:i,j])
        for j in range(i+1, n):
            if i == 0:
                L[j][i] = arr[j][i] / U[i][i]
            else:
                L[j][i] = (arr[j][i] - np.sum(L[j,:i] * U[:i,i])) / U[i][i]
    return L, U

def inv_matrix(arr):
    L, U = LUdecompose(arr)
    n = arr.shape[0]
    inv_L = np.zeros((n, n))
    inv_U = np.zeros((n, n))
    for i in range(n):
        inv_L[i][i] = 1 / L[i][i]
        for j in range(i):
            inv_L[i][j] = -np.sum(L[i,j:i] * inv_L[j:i,j]) / L[i][i]
    for i in range(n-1, -1, -1):
        inv_U[i][i] = 1 / U[i][i]
        for j in range(n-1, i, -1):
            inv_U[i][j] = -np.sum(U[i,i+1:j+1] * inv_U[i+1:j+1,j]) / U[i][i]
    inv_arr = np.dot(inv_U, inv_L)
    return inv_arr
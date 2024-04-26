import numpy as np

def main():
    # example from 421 homework ------------------------------------------------
    # sa = np.array([[-np.sqrt(2) / 2, np.sqrt(2) / 2, 0], [np.sqrt(2) / 2, np.sqrt(2) / 2, 0]])
    # print("sa = " + str(sa))
    # sb = np.array([[-1, -0, 0], [0, 1, 0]])
    # print("sb = " + str(sb))
    # w = np.array([[1], [1]])
    # print(quest(sa, sb, w))

    # de ruiter section 25.2.5 example -----------------------------------------
    # inertial vectors
    sa = np.array([[0.0,1.0,2.0], [1.0,3.0,0.0], [-5.0,0.0,1.0], [1.0,-1.0,4.0], [1.0,1.0,1.0]])
    for i in range(5):
        sa[i,:] = sa[i,:] / np.linalg.norm(sa[i,:])
    # body measurements
    sb = np.array([[0.9082,0.3185,0.2715], [0.5670,0.3732,-0.7343], [-0.2821,0.7163,0.6382], [0.7510,-0.3303,0.5718], [0.9261,-0.2053,-0.3166]])
    # weights
    w = np.array([[1 / 0.01**2], [1 / 0.0325**2], [1 / 0.055**2], [1 / 0.0775**2], [1 / 0.1**2]])
    # quest
    q = quest(sa, sb, w)
    print(q)
    C = q2C(q)
    print(C)
    
def quest(sa, sb, w):
    """
    Uses the QUEST algorithm to compute the optimal quaternion.

    Parameters
    ----------
    sa : np.ndarray
        nx3 array of inertial vectors
    sb : np.ndarray
        nx3 array of body vectors
    w : np.ndarray
        nx1 vector of weights

    Returns
    -------
    q : np.ndarray
        4x1 quaternion [epsilon eta]
    """

    B = np.zeros((3,3))
    lam0 = 0
    n = len(w)
    for i in range(n):
        sa_i = np.atleast_2d(sa[i,:]).T
        sb_i = np.atleast_2d(sb[i,:])
        B += w[i,0] * np.matmul(sa_i, sb_i)
        lam0 += w[i,0] # initial guess as sum of weights 
    B = B.T

    K12 = np.array([[B[1,2] - B[2,1]], [B[2,0] - B[0,2]], [B[0,1] - B[1,0]]])
    K22 = np.trace(B)

    S = B + B.T

    a = K22**2 - np.trace(adj3x3(S))
    b = (K22**2 + np.matmul(K12.T, K12))[0,0]
    c = (np.linalg.det(S) + np.matmul(K12.T, np.matmul(S, K12)))[0,0]
    d = np.matmul(K12.T, np.matmul(np.matmul(S,S), K12))[0,0]

    # lam = newtonRaphson(lam0, a, b, c, d, K22)
    lam = newtonsMethod(lam0, a, b, c, d, K22, 1e-3)

    alpha = lam**2 - K22**2 + np.trace(adj3x3(S))
    beta = lam - K22
    gamma = (lam + K22) * alpha - np.linalg.det(S)
    x = np.matmul(alpha * np.identity(3) + beta * S + np.matmul(S,S), K12)

    q = 1 / np.sqrt(gamma**2 + np.matmul(x.T, x)) * np.atleast_2d(np.append(x, gamma)).T
    return q

# based on matlab's adjoint function - unclear if this is right 
def adj3x3(A):
    """
    Computes the adjoint of a 3x3 matrix, based on MATLAB's implementation

    Parameters
    ----------
    A : numpy.ndarray
        3x3 matrix

    Returns
    -------
    X : numpy.ndarray
        3x3 matrix adjoint
    """
    U, S, Vh = np.linalg.svd(A)
    adjs = np.zeros((3,3))
    adjs[0,0] = S[1] * S[2]
    adjs[1,1] = S[0] * S[2]
    adjs[2,2] = S[0] * S[1]
    X = np.linalg.det(np.matmul(U, Vh)) * np.matmul(Vh.T, np.matmul(adjs, U.T))
    return X

def func(lam, a, b, c, d, K22):
    return lam**4 - (a + b) * lam**2 - c * lam + (a * b + c * K22 - d)

def funcPrime(lam, a, b, c):
    return 4 * lam**3 - 2 * (a + b) * lam - c

# https://www.geeksforgeeks.org/program-for-newton-raphson-method/
def newtonRaphson(lam, a, b, c, d, K22):
    h = func(lam, a, b, c, d, K22) / funcPrime(lam, a, b, c)
    while abs(h) >= 0.0001: # CHANGE TOLERANCE
        h = func(lam, a, b, c, d, K22)/funcPrime(lam, a, b, c)
        lam = lam - h
    return lam

def newtonsMethod(lam0, a, b, c, d, K22, tol):
    lam1 = lam0 - func(lam0, a, b, c, d, K22) / funcPrime(lam0, a, b, c)
    err = abs(lam1 - lam0)
    while err > tol:
        lam0 = lam1
        lam1 = lam0 - func(lam0, a, b, c, d, K22) / funcPrime(lam0, a, b, c)
        err = abs(lam1 - lam0)
    return lam1

def q2C(q):
    epsilon = q[0:3,:]
    eta = q[3,0]
    C = (2 * eta**2 - 1) * np.eye(3) + 2 * np.dot(epsilon, epsilon.T) - 2 * eta * crossMatrix(epsilon)
    return C

def crossMatrix(a):
    ax = a[0,0]
    ay = a[1,0]
    az = a[2,0]
    a = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]])
    return a

main()
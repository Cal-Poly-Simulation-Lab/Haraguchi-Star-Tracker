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
    sa = np.array([[0,1,2], [1,3,0], [-5,0,1], [1,-1,4], [1,1,1]])
    # body measurements
    sb = np.array([[0.9082,0.3185,0.2715], [0.5670,0.3732,-0.7343], [-0.2821,0.7163,0.6382], [0.7510,-0.3303,0.5718], [0.9261,-0.2053,-0.3166]])
    # weights
    sigma = np.array([[0.01], [0.0325], [0.055], [0.0775], [0.1]])
    # quest
    q = quest(sa, sb, sigma)
    print(q)
    C = q2C(q)
    print(C)
    # wrong :( 
    
def quest(sa, sb, sigma):
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
    n = len(sigma)
    for i in range(n):
        sa_i = np.atleast_2d(sa[i,:]).T
        sb_i = np.atleast_2d(sb[i,:]).T
        B += 1 / sigma[i,0]**2 * np.matmul(sa_i, sb_i.T)
    print("B = " + str(B))

    K12 = np.array([[B[1,2] - B[2,1]], [B[2,0] - B[0,2]], [B[0,1] - B[1,0]]])
    K22 = np.trace(B)
    print("K12 = " + str(K12))
    print("K22 = " + str(K22))

    S = B + B.T

    a = K22**2 - np.trace(adj(S))
    b = (K22**2 + np.matmul(K12.T, K12))[0,0]
    c = (np.linalg.det(S) + np.matmul(K12.T, np.matmul(S, K12)))[0,0]
    d = np.matmul(K12.T, np.matmul(S**2, K12))[0,0]

    lam0 = 0
    for i in range(n):
        lam0 += sigma[i,0]
    lam = newtonRaphson(lam0, a, b, c, d, K22)
    print("lam = " + str(lam))

    alpha = lam**2 - K22**2 + np.trace(adj(S))
    beta = lam - K22
    gamma = (lam + K22) * alpha - np.linalg.det(S)
    x = np.matmul(alpha * np.identity(3) + beta * S + S**2, K12)

    q = 1 / np.sqrt(gamma**2 + np.matmul(x.T, x)) * np.atleast_2d(np.append(x, gamma)).T
    return q

# https://stackoverflow.com/questions/51010662/getting-the-adjugate-of-matrix-in-python
def adj(A):
    sel_rows = np.ones(A.shape[0],dtype=bool)
    sel_columns = np.ones(A.shape[1],dtype=bool)
    CO = np.zeros_like(A)
    sgn_row = 1
    for row in range(A.shape[0]):
        # Unselect current row
        sel_rows[row] = False
        sgn_col = 1
        for col in range(A.shape[1]):
            # Unselect current column
            sel_columns[col] = False
            # Extract submatrix
            MATij = A[sel_rows][:,sel_columns]
            CO[row,col] = sgn_row*sgn_col*np.linalg.det(MATij)
            # Reselect current column
            sel_columns[col] = True
            sgn_col = -sgn_col
        sel_rows[row] = True
        # Reselect current row
        sgn_row = -sgn_row
    return CO.T

def func(lam, a, b, c, d, K22):
    return lam**4 - (a + b) * lam**2 - c * lam + (a * b + c * K22 - d)

def funcPrime(lam, a, b, c):
    return 4 * lam**3 + 2 * (a + b) * lam - c

# https://www.geeksforgeeks.org/program-for-newton-raphson-method/
def newtonRaphson(lam, a, b, c, d, K22):
    h = func(lam, a, b, c, d, K22) / funcPrime(lam, a, b, c)
    while abs(h) >= 0.0001: # CHANGE TOLERANCE
        h = func(lam, a, b, c, d, K22)/funcPrime(lam, a, b, c)
        lam = lam - h
    return lam

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
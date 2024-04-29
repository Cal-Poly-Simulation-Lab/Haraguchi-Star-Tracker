import numpy as np
import pandas as pd

def pyramid(path, s, numCandidates, center, unitVectors, a0=0.2588183885561099, a1=1.7755617311227315e-05):
    KSIJ = pd.read_csv(path)
    KSIJ = KSIJ.to_numpy()
    K = KSIJ[:,0]
    S = KSIJ[:,1]
    I = KSIJ[:,2]
    J = KSIJ[:,3]

    r = center # start with center star as reference
    print("reference star = " + str(r))

    I_rk = []
    J_rk = []
    for k in range(numCandidates):
        if k != r:
            print("star k = " + str(k))
            sk = unitVectors[k,:]
            rk = unitVectors[r,:]
            cosTheta = np.dot(sk, rk)
            print("cos = " + str(cosTheta))
            l_bot = max(int(np.floor((cosTheta - a0) / a1)), 0)
            l_top = min(int(np.ceil((cosTheta - a0) / a1)), len(K) - 1)
            print("bottom l = " + str(l_bot))
            print("top l = " + str(l_top))
            k_start = int(K[l_bot] + 1)
            k_end = int(K[l_top])
            print("k start = " + str(k_start))
            print("k end = " + str(k_end))
            for i in range(k_start, k_end + 1):
                I_rk.append(I[i])
                J_rk.append(J[i])
            print(" ")
    print("I = " + str(I_rk))
    print("J = " + str(J_rk))

            # apply k-vector technique to find



import cv2 as cv 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def stOperation(img, minArea, maxArea, h, w, f):
    """
    covers all aspects of star tracker operaiton
    """
    meanIntensity = np.mean(img) # calculate pixel intensity mean for the whole image

    ret,meanThresh = cv.threshold(img, 200, 255, cv.THRESH_BINARY) # threshold using that value
    analysis = cv.connectedComponentsWithStats(meanThresh, 4, cv.CV_16U) # or 32 bit signed, also decide 4 or 8 connectivity
    (numLabels, labeledImg, stats, geomCentroids) = analysis

    # print("numLabels = " + str(numLabels))
    # cv.imshow("binary", meanThresh)
    # cv.waitKey()
    # plt.imshow(meanThresh, cmap='gray', vmin=0, vmax=255)
    plt.imshow(img)
    # plt.show()

    # plt.subplot(1,2,1)

    centroids = []
    unitVectors = []

    # iterate through each label, checking area and circularity
    numCandidates = 0
    centerStar = 0
    minDist = np.sqrt((h/2)**2 + (w/2)**2)
    centerStarList = [] # np.zeros((numLabels-1, 2))
    for i in range(numLabels):
        area = stats[i, cv.CC_STAT_AREA]
        # print(area)
        minArea = 0
        maxArea = 200
        if area >= minArea and area <= maxArea: # area falls within expected range
            circularity = abs(stats[i,cv.CC_STAT_WIDTH] - stats[i,cv.CC_STAT_HEIGHT])
            if circularity <= 2: # circularity falls within expected range
                plt.text(geomCentroids[i,0], geomCentroids[i,1], str(numCandidates), color='white')
                # use stats left and top to get the subset of the original image that contains this star 
                region = img[stats[i,cv.CC_STAT_TOP]:stats[i,cv.CC_STAT_TOP]+stats[i,cv.CC_STAT_HEIGHT],stats[i,cv.CC_STAT_LEFT]:stats[i,cv.CC_STAT_LEFT]+stats[i,cv.CC_STAT_WIDTH]]
                # make mask of the same size to zero out not included pixels in this subregion
                mask = labeledImg[stats[i,cv.CC_STAT_TOP]:stats[i,cv.CC_STAT_TOP]+stats[i,cv.CC_STAT_HEIGHT],stats[i,cv.CC_STAT_LEFT]:stats[i,cv.CC_STAT_LEFT]+stats[i,cv.CC_STAT_WIDTH]]
                mask = (mask == i).astype("uint8") * 255 # unclear if the masking step is even necessary 
                starCandidate = cv.bitwise_and(region, mask) # check that this should be and and not or
                intensityCentroid = getIntensityCentroid(starCandidate, stats[i,cv.CC_STAT_LEFT], stats[i,cv.CC_STAT_TOP])
                # s = getUnitVector(intensityCentroid)

                # use list instead since numpy not meant to be appended to 
                centroids.append(intensityCentroid)
                # unitVectors.append(s)

                # print("star candidate")
                # print("geometric centroid = " + str(geomCentroids[i]))
                # print("intensity centroid = " + str(intensityCentroid))
                # print("X,Y = " + str(rc2XY(intensityCentroid[0], intensityCentroid[1], h, w)))
                X,Y = rc2XY(intensityCentroid[0], intensityCentroid[1], h, w)

                # find which star is the centermost star by computing distance from (X,Y) to the origin
                dist = np.sqrt(X**2 + Y**2)
                if dist < minDist:
                    minDist = dist
                    centerStar = numCandidates
                centerStarList.append([numCandidates, dist])
                # centerStarList[numCandidates,0] = numCandidates
                # centerStarList[numCandidates,1] = dist


                s = getUnitVector(X, Y, f)
                # print(s)
                unitVectors.append(s)
                numCandidates += 1
                # plt.subplot(1,3,1)
                # plt.imshow(region)
                # plt.subplot(1,3,2)
                # plt.imshow(mask)
                # plt.subplot(1,3,3)
                # plt.imshow(starCandidate)
                # plt.show()
    plt.savefig("local_label.png")
    # plt.show()
    # sort center star list by second column 
    centerStarList = np.asarray(centerStarList)
    centerStarList = centerStarList[centerStarList[:,1].argsort()]
    centerStarList = centerStarList[:,0]

    # convert lists to numpy arrays
    centroids = np.array(centroids)

    unitVectors = np.array(unitVectors) # unit vectors don't actually need to be in this function - uhhhh yeah they do 
    t1 = unitVectors[0,:]
    t2 = unitVectors[1,:]
    print(t1)
    print(t2)
    print(np.matmul(t1.T, t2))
    # get a values from K-vector
    a_values = pd.read_csv("a_values.csv")
    a_values = a_values.to_numpy()
    a0 = a_values[0,0]
    a1 = a_values[1,0]

    # iterate through stars, starting with center, until found a match 
    match = False
    for i in range(numCandidates):
        matchesFound = 0

        centerStar = int(centerStarList[i])
        # print("trying star " + str(centerStar))

        # find three other stars closest to center star
        # define center star as s_i
        neighbors = nearestNeighbors(unitVectors, centerStar)
        print("center star = " + str(centerStar))
        print("neighbors = " + str(neighbors))

        # give the stars the right names
        si = centerStar
        sj = neighbors[0]
        sk = neighbors[1]
        sr = neighbors[2]

        # # make hash table for i-k pairs 
        Mik = databaseQuery(si, sk, unitVectors, a0, a1)
        print("Mik = " + str(Mik))

        # make hash table for i-r pairs
        Mir = databaseQuery(si, sr, unitVectors, a0, a1)
        print("Mir = " + str(Mir))

        # repeat for i-j pairs
        Mij = databaseQuery(si, sj, unitVectors, a0, a1)
        print("Mij = " + str(Mij))

        # iterate through pairs in Mij 
        num_ij = len(Mij) # number of pairs in ij 
        i_list = list(Mij.keys())
        j_list = list(Mij.values())
        for j in range(num_ij):
            ci = i_list[j]
            cj = j_list[j]
            ck = Mik.get(ci)
            cr = Mir.get(ci)
            if ck != None and cr != None:
                # check that all 6 distances agree - it's matching wrong if a different pair comes up first 
                # match = True
                matchesFound += 1
                # saving last found in case it's the only one
                ci_match = ci
                cj_match = cj
                ck_match = ck
                cr_match = cr
                # print("found a match")
                # print("ci = " + str(ci) + ", si = " + str(si))
                # print("cj = " + str(cj) + ", sj = " + str(sj))
                # print("ck = " + str(ck) + ", sk = " + str(sk))
                # print("cr = " + str(cr) + ", sr = " + str(sr))
                if matchesFound > 1:
                    break
                # break
        if matchesFound == 1:
            ci = ci_match
            cj = cj_match
            ck = ck_match
            cr = cr_match
            # print("checking distances")
            if distanceCheck(si, sj, sk, sr, ci, cj, ck, cr, unitVectors, a0, a1):
                break
            # check distances here, if they agree then break, if not then check new star

            # break
        # print(str(matchesFound) + " matches found with star " + str(centerStar))

        # need to actually check that the 6 distances agree 

    if matchesFound == 1:
        # print("confirmed")
        # print("ci = " + str(ci) + ", si = " + str(si))
        # print("cj = " + str(cj) + ", sj = " + str(sj))
        # print("ck = " + str(ck) + ", sk = " + str(sk))
        # print("cr = " + str(cr) + ", sr = " + str(sr))

        # make arrays of unit vectors and weights 
        sa = np.empty((4,3))
        sb = np.empty((4,3))
        w = np.ones((4,1))
        
        sb[0,:] = unitVectors[si]
        sb[1,:] = unitVectors[sj]
        sb[2,:] = unitVectors[sk]
        sb[3,:] = unitVectors[sr]

        inertialVectors = pd.read_csv("v_unit_vectors.csv")
        inertialVectors = inertialVectors.to_numpy()

        sa[0,:] = inertialVectors[int(ci)]
        sa[1,:] = inertialVectors[int(cj)]
        sa[2,:] = inertialVectors[int(ck)]
        sa[3,:] = inertialVectors[int(cr)]

        q_ba = quest(sa, sb, w)
        C_ba = q2C(q_ba)

        return q_ba, C_ba
    
    return 0

# finds all possible star distances that agree with the distances between star
# 1 and star 2 
# make a0 and a1 passed in better 
def databaseQuery(s1, s2, unitVectors, a0, a1):
    KSIJ = pd.read_csv("KSIJ_arrays.csv")
    KSIJ = KSIJ.to_numpy()
    K = KSIJ[:,0]
    S = KSIJ[:,1]
    I = KSIJ[:,2]
    J = KSIJ[:,3]
    
    M12 = {}
    # M12 = hashTable()
    I_12 = []
    J_12 = []
    angle = np.arccos(np.dot(unitVectors[s1], unitVectors[s2]))
    error = 0.001 # 0.00078507 # have this value calculated somewhere!!
    l_bot = int(np.floor((np.cos(angle + error) - a0) / a1) - 1)
    l_top = int(np.ceil((np.cos(angle - error) - a0) / a1) - 1) - 1
    k_start = int(K[l_bot])
    k_end = int(K[l_top]) - 1
    # there shouldn't be a plus 1 here but then it doesn't work 
    # print("dist = " + str(dist))
    # print("l = " + str(l_bot) + ", " + str(l_top))
    # print("k = " + str(k_start) + ", " + str(k_end))
    for i in range(k_start, k_end+1):
        I_12.append(I[i])
        J_12.append(J[i])
        M12.update({I[i] : J[i]})
        M12.update({J[i] : I[i]})
        # M12.insert(I[i], J[i])
        # M12.insert(J[i], I[i])
    return M12

def distanceCheck(si, sj, sk, sr, ci, cj, ck, cr, unitVectors, a0, a1):
    KSIJ = pd.read_csv("KSIJ_arrays.csv")
    KSIJ = KSIJ.to_numpy()
    K = KSIJ[:,0]
    S = KSIJ[:,1]
    I = KSIJ[:,2]
    J = KSIJ[:,3]

    local_id = [si, sj, sk, sr]
    global_id = [ci, cj, ck, cr]
    error = 0.00078507 # have this value calculated somewhere!!

    # inertial unit vectors for distance check
    sa = np.empty((4,3))
    inertialVectors = pd.read_csv("v_unit_vectors.csv")
    inertialVectors = inertialVectors.to_numpy()
    sa[0,:] = inertialVectors[int(ci)]
    sa[1,:] = inertialVectors[int(cj)]
    sa[2,:] = inertialVectors[int(ck)]
    sa[3,:] = inertialVectors[int(cr)]

    # body unit vectors for distance check
    sb = np.empty((4,3))
    sb[0,:] = unitVectors[int(si)]
    sb[1,:] = unitVectors[int(sj)]
    sb[2,:] = unitVectors[int(sk)]
    sb[3,:] = unitVectors[int(sr)]

    for s1 in range(4):
        for s2 in range(s1+1,4):
            if s1 != s2:
                # for local angles
                local1 = np.atleast_2d(sb[s1,:]).T
                local2 = np.atleast_2d(sb[s2,:]).T
                angle = np.arccos(np.matmul(local1.T, local2))
                local_bot = np.cos(angle + error)
                local_top = np.cos(angle - error)
                # for global angles
                global1 = np.atleast_2d(sa[s1,:]).T
                global2 = np.atleast_2d(sa[s2,:]).T
                dist = np.matmul(global1.T, global2)
                if dist < local_bot or dist > local_top:
                    # print("distances disagree")
                    return False
    return True


# given the id of the center star, finds the three other stars that are the 
# closest to it 
def nearestNeighbors(unitVectors, centerStar):
    neighbors = [0, 0, 0]
    distances = [0, 0, 0]
    numStars = len(unitVectors)
    for i in range(numStars):
        if i != centerStar:
            dist = np.dot(unitVectors[centerStar], unitVectors[i])
            if dist > min(distances):
                idx = distances.index(min(distances))
                distances[idx] = dist
                neighbors[idx] = i
    return neighbors

# creates SIJ search vectors from array of unit vectors
def searchAngles(unitVectors):
    n = len(unitVectors)
    P = [] # P, I, J
    for i in range(n):
        for j in range(i+1,n):
            if i != j:
                # looks like P should just be dot product, but then sorting it doesn't make sense 
                dot = np.dot(unitVectors[i], unitVectors[j])
                denom = np.linalg.norm(unitVectors[i]) * np.linalg.norm(unitVectors[j])
                angle = np.arccos(dot / denom)
                P.append([dot, i, j])
    P = np.array(P)
    # print("P = ")
    # print(P)

    # sort low to high to get S
    S = P[P[:,0].argsort()] # S, I, J
    # print("S = ") 
    # print(S)

    return S

def getIntensityCentroid(starCandidate, left, top): # need x and y coordinates so need the whole image - or just the new origin?? 
    row,col = starCandidate.shape
    r0_num = 0
    c0_num = 0
    denom = 0
    for r in range(row):
        for c in range(col):
            if starCandidate[r,c] != 0:
                r0_num += starCandidate[r,c] * (top + r) # had a top and left here 
                c0_num += starCandidate[r,c] * (left + c)
                denom += starCandidate[r,c]
    r0 = r0_num / denom
    c0 = c0_num / denom
    return [r0, c0]

# based on markley and crassidis pg 126
def getUnitVector(X, Y, f):
    f = 3385.714286
    s = np.array([X, Y, f]) # as row vector for appending to list 
    s = 1 / np.sqrt(X**2 + Y**2 + f**2) * s
    return s

def rc2XY(r, c, h, w):
    X = c - w/2
    # Y = h/2 - r
    Y = r - h/2
    return (X, Y)

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
    C = (eta**2  - np.matmul(epsilon.T[0], epsilon)) * np.eye(3) + 2 * np.matmul(epsilon, epsilon.T) - 2 * eta * crossMatrix(epsilon)
    return C

def crossMatrix(a):
    ax = a[0,0]
    ay = a[1,0]
    az = a[2,0]
    a = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]])
    return a

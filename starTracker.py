import cv2 as cv 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def stOperation(img, minArea, maxArea, h, w, f):
    """
    Star tracker operation, takes in an image and returns estimated attitude

    Parameters
    ----------
    img : numpy.ndarray
        Image array
    minArea : int
        Minimum area for star spots
    maxArea : int
        Maximum area for star spots
    h : int
        Image height in pixels
    w : int
        Image width in pixels
    f : float
        System focal length in pixels
    
    Returns
    -------
    q_ba : numpy.ndarray
        Quaternion representing rotation from inertial to body frame
    C_ba : numpy.ndarray
        Rotation matrix from inertial to body frame 
    """

    meanIntensity = np.mean(img) # calculate pixel intensity mean for the whole image
    stddevIntensity = np.std(img)

    # threshold and connected components analysis 
    ret,meanThresh = cv.threshold(img, meanIntensity + 2 * stddevIntensity, 255, cv.THRESH_BINARY)
    analysis = cv.connectedComponentsWithStats(meanThresh, 4, cv.CV_16U)
    (numLabels, labeledImg, stats, geomCentroids) = analysis

    centroids = []
    unitVectors = []

    # iterate through each label, checking area and circularity
    numCandidates = 0
    centerStar = 0
    minDist = np.sqrt((h/2)**2 + (w/2)**2)
    centerStarList = []
    for i in range(numLabels):
        area = stats[i, cv.CC_STAT_AREA]
        if area >= minArea and area <= maxArea: # area falls within expected range
            circularity = abs(stats[i,cv.CC_STAT_WIDTH] - stats[i,cv.CC_STAT_HEIGHT])
            if circularity <= 2: # circularity falls within expected range
                plt.text(geomCentroids[i,0], geomCentroids[i,1], str(numCandidates), color='white')
                # use stats left and top to get the subset of the original image that contains this star 
                region = img[stats[i,cv.CC_STAT_TOP]:stats[i,cv.CC_STAT_TOP]+stats[i,cv.CC_STAT_HEIGHT],stats[i,cv.CC_STAT_LEFT]:stats[i,cv.CC_STAT_LEFT]+stats[i,cv.CC_STAT_WIDTH]]
                # make mask of the same size to zero out not included pixels in this subregion
                mask = labeledImg[stats[i,cv.CC_STAT_TOP]:stats[i,cv.CC_STAT_TOP]+stats[i,cv.CC_STAT_HEIGHT],stats[i,cv.CC_STAT_LEFT]:stats[i,cv.CC_STAT_LEFT]+stats[i,cv.CC_STAT_WIDTH]]
                mask = (mask == i).astype("uint8") * 255
                starCandidate = cv.bitwise_and(region, mask)
                intensityCentroid = getIntensityCentroid(starCandidate, stats[i,cv.CC_STAT_LEFT], stats[i,cv.CC_STAT_TOP])
                centroids.append(intensityCentroid)

                X,Y = rc2XY(intensityCentroid[0], intensityCentroid[1], h, w)

                # find which star is the centermost star by computing distance from (X,Y) to the origin
                dist = np.sqrt(X**2 + Y**2)
                if dist < minDist:
                    minDist = dist
                    centerStar = numCandidates
                centerStarList.append([numCandidates, dist])

                s = getUnitVector(X, Y, f)
                unitVectors.append(s)
                numCandidates += 1

    # sort center star list by second column 
    centerStarList = np.asarray(centerStarList)
    centerStarList = centerStarList[centerStarList[:,1].argsort()]
    centerStarList = centerStarList[:,0]

    # convert lists to numpy arrays
    centroids = np.array(centroids)
    unitVectors = np.array(unitVectors) 

    # get a values from K-vector
    a_values = pd.read_csv("a_values.csv")
    a_values = a_values.to_numpy()
    a0 = a_values[0,0]
    a1 = a_values[1,0]

    # iterate through stars, starting with center, until found a match 
    for i in range(numCandidates):
        matchesFound = 0

        centerStar = int(centerStarList[i])

        # find three other stars closest to center star
        neighbors = nearestNeighbors(unitVectors, centerStar)

        # give the stars the right names
        si = centerStar
        sj = neighbors[0]
        sk = neighbors[1]
        sr = neighbors[2]

        # make hash table for i-k pairs 
        Mik = databaseQuery(si, sk, unitVectors, a0, a1, f)
        # make hash table for i-r pairs
        Mir = databaseQuery(si, sr, unitVectors, a0, a1, f)
        # repeat for i-j pairs
        Mij = databaseQuery(si, sj, unitVectors, a0, a1, f)

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
                matchesFound += 1
                # saving last found in case it's the only one
                ci_match = ci
                cj_match = cj
                ck_match = ck
                cr_match = cr
                if matchesFound > 1:
                    break
        if matchesFound == 1:
            ci = ci_match
            cj = cj_match
            ck = ck_match
            cr = cr_match
            # check distances 
            if distanceCheck(si, sj, sk, sr, ci, cj, ck, cr, unitVectors, f):
                break

    if matchesFound == 1:
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

        # estimate quaternion and convert to rotation matrix 
        q_ba = quest(sa, sb, w)
        C_ba = q2C(q_ba)

        return q_ba, C_ba
    
    return 0

def stOperationHardware(img, minArea, maxArea, f):
    """
    Star tracker operation, takes in an image and returns estimated attitude.
    Uses slightly different implementation than software loop, accounts for more
    possible error but takes longer 

    Parameters
    ----------
    img : numpy.ndarray
        Image array
    minArea : int
        Minimum area for star spots
    maxArea : int
        Maximum area for star spots
    f : float
        System focal length in pixels
    
    Returns
    -------
    q_ba : numpy.ndarray
        Quaternion representing rotation from inertial to body frame
    C_ba : numpy.ndarray
        Rotation matrix from inertial to body frame 
    """

    h,w = img.shape

    # threshold and connected component analysis 
    ret,meanThresh = cv.threshold(img, 200, 255, cv.THRESH_BINARY) # threshold using that value
    analysis = cv.connectedComponentsWithStats(meanThresh, 4, cv.CV_16U) # or 32 bit signed, also decide 4 or 8 connectivity
    (numLabels, labeledImg, stats, geomCentroids) = analysis

    centroids = []
    unitVectors = []

    # iterate through each label, checking area and circularity
    numCandidates = 0
    centerStar = 0
    minDist = np.sqrt((h/2)**2 + (w/2)**2)
    centerStarList = []
    for i in range(numLabels):
        area = stats[i, cv.CC_STAT_AREA]
        if area >= minArea and area <= maxArea: # area falls within expected range
            circularity = abs(stats[i,cv.CC_STAT_WIDTH] - stats[i,cv.CC_STAT_HEIGHT])
            if circularity <= 20: # circularity falls within expected range
                plt.text(geomCentroids[i,0], geomCentroids[i,1], str(numCandidates), color='white')
                # use stats left and top to get the subset of the original image that contains this star 
                region = img[stats[i,cv.CC_STAT_TOP]:stats[i,cv.CC_STAT_TOP]+stats[i,cv.CC_STAT_HEIGHT],stats[i,cv.CC_STAT_LEFT]:stats[i,cv.CC_STAT_LEFT]+stats[i,cv.CC_STAT_WIDTH]]
                # make mask of the same size to zero out not included pixels in this subregion
                mask = labeledImg[stats[i,cv.CC_STAT_TOP]:stats[i,cv.CC_STAT_TOP]+stats[i,cv.CC_STAT_HEIGHT],stats[i,cv.CC_STAT_LEFT]:stats[i,cv.CC_STAT_LEFT]+stats[i,cv.CC_STAT_WIDTH]]
                mask = (mask == i).astype("uint8") * 255 # unclear if the masking step is even necessary 
                starCandidate = cv.bitwise_and(region, mask) # check that this should be and and not or
                intensityCentroid = getIntensityCentroid(starCandidate, stats[i,cv.CC_STAT_LEFT], stats[i,cv.CC_STAT_TOP])
                centroids.append(intensityCentroid)

                X,Y = rc2XY(intensityCentroid[0], intensityCentroid[1], h, w)

                # find which star is the centermost star by computing distance from (X,Y) to the origin
                dist = np.sqrt(X**2 + Y**2)
                if dist < minDist:
                    minDist = dist
                    centerStar = numCandidates
                centerStarList.append([numCandidates, dist])

                s = getUnitVector(X, Y, f)
                unitVectors.append(s)
                numCandidates += 1

    centerStarList = np.asarray(centerStarList)
    centerStarList = centerStarList[centerStarList[:,1].argsort()]
    centerStarList = centerStarList[:,0]

    # convert lists to numpy arrays
    centroids = np.array(centroids)
    unitVectors = np.array(unitVectors) 

    # get a values from K-vector
    a_values = pd.read_csv("a_values.csv")
    a_values = a_values.to_numpy()
    a0 = a_values[0,0]
    a1 = a_values[1,0]

    # iterate through stars, starting with center, until found a match 
    goodDistance = False
    for i in range(numCandidates): 
        matchesFound = 0

        centerStar = int(centerStarList[i])

        # find three other stars closest to center star
        neighbors = nearestNeighbors(unitVectors, centerStar)

        # give the stars the right names
        si = centerStar
        sj = neighbors[0]
        sk = neighbors[1]
        sr = neighbors[2]

        # # make hash table for i-k pairs 
        Mik = databaseQuery(si, sk, unitVectors, a0, a1, f)
        # make hash table for i-r pairs
        Mir = databaseQuery(si, sr, unitVectors, a0, a1, f)
        # repeat for i-j pairs
        Mij = databaseQuery(si, sj, unitVectors, a0, a1, f)

        # iterate through pairs in Mij 
        num_ij = len(Mij) # number of pairs in ij 
        i_list = list(Mij.keys())
        j_list = list(Mij.values())
        for j in range(num_ij):
            ci = i_list[j]
            cj = j_list[j]
            ck = Mik.get(ci)
            cr = Mir.get(ci)
            if ck != None and cr != None and len(set([ci, cj, ck, cr])) == 4:
                matchesFound += 1
                if distanceCheck(si, sj, sk, sr, ci, cj, ck, cr, unitVectors, a0, a1):
                    goodDistance = True
                    break

        if goodDistance:
            break

    if goodDistance:
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

        # estimate quaternion and convert to rotation matrix 
        q_ba = quest(sa, sb, w)
        C_ba = q2C(q_ba)

        return q_ba, C_ba
    
    return 0

def databaseQuery(s1, s2, unitVectors, a0, a1, f):
    """
    Performs a database query using the K-vector technique

    Parameters
    ----------
    s1 : int
        Index of first star
    s2 : int
        Index of second star
    unitVectors : numpy.ndarray 
        Array of unit vectors
    a0 : float
        Intercept of costheta line
    a0 : foat
        Slope of costheta line
    f : float
        System focal length

    Returns
    -------
    M12 : hash table
        Hash table mapping possible star 1 indices to possible star 2 indices
    """
    
    KSIJ = pd.read_csv("KSIJ_arrays.csv")
    KSIJ = KSIJ.to_numpy()
    K = KSIJ[:,0]
    I = KSIJ[:,2]
    J = KSIJ[:,3]
    
    M12 = {}
    I_12 = []
    J_12 = []
    angle = np.arccos(np.dot(unitVectors[s1], unitVectors[s2]))
    error = representationError(f)
    l_bot = int(np.floor((np.cos(angle + error) - a0) / a1) - 1)
    l_top = int(np.ceil((np.cos(angle - error) - a0) / a1) - 1) - 1
    k_start = int(K[l_bot])
    k_end = int(K[l_top]) - 1

    for i in range(k_start, k_end+1):
        I_12.append(I[i])
        J_12.append(J[i])
        M12.update({I[i] : J[i]})
        M12.update({J[i] : I[i]})

    return M12

def distanceCheck(si, sj, sk, sr, ci, cj, ck, cr, unitVectors, f):
    """
    Checks if all six interstar distances agree

    Parameters
    ----------
    si-sr : int
        Indices of stars in image
    ci-cr : int
        Indices of stars found through matching
    unitVectors : numpy.ndarray
        Array of unit vectors
    f : float
        System focal length

    Returns
    -------
    boolean
        True if distances agree, false otherwise 
    """
    
    KSIJ = pd.read_csv("KSIJ_arrays.csv")
    KSIJ = KSIJ.to_numpy()
    K = KSIJ[:,0]
    S = KSIJ[:,1]
    I = KSIJ[:,2]
    J = KSIJ[:,3]

    error = representationError(f)

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
                    return False
    return True

def nearestNeighbors(unitVectors, centerStar):
    """
    Given a star, finds its three nearest neighbors

    Parameters
    ----------
    unitVectors : numpy.ndarray
        Array of unit vectors
    centerStar : int
        Index of star to find neighbors for

    Returns
    -------
    neighbors : list
        List of three nearest neighbors
    """
    
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

def getIntensityCentroid(starCandidate, left, top): # need x and y coordinates so need the whole image - or just the new origin?? 
    """
    Computes the centeroid based on the intensity spread of the star spot

    Parameters
    ----------
    starCandidate : numpy.ndarray
        Region of original image constituting a star candidate
    left : int
        Pixel coordinate of left column of candidate region
    top : int
        Pixel coordinate of top row of candidate region

    Returns
    -------
    r0 : float
        Centroid r coordinate
    c0 : float
        Centroid c coordinate 
    """
    
    row,col = starCandidate.shape
    r0_num = 0
    c0_num = 0
    denom = 0
    for r in range(row):
        for c in range(col):
            if starCandidate[r,c] != 0:
                r0_num += starCandidate[r,c] * (top + r)
                c0_num += starCandidate[r,c] * (left + c)
                denom += starCandidate[r,c]
    r0 = r0_num / denom
    c0 = c0_num / denom
    return [r0, c0]

def getUnitVector(X, Y, f):
    """
    Converts image plane coordinates into a unit vector

    Parameters
    ----------
    X : float
        Image plane X coordinate
    Y : float
        Image plane Y coordinate
    f : float
        System focal length in pixels

    Returns
    -------
    s : numpy.ndarray 
        Unit vector in the star tracker frame 
    """
    
    s = np.array([X, Y, f])
    s = 1 / np.sqrt(X**2 + Y**2 + f**2) * s
    return s

def rc2XY(r, c, h, w):
    """
    Convers (r,c) coordinates to (X,Y) coordinates

    Parameters
    ----------
    r : float
        r coordinate
    c : float
        c coordinate
    h : int
        Image height in pixels
    w : int
        Image width in pixels
    
    Returns
    -------
    X : float
        X coordinate
    Y : float
        Y coordinate
    """

    X = c - w/2
    Y = r - h/2
    return (X, Y)

def quest(sa, sb, w):
    """
    Uses the QUEST algorithm to compute the optimal quaternion

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

    lam = newtonsMethod(lam0, a, b, c, d, K22, 1e-3)

    alpha = lam**2 - K22**2 + np.trace(adj3x3(S))
    beta = lam - K22
    gamma = (lam + K22) * alpha - np.linalg.det(S)
    x = np.matmul(alpha * np.identity(3) + beta * S + np.matmul(S,S), K12)

    q = 1 / np.sqrt(gamma**2 + np.matmul(x.T, x)) * np.atleast_2d(np.append(x, gamma)).T
    return q

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
    """
    Polynomial function to solve in QUEST
    """

    return lam**4 - (a + b) * lam**2 - c * lam + (a * b + c * K22 - d)

def funcPrime(lam, a, b, c):
    """
    First derivative of polynomial function to solve in QUEST
    """
    
    return 4 * lam**3 - 2 * (a + b) * lam - c

def newtonsMethod(lam0, a, b, c, d, K22, tol):
    """
    Netwon's method for solving a polynomial

    Parameters
    ----------
    lam0 : float
        Initial guess
    a, b, c, d, K22 : float
        Constants of equation
    tol : float
        Error tolerance
    
    Returns
    -------
    lam1 : float
        Solved value for lambda 
    """
    
    lam1 = lam0 - func(lam0, a, b, c, d, K22) / funcPrime(lam0, a, b, c)
    err = abs(lam1 - lam0)
    while err > tol:
        lam0 = lam1
        lam1 = lam0 - func(lam0, a, b, c, d, K22) / funcPrime(lam0, a, b, c)
        err = abs(lam1 - lam0)
    return lam1

def q2C(q):
    """
    Converts a quaternion to rotation matrix

    Parameters
    ----------
    q : numpy.ndarray
        4x1 quaternion
    
    Returns
    -------
    C : numpy.ndarray
        3x3 rotation matrix 
    """
    
    epsilon = q[0:3,:]
    eta = q[3,0]
    C = (eta**2  - np.matmul(epsilon.T[0], epsilon)) * np.eye(3) + 2 * np.matmul(epsilon, epsilon.T) - 2 * eta * crossMatrix(epsilon)
    return C

def crossMatrix(a):
    """
    Computes the cross matrix of vector a

    Parameters
    ----------
    a : numpy.ndarray
        3x1 vector
    
    Returns
    -------
    a_cross : numpy.ndarray
        3x3 cross matrix 
    """
    
    ax = a[0,0]
    ay = a[1,0]
    az = a[2,0]
    a_cross = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]])
    return a_cross

def representationError(f):
    """
    Computes the expected error in star separation angles based on the pixelated
    representation of star spots

    Parameters
    ----------
    f : float
        System focal length
    
    Returns
    -------
    error : float
        Expected error value 
    """
    
    center = getUnitVector(0.5, 0.5, f)
    corner = getUnitVector(1, 1, f)
    coserror = np.matmul(center.T, corner)
    error = np.arccos(coserror)
    return error

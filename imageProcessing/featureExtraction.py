import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import QUEST

'''
goal of this piece of code is to:
    detect stars in the image that are brighter than the threshold - done in thresholding
    filter out too small and grouped stars - done in connected components labeling
    determine centroids and convert to body vectors 
    calculate brightness of each star - start with purely geometric information 
'''

def detectionAndCentroiding(img, minArea, maxArea, h, w, f):
    meanIntensity = np.mean(img) # calculate pixel intensity mean for the whole image

    ret,meanThresh = cv.threshold(img, meanIntensity, 255, cv.THRESH_BINARY) # threshold using that value
    analysis = cv.connectedComponentsWithStats(meanThresh, 4, cv.CV_16U) # or 32 bit signed, also decide 4 or 8 connectivity
    (numLabels, labeledImg, stats, geomCentroids) = analysis

    # print("numLabels = " + str(numLabels))

    # cv.imshow("binary", meanThresh)
    # cv.waitKey()
    # plt.imshow(meanThresh, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    # plt.subplot(1,2,1)
    plt.imshow(labeledImg)
    plt.title("original labeled image")

    centroids = []
    unitVectors = []

    # iterate through each label, checking area and circularity
    numCandidates = 0
    centerStar = 0
    minDist = np.sqrt((h/2)**2 + (w/2)**2)
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
                
    # plt.savefig('big_dipper_labeled.png')
    # plt.show()

    # convert lists to numpy arrays
    centroids = np.array(centroids)
    # print(centroids)
    unitVectors = np.array(unitVectors) # unit vectors don't actually need to be in this function - uhhhh yeah they do 

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
    Mik = databaseQuery(si, sk, unitVectors)
    print("Mik = " + str(Mik))

    # make hash table for i-r pairs
    Mir = databaseQuery(si, sr, unitVectors)
    print("Mir = " + str(Mir))

    # repeat for i-j pairs
    Mij = databaseQuery(si, sj, unitVectors)
    print("Mij = " + str(Mij))

    # iterate through pairs in Mij 
    num_ij = len(Mij) # number of pairs in ij 
    i_list = list(Mij.keys())
    j_list = list(Mij.values())
    ck = 0
    cr = 0
    for i in range(num_ij):
        ci = i_list[i]
        cj = j_list[i]
        ck = Mik.get(ci)
        cr = Mir.get(ci)
        if ck != 0 and cr != 0:
            break
    print("confirmed??")
    print("ci = " + str(ci) + ", si = " + str(si))
    print("cj = " + str(cj) + ", sj = " + str(sj))
    print("ck = " + str(ck) + ", sk = " + str(sk))
    print("cr = " + str(cr) + ", sr = " + str(sr))

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

    q = QUEST.quest(sa, sb, w)
    print(q)

    C = q2C(q)
    print(C)

    # plt.subplot(1,2,2)
    # plt.ylim(600,0)
    # plt.scatter(geomCentroids[:,0], geomCentroids[:,1], marker='x')
    # plt.scatter(centroids[:,1], centroids[:,0], marker='x') # converting r,c to x,y
    # plt.axis('scaled')
    # plt.title("accepted centroids")
    # plt.show()

    return (unitVectors, numCandidates, centerStar)

# finds all possible star distances that agree with the distances between star
# 1 and star 2 
def databaseQuery(s1, s2, unitVectors, a0=0.2588183885561099, a1=1.7755617311227315e-05):
    KSIJ = pd.read_csv("KSIJ_arrays.csv")
    KSIJ = KSIJ.to_numpy()
    K = KSIJ[:,0]
    S = KSIJ[:,1]
    I = KSIJ[:,2]
    J = KSIJ[:,3]
    
    M12 = {}
    I_12 = []
    J_12 = []
    dist = np.dot(unitVectors[s1], unitVectors[s2])
    l_bot = max(int(np.floor((dist - a0) / a1)), 0)
    l_top = min(int(np.ceil((dist - a0) / a1)), len(K) - 1)
    k_start = int(K[l_bot] + 1)
    k_end = int(K[l_top] + 1) # there shouldn't be a plus 1 here but then it doesn't work 
    # print("dist = " + str(dist))
    # print("l = " + str(l_bot) + ", " + str(l_top))
    # print("k = " + str(k_start) + ", " + str(k_end))
    for i in range(k_start, k_end + 1):
        I_12.append(I[i])
        J_12.append(J[i])
        M12.update({I[i] : J[i]})
    return M12

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
    s = np.array([X, Y, -f]) # as row vector for appending to list 
    s = 1 / np.sqrt(X**2 + Y**2 + f**2) * s
    print(s)
    return s

def rc2XY(r, c, h, w):
    X = c - w/2
    # Y = h/2 - r
    Y = r - h/2
    return (X, Y)

def q2C(q):
    epsilon = q[0:3,:]
    eta = q[3,0]
    C = (eta**2  - np.matmul(epsilon.T[0], epsilon)) * np.eye(3) + 2 * np.matmul(epsilon, epsilon.T) - 2 * eta * crossMatrix(epsilon)
    return C

def crossMatrix(a):
    return np.array([[0, -a[2,0], a[1,0]], [a[2,0], 0, -a[0,0]], [-a[1,0], a[0,0], 0]])

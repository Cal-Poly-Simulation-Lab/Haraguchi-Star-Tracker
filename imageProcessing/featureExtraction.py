import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
goal of this piece of code is to:
    detect stars in the image that are brighter than the threshold - done in thresholding
    filter out too small and grouped stars - done in connected components labeling
    determine centroids and convert to body vectors 
    calculate brightness of each star - start with purely geometric information 
'''

def detectionAndCentroiding(img, minArea, maxArea, h, w, f):
    row,col = img.shape
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
                numCandidates += 1
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
                unitVectors.append(s)
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

    # plot test -------------------------------------------------
    # x = unitVectors[:,0]
    # y = unitVectors[:,1]
    # z = unitVectors[:,2]
    # # Create a sphere
    # r = 1
    # pi = np.pi
    # cos = np.cos
    # sin = np.sin
    # phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    # X = r*sin(phi)*cos(theta)
    # Y = r*sin(phi)*sin(theta)
    # Z = r*cos(phi)

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter(x,y,z)
    # ax.plot_surface(
    #     X, Y, Z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)
    # ax.set_xlim([-1,1])
    # ax.set_ylim([-1,1])
    # ax.set_zlim([-1,1])
    # ax.set_aspect("equal")
    # plt.tight_layout()
    # plt.show()
    # looks good --------------------------------------------------------------



    # plt.subplot(1,2,2)
    # plt.ylim(600,0)
    # plt.scatter(geomCentroids[:,0], geomCentroids[:,1], marker='x')
    # plt.scatter(centroids[:,1], centroids[:,0], marker='x') # converting r,c to x,y
    # plt.axis('scaled')
    # plt.title("accepted centroids")
    # plt.show()

    return (unitVectors, numCandidates, centerStar)

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
    return s

def rc2XY(r, c, h, w):
    X = c - w/2
    Y = h/2 - r
    return (X, Y)

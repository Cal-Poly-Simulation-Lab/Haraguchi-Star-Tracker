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

def detectionAndCentroiding(img, minArea, maxArea):
    row,col = img.shape
    meanIntensity = np.mean(img) # calculate pixel intensity mean for the whole image
    ret,meanThresh = cv.threshold(img, meanIntensity, 255, cv.THRESH_BINARY) # threshold using that value
    analysis = cv.connectedComponentsWithStats(meanThresh, 4, cv.CV_16U) # or 32 bit signed, also decide 4 or 8 connectivity
    (numLabels, labeledImg, stats, geomCentroids) = analysis

    # plt.subplot(1,2,1)
    plt.imshow(labeledImg)
    plt.title("original labeled image")

    centroids = []
    unitVectors = []

    # iterate through each label, checking area and circularity
    numCandidates = 0
    for i in range(numLabels):
        area = stats[i, cv.CC_STAT_AREA]
        if area >= minArea and area <= maxArea: # area falls within expected range
            circularity = abs(stats[i,cv.CC_STAT_WIDTH] - stats[i,cv.CC_STAT_HEIGHT])
            if circularity <= 2: # circularity falls within expected range
                numCandidates += 1
                # use stats left and top to get the subset of the original image that contains this star 
                region = img[stats[i,cv.CC_STAT_TOP]:stats[i,cv.CC_STAT_TOP]+stats[i,cv.CC_STAT_HEIGHT],stats[i,cv.CC_STAT_LEFT]:stats[i,cv.CC_STAT_LEFT]+stats[i,cv.CC_STAT_WIDTH]]
                # make mask of the same size to zero out not included pixels in this subregion
                mask = labeledImg[stats[i,cv.CC_STAT_TOP]:stats[i,cv.CC_STAT_TOP]+stats[i,cv.CC_STAT_HEIGHT],stats[i,cv.CC_STAT_LEFT]:stats[i,cv.CC_STAT_LEFT]+stats[i,cv.CC_STAT_WIDTH]]
                mask = (mask == i).astype("uint8") * 255 # unclear if the masking step is even necessary 
                starCandidate = cv.bitwise_and(region, mask) # check that this should be and and not or
                intensityCentroid = getIntensityCentroid(starCandidate, stats[i,cv.CC_STAT_LEFT], stats[i,cv.CC_STAT_TOP])
                s = getUnitVector(intensityCentroid)

                # use list instead since numpy not meant to be appended to 
                centroids.append(intensityCentroid)
                unitVectors.append(s)

                # print("star candidate")
                # print("geometric centroid = " + str(geomCentroids[i]))
                # print("intensity centroid = " + str(intensityCentroid))
                # print(s)
                # plt.subplot(1,3,1)
                # plt.imshow(region)
                # plt.subplot(1,3,2)
                # plt.imshow(mask)
                # plt.subplot(1,3,3)
                # plt.imshow(starCandidate)
                # plt.show()
    # convert lists to numpy arrays
    centroids = np.array(centroids)
    unitVectors = np.array(unitVectors) # unit vectors don't actually need to be in this function 

    # plt.subplot(1,2,2)
    plt.scatter(geomCentroids[:,0], geomCentroids[:,1], marker='x')
    plt.scatter(centroids[:,1], centroids[:,0], marker='x') # converting r,c to x,y
    plt.axis('scaled')
    plt.title("accepted centroids")
    plt.show()

    return centroids

def getIntensityCentroid(starCandidate, left, top): # need x and y coordinates so need the whole image - or just the new origin?? 
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

# based on markley and crassidis pg 126
def getUnitVector(centroid, f=4.74):
    # f is focal length of the optics in mm, should be in pixels though 
    # (u0, v0) center of the focal plane, raster scanning coordinate system
    u0 = 2592 / 2
    v0 = 4608 / 2
    u = centroid[0]
    v = centroid[1]
    factor = 1 / np.sqrt(f**2 + (u - u0)**2 + (v - v0)**2)
    s = np.array([[factor * (u - u0)], [factor * (v - v0)], [factor * f]])
    return s

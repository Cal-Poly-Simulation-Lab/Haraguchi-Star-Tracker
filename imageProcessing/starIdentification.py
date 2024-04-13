import cv2 as cv
import numpy as np

# not doing it this way at all 

def featureId(img):
    # initialize FAST object with threshold value 
    fast = cv.FastFeatureDetector_create(threshold=100)
    # find and draw the keypoints
    kp = fast.detect(img, None)
    img2 = cv.drawKeypoints(img, kp, None, color=(0,0,255))
    cv.imshow("keypoints", img2)

    # pts = cv.KeyPoint_convert(kp)
    pts = cv.KeyPoint.convert(kp).reshape(-1, 1, 2)
    print(pts)

    cv.waitKey(0)
    cv.destroyAllWindows()
    return

img = cv.imread("sampleStellarium.png", cv.IMREAD_GRAYSCALE)
featureId(img)

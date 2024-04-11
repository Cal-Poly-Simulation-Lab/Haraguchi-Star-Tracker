import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# read in image that will be used to test different threshold values
img = cv.imread("sampleStellarium.png", cv.IMREAD_GRAYSCALE)

# different intensity characteristics
meanIntensity = np.mean(img)
stddevIntensity = np.std(img)

# test different threshold approaches
ret,meanThresh = cv.threshold(img, meanIntensity, 255, cv.THRESH_BINARY)
ret,stddevThresh = cv.threshold(img, meanIntensity - stddevIntensity, 255, cv.THRESH_BINARY)

# apply mean thresholding to image blurred with 3x3 Gaussian
img_blurred = cv.GaussianBlur(img, (3,3), 0)
meanBlurred = np.mean(img_blurred)
ret,blurThresh = cv.threshold(img_blurred, meanBlurred, 255, cv.THRESH_BINARY)

titles = ['Original', 'Mean', 'Stddev below Mean', 'Blurred Mean']
images = [img, meanThresh, stddevThresh, blurThresh]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
plt.show()

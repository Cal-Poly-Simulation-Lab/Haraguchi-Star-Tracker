import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tikzplotlib as tp

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


# read in image that will be used to test different threshold values
img = cv.imread("test_images\sampleStellarium.png", cv.IMREAD_GRAYSCALE)

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

# # apply median threshold to remove salt and pepper noise
# img_median = cv.medianBlur(img,9)
# ret, medianThresh = cv.threshold(img_median, meanIntensity, 255, cv.THRESH_BINARY)

titles = ['Original', 'Mean', 'StdDev Below Mean', 'Blurred Mean']
images = [img, meanThresh, stddevThresh, blurThresh]

for i in range(4):
    fig = plt.figure()
    # plt.subplot(2,2,i+1)
    plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
    # plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
    fig.savefig(titles[i]+' Analysis.png', bbox_inches='tight', pad_inches=0)


import cv2 as cv
import featureExtraction as fe
import databaseGeneration as dg

catalogPath = "bs5_brief.csv"
minMag = 4.4377
maxMag = -1.5829
vPath = "v_unit_vectors.csv"

# only happens once or if parameters have changed
# dg.parseCatalog(catalogPath, minMag, maxMag) # parse catalog to get unit vectors
# dg.K_vector(vPath) # compute K-vector - then can read in vectors from csv file later  


# read image in grayscale
img = cv.imread("test_images/blurredTest.png", cv.IMREAD_GRAYSCALE)

cv.imshow("img", img)
cv.waitKey()

# threshold
fe.detectionAndCentroiding(img, 0, 100)

# tikzplotlib (formerly matplotlib2tikz) creates LaTeX code using the pgfplots package. It's available on Pypi, so it can be installed with pip.

# Basically do

# import tikzplotlib
# and then

# tikzplotlib.save('filename.tex')
# after generating a figure. In your LaTeX-file add

# \usepackage{pgfplots}
# to the preamble, and

# \input{filename}
# to add the figure.

# The tikzplotlib.save function has multiple options for modifying the code, so have a look at its docstring.

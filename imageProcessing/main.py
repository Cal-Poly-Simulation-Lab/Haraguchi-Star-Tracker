import cv2 as cv
import featureExtraction as fe
import databaseGeneration as dg
import starIdentification as si

h = 600 # pixels, screen height
w = 1024 # pixels, screen width
h_cm = 8.988617 # screen height in cm
l = 13.06 # camera / screen separation in cm - should it be lens / screen?
f = h / h_cm * l # system focal length in pixels

catalogPath = "bs5_brief.csv"
# my fake specs
# minMag = 4.4377
# maxMag = -1.5829
# jpl specs
minMag = 3.9045
maxMag = -2.1161
# all stars in catalog
# minMag = 7.96
# maxMag = -1.46
vPath = "v_unit_vectors.csv"

# only happens once or if parameters have changed
# dg.parseCatalog(catalogPath, minMag, maxMag) # parse catalog to get unit vectors
# dg.K_vector(vPath) # compute K-vector - then can read in vectors from csv file later  

# read image in grayscale
img = cv.imread("test_images/big_dipper.png", cv.IMREAD_GRAYSCALE)

# cv.imshow("img", img)
# cv.waitKey()

# threshold
unitVectors, numStars, center = fe.detectionAndCentroiding(img, 9, 25, h, w, f)

# s = fe.searchAngles(unitVectors)

# si.pyramid("KSIJ_arrays.csv", s, numStars, center, unitVectors)


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

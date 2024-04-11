import cv2 as cv
import featureExtraction as fe

# read image in grayscale
img = cv.imread("sampleStellarium.png", cv.IMREAD_GRAYSCALE)

# threshold
fe.detectionAndCentroiding(img, 3, 100)

'''
tikzplotlib (formerly matplotlib2tikz) creates LaTeX code using the pgfplots package. It's available on Pypi, so it can be installed with pip.

Basically do

import tikzplotlib
and then

tikzplotlib.save('filename.tex')
after generating a figure. In your LaTeX-file add

\usepackage{pgfplots}
to the preamble, and

\input{filename}
to add the figure.

The tikzplotlib.save function has multiple options for modifying the code, so have a look at its docstring.
'''
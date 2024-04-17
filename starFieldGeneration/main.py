from param import *
from catalogParsing import * 
import catalogParsing as cp
import staticImage as si
from matplotlib import pyplot as plt
import cv2 as cv

# only happens once or if parameters have changed 
# cp.parseCatalog(catalogPath, minMag, maxMag)

img = si.static_image(dataPath, q_ECI_st, u_st_st, fov, f, h, w, maxStars)

# plt.imshow(img, cmap='gray', vmin=0, vmax=255)
# plt.show()

cv.imshow("img", img)
cv.waitKey()

cv.imwrite("blurredTest.png", img)

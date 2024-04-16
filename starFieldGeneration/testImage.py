import cv2 as cv
import numpy as np

img = np.zeros((600,1024), np.uint8)

# maybe change it to have more rows, or this is good enough 
for i in range(256):
    img[:,i*4:i*4+4] = i

cv.imshow("test", img)
cv.waitKey()

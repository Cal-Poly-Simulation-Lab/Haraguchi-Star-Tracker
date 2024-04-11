import cv2
import numpy as np

img = cv2.imread('sampleStellarium.png', cv2.IMREAD_GRAYSCALE)

ret, im2 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

cv2.imshow("image1", img)
cv2.imshow("image2", im2)
print(type(im2[1,1]))

# orb = cv2.ORB_create()
# key_points, descriptors = orb.detectAndCompute(im2, None)
# result = cv2.drawKeypoints(im2, key_points, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow('ORB Features', result)

test = np.zeros((20,20), np.uint8)
for i in range(10):
    for j in range(10):
        test[i,j] = 255
print(test)

testimg = cv2.imwrite("black.png", test)
readimg = cv2.imread('black.png', 1)
cv2.imshow("image3", readimg)


cv2.waitKey(0)

cv2.destroyAllWindows()

import numpy as np
import cv2 as cv
import scipy.stats as st
from matplotlib import pyplot as plt

def gauss2D(sigma, mean, x):
    Sigma = np.array([[sigma**2, 0], [0, sigma**2]])
    t1 = 1 / (2 * np.pi * np.sqrt(np.linalg.det(Sigma)))
    arg = -0.5 * np.matmul(np.matmul((x - mean).T, np.linalg.inv(Sigma)), (x - mean))
    p = t1 * np.exp(arg)
    return p[0][0]

def gaussianKernel(size, mean, sigma):
    H = np.empty((size,size))
    for i in range(size):
        for j in range(size):
            H[i,j] = gauss2D(sigma, mean, np.array([[i+0.5],[j+0.5]]))
    H = np.divide(H, H[size//2, size//2])
    return H

# centroid = np.array([[2.06], [2.72]])
centroid = np.array([[2.5], [2.5]])
H = gaussianKernel(5, centroid, 1.2)
H = H * 255
print(H)

plt.imshow(H, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.savefig('5x5centered.png', bbox_inches='tight', pad_inches=0)
plt.show()

# img = np.zeros((20,20), np.uint8)

# # get kernel 
# shape = (5,5)
# sigma = 1.2
# H = matlab_style_gauss2D(shape,sigma)
# print(H)

# # normalize so center = 1
# H = H / H[2,2]
# print(H)

# for i in range(5):
#     for j in range(5):
#         img[i+10,j+10] = 255 * H[i,j]

# plt.imshow(img, cmap='gray', vmin=0, vmax=255)
# plt.show()
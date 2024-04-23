import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2 as cv

# problem is here!!!
# as written in textbook 
# def radec2M(alpha, delta, phi):
#     a1 = np.sin(alpha) * np.cos(phi) - np.cos(alpha) * np.sin(delta) * np.sin(phi)
#     a2 = -np.sin(alpha) * np.sin(phi) - np.cos(alpha) * np.sin(delta) * np.cos(phi)
#     a3 = -np.cos(alpha) * np.cos(delta)
#     b1 = -np.cos(alpha) * np.cos(phi) - np.sin(alpha) * np.sin(delta) * np.sin(phi)
#     b2 = np.cos(alpha) * np.sin(phi) - np.sin(alpha) * np.sin(delta) * np.cos(phi)
#     b3 = -np.sin(alpha) * np.cos(delta)
#     c1 = np.cos(alpha) * np.sin(phi)
#     c2 = np.cos(alpha) * np.cos(phi)
#     c3 = -np.sin(delta)
#     M = np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]])
#     return M
# precalculating trig values
# def radec2M(alpha, delta, phi):
#     sa = np.sin(alpha)
#     sd = np.sin(delta)
#     sp = np.sin(phi)
#     ca = np.cos(alpha)
#     cd = np.cos(delta)
#     cp = np.cos(phi)
#     a1 = sa * cp - ca * sd * sp
#     a2 = -sa * sp - ca * sd * cp
#     a3 = -ca * cd
#     b1 = -ca * cp - sa * sd * sp
#     b2 = ca * sp - sa * sd * cp
#     b3 = -sa * cd
#     c1 = ca * sp
#     c2 = ca * cp
#     c3 = -sd
#     M = np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]])
#     return M
# doing each rotation separately
def radec2M(alpha, delta, phi):
    M1 = np.array([[np.cos(alpha - np.pi/2), -np.sin(alpha - np.pi/2), 0], [np.sin(alpha - np.pi/2), np.cos(alpha - np.pi/2), 0], [0, 0, 1]])
    M2 = np.array([[1, 0, 0], [0, np.cos(delta + np.pi/2), -np.sin(delta + np.pi/2)], [0, np.sin(delta + np.pi/2), np.cos(delta + np.pi/2)]])
    M3 = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    Mi = np.matmul(M2, M3)
    M = np.matmul(M1, Mi)
    return M
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

# system parameters
fovx = np.deg2rad(66) # rad, square fov
fovy = np.deg2rad(41)
h = 600 # pixels, screen height
w = 1024 # pixels, screen width
h_cm = 8.988617 # screen height in cm
l = 13.06 # camera / screen separation in cm - should it be lens / screen?
f = h / h_cm * l # system focal length in pixels 
star_size = 5
border = star_size // 2

# big dipper
ra0 = np.deg2rad(195)
dec0 = np.deg2rad(55)
roll0 = np.deg2rad(180)
# orion
# ra0 = np.deg2rad(75)
# dec0 = np.deg2rad(0.083333)
# roll0 = np.deg2rad(0)

data = pd.read_csv("star_generation_data.csv")
data_all = data.to_numpy()

R = np.sqrt(fovx**2 + fovy**2) / 2

ra_min = ra0 - R / np.cos(dec0) # min/max ra 
ra_max = ra0 + R / np.cos(dec0)

ra_case = 0 # case 0 = ok, case 1 = split
if ra_min < 0:
    ra_min += 2 * np.pi
    ra_case = 1
elif ra_max > 2 * np.pi:
    ra_max -= 2 * np.pi
    ra_case = 1

dec_min = dec0 - R # dec doesn't wrap you goof 
dec_max = dec0 + R

num_entries = len(data_all)
ra = []
dec = []
inten = []
for i in range(num_entries):
    ra_in_view = False
    dec_in_view = False
    match ra_case: # ra is first column no data_all[i,0]
        case 0:
            if data_all[i,0] > ra_min and data_all[i,0] < ra_max:
                ra_in_view = True
        case 1:
            if data_all[i,0] > ra_min or data_all[i,0] < ra_max:
                ra_in_view = True
    if data_all[i,1] > dec_min and data_all[i,1] < dec_max:
        dec_in_view = True
    if ra_in_view and dec_in_view:
        ra.append(data_all[i,0])
        dec.append(data_all[i,1])
        inten.append(data_all[i,3])

# rotation matrix from icrf to st 
M =  radec2M(ra0, dec0, roll0)
# print("RA = " + str(ra0))
# print("M =  " + str(M))
# print("M.T * M = " + str(np.matmul(M.T, M)))
# print("det(M) = " + str(np.linalg.det(M)))

# create image array
img = np.zeros((h,w), np.uint8)

num_stars = len(ra)
x = []
y = []
z = []
for i in range(num_stars):
    rai = ra[i]
    deci = dec[i]
    u_star_ECI = np.array([[np.cos(deci) * np.cos(rai)],
                            [np.cos(deci) * np.sin(rai)],
                            [np.sin(deci)]])
    # transform to star tracker frame 
    u_star_st = np.matmul(M.T, u_star_ECI)
    # x.append(u_star_st[0,0])
    # y.append(u_star_st[1,0])
    # z.append(u_star_st[2,0])
    X = f * u_star_st[0,0] / u_star_st[2,0] # check both ways 
    Y = f * u_star_st[1,0] / u_star_st[2,0]

    r = -1 * (Y - (h/2)) # unrounded centroids, directly from math in r,c coordinates 
    c = X + (w/2)
    r_local = (r % 1) + border # local centroids, center of starSize x starSize square 
    c_local = (c % 1) + border
    r_floor = int(r // 1) # floored centroids, used for finding center 
    c_floor = int(c // 1)
    # print("r floor, c floor = " + str(r_floor) + ", " + str(c_floor))
    H = gaussianKernel(star_size, np.array([[r_local],[c_local]]), 1.2)
    if 0 <= r_floor-border and r_floor+border < h and 0 <= c_floor-border and c_floor+border < w:
        for i in range(star_size):
            for j in range(star_size):
                if img[r_floor-border+i, c_floor-border+j] + 255 * H[i,j] > 255:
                    img[r_floor-border+i, c_floor-border+j] = 255
                else:
                    img[r_floor-border+i, c_floor-border+j] += 255 * H[i,j]
# plt.imshow(img, cmap='gray', vmin=0, vmax=255)
# plt.show()
cv.imshow("big dipper", img)
cv.waitKey()
# cv.imwrite("big_dipper.png", img)

    # # Create a sphere
    # r = 1
    # pi = np.pi
    # cos = np.cos
    # sin = np.sin
    # phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    # X = r*sin(phi)*cos(theta)
    # Y = r*sin(phi)*sin(theta)
    # Z = r*cos(phi)

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter(x,y,z)
    # ax.plot_surface(
    #     X, Y, Z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)
    # ax.set_xlim([-1,1])
    # ax.set_ylim([-1,1])
    # ax.set_zlim([-1,1])
    # ax.set_aspect("equal")
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig('dec_st_vid/'+str(angle)+'.png')
    # plt.close()


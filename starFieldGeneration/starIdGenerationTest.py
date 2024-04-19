import numpy as np
import catalogParsing as cp
import cv2 as cv
from matplotlib import pyplot as plt
import pandas as pd

# tests the star field generation algorithm using parameters and sample image 
# provided in star id book 

def main():
    # attitude parameters 
    ra = np.deg2rad(249.2104) # rad
    dec = np.deg2rad(-12.0386) # rad
    roll = np.deg2rad(13.3845) # rad
    ra = 0
    dec = 0
    roll = 0
    # ra = np.deg2rad(0)
    # dec = np.deg2rad(10)
    # roll = np.deg2rad(30)

    # book example -------------------------------------
    # physical system parameters 
    fovx = np.deg2rad(12) # rad, square fov
    fovy = np.deg2rad(12)
    h = 1024 # pixels, screen height
    w = 1024 # pixels, screen width
    f = 58.4536 # mm, system focal length
    # 1 pixel is 12x12 micrometers
    f = f * 12 * 0.001
    # new paper idea
    f = 1 / np.tan(fovx / 2)
    print("f = " + str(f))

    # my parameters -------------------------------------
    # physical system parameters 
    fovx = np.deg2rad(66) # rad, square fov
    fovy = np.deg2rad(41)
    h = 600 # pixels, screen height
    w = 1024 # pixels, screen width
    h_cm = 8.988617 # screen height in cm
    l = 13.06 # camera / screen separation in cm - should it be lens / screen?
    f = h / h_cm * l # system focal length in pixels 
    # print("f = " + str(f))

    # catalog parameters
    minMag = 6
    maxMag = -1.46

    # cp.parseCatalog("bs5_brief.csv", minMag, maxMag)
    img = static_image("star_generation_data.csv", ra, dec, roll, fovx, fovy, f, h, w, 50)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()

def static_image(dataPath, ra0, dec0, roll0, fovx, fovy, f, h, w, maxStars):
    """Generates a star field image in a given location
    
    Parameters
    ----------
    path : str
        Path to csv file with star generation data
    q_ECI_st : numpy.ndarray
        Quaternion for rotation from star tracker frame to ECI
    u_st_st : numpy.ndarray
        Star tracker boresight direction in star tracker frame
    fov : float

    f : float
        System focal length in pixels
    h : float
        Screen height in pixels
    w : float
        Screen width in pixels

    Returns
    -------
    img : numpy.ndarray
        Image array
    """

    data = pd.read_csv(dataPath)
    data_all = data.to_numpy()
    
    # convert st boresight direction to celestial ra and dec 
    # u_st_ECI = rot(u_st_st, q_ECI_st) # st boresight in ECI
    # (alpha, delta) = r2radec(u_st_ECI) # ECI vector to RA and DEC
    alpha = ra0
    delta = dec0
    # print("looking at sky at RA = %f rad, DEC = %f rad" %(alpha, delta))

    # find range of star ra/dec values that will be in view 
    R = np.sqrt(fovx**2 + fovy**2) # from new paper 
    # R = np.sqrt(2 * fov**2) / 2 # radius of circular field of view from star id book 
    ra_min = alpha - R / np.cos(delta) # min/max ra 
    ra_max = alpha + R / np.cos(delta)

    ra_case = 0 # case 0 = ok, case 1 = split
    if ra_min < 0:
        ra_min += 2 * np.pi
        ra_case = 1
    elif ra_max > 2 * np.pi:
        ra_max -= 2 * np.pi
        ra_case = 21

    dec_min = delta - R
    dec_max = delta + R

    dec_case = 0 # case 0 = ok, case 1 = split
    if dec_min < -np.pi / 2:
        dec_min += np.pi
        dec_case = 1
    if dec_max > np.pi / 2:
        dec_max -= np.pi
        dec_case = 1

    print("ra min and max %f, %f dec min and max %f, %f" %(ra_min, ra_max, dec_min, dec_max))
    
    # get lists of all stars in view 
    num_entries = len(data_all) # number of entries in catalog arrays
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
        match dec_case: # dec is data_all[i,1]
            case 0:
                if data_all[i,1] > dec_min and data_all[i,1] < dec_max:
                    dec_in_view = True
            case 1:
                if data_all[i,1] > dec_min or data_all[i,1] < dec_max:
                    dec_in_view = True
        if ra_in_view and dec_in_view:
            ra.append(data_all[i,0])
            dec.append(data_all[i,1])
            inten.append(data_all[i,3])
    
    # create image array 
    img = np.zeros((h,w), np.uint8)

    # convert ra,dec -> star tracker -> focal plane (u,v)
    # num_stars = min(len(ra), maxStars)
    num_stars = len(ra)
    for i in range(num_stars):
        rai = ra[i]
        deci = dec[i]
        u_star_ECI = np.array([[np.cos(deci) * np.cos(rai)],
                               [np.cos(deci) * np.sin(rai)],
                               [np.sin(deci)]])
        M = radec2M(ra0, dec0, roll0)
        u_star_st = np.matmul(M.T, u_star_ECI)
        # print(u_star_st)
        # avoid matrix multiplication
        X = f * (M[0,0] * u_star_ECI[0,0] + M[1,0] * u_star_ECI[1,0] + M[2,0] * u_star_ECI[2,0]) / (M[0,2] * u_star_ECI[0,0] + M[1,2] * u_star_ECI[1,0] + M[2,2] * u_star_ECI[2,0])
        Y = f * (M[0,1] * u_star_ECI[0,0] + M[1,1] * u_star_ECI[1,0] + M[2,1] * u_star_ECI[2,0]) / (M[0,2] * u_star_ECI[0,0] + M[1,2] * u_star_ECI[1,0] + M[2,2] * u_star_ECI[2,0])

        u0 = w / 2 # check which ones these should be - or if we make it zero will origin be easy?? 
        v0 = h / 2

        T = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]])

        coords = np.matmul(T, u_star_st)
        # X = coords[0,0] / coords[2,0]
        # Y = coords[1,0] / coords[2,0]
        # print("X,Y + " + str(X) + ", " + str(Y))

        # u = f * u_star_st[1,0] / u_star_st[0,0] # have this be a condition depending on boresight direction
        # v = f * u_star_st[2,0] / u_star_st[0,0] # what did I mean by the above 
        # print("u,v = " + str(u) + ", " + str(v))
        starSize = 9
        border = starSize // 2

        # used to be v and u for r and c
        r = -1 * (Y - (h/2)) # unrounded centroids, directly from math in r,c coordinates 
        c = X + (w/2)

        # print("r,c = " + str(r) + ", " + str(c))

        r_local = (r % 1) + border # local centroids, center of starSize x starSize square 
        c_local = (c % 1) + border
        r_floor = int(r // 1) # floored centroids, used for finding center 
        c_floor = int(c // 1)
        # print("r floor, c floor = " + str(r_floor) + ", " + str(c_floor))
        H = gaussianKernel(starSize, np.array([[r_local],[c_local]]), 1.2)
        if 0 <= r_floor-border and r_floor+border < h and 0 <= c_floor-border and c_floor+border < w:
            for i in range(starSize):
                for j in range(starSize):
                    if img[r_floor-border+i, c_floor-border+j] + 255 * H[i,j] > 255:
                        img[r_floor-border+i, c_floor-border+j] = 255
                    else:
                        img[r_floor-border+i, c_floor-border+j] += 255 * H[i,j]

    return img

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

def q_star(q):
    """
    Computes the quaternion conjugate q*

    Parameters
    ----------
    q : numpy.ndarray
        Quaternion [epsilon; eta]

    Returns
    -------
    q_star : numpy.ndarray
        Quaternion conjugate [epsilon; eta]
    """
    q_star = np.array([[-1 * q[0,0]], [-1 * q[1,0]], [-1 * q[2,0]], [q[3,0]]])
    return q_star

def q_mult_cross(p, q):
    """
    Computes the cross form of quaternion multiplication

    Parameters
    ----------
    p : numpy.ndarray
        First quaternion
    q : numpy.ndarray
        Second quaterion

    Returns
    -------
    q_ret : numpy.ndarray
        Quaternion product 
    """
    q_ret = np.array([p[0] * q[3] - p[1] * q[2] + p[2] * q[1] + p[3] * q[0],
                      p[0] * q[2] - p[2] * q[0] + p[1] * q[3] + p[3] * q[1],
                      p[1] * q[0] - p[0] * q[1] + p[2] * q[3] + p[3] * q[2],
                      p[3] * q[3] - np.vdot(p[0:3], q[0:3])])
    return q_ret

def rot(v, q):
    """
    Rotates vector v by the quaternion q

    Parameters
    ----------
    v : numpy.ndarray
        Vector
    q : numpy.ndarray
        Quaternion

    Returns
    -------
    v_rot : numpy.ndarray
        Rotated vector
    """
    q_s = q_star(q)
    v_q = np.array([v[0], v[1], v[2], [0]])
    v_rot = q_mult_cross(q_mult_cross(q, v_q), q_s)
    v_rot = v_rot[0:3]
    return v_rot

def r2radec(r):
    """
    Converts vector r to right ascension and declination

    Parameters
    ----------
    r : numpy.ndarray
        Vector

    Returns
    -------
    ra : float
        Right ascension
    dec : float
        Declination 
    """
    u = r / np.linalg.norm(r)
    l = u[0]
    m = u[1]
    n = u[2]
    dec = np.arcsin(n)
    ra = np.arccos(l / np.cos(dec))
    if m <= 0:
        ra = 2 * np.pi - ra
    return ra[0], dec[0]

# check these copied over from text
def radec2M(ra, dec, roll):
    a1 = np.sin(ra) * np.cos(roll) - np.cos(ra) * np.sin(dec) * np.sin(roll)
    a2 = -np.sin(ra) * np.sin(roll) - np.cos(ra) * np.sin(dec) * np.cos(roll)
    a3 = -np.cos(ra) * np.cos(dec)
    b1 = -np.cos(ra) * np.cos(roll) - np.sin(ra) * np.sin(dec) * np.sin(roll)
    b2 = np.cos(ra) * np.sin(roll) - np.sin(ra) * np.sin(dec) * np.cos(roll)
    b3 = -np.sin(ra) * np.cos(dec)
    c1 = np.cos(ra) * np.sin(roll)
    c2 = np.cos(ra) * np.cos(roll)
    c3 = -np.sin(dec)
    M = np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]])
    return M

main()
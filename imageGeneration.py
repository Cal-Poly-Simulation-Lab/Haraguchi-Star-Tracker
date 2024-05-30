import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def staticImage(dataPath, ra0, dec0, roll0, fovx, fovy, f, h, w, starSize, sigma):
    """Generates a star field image in a given location
    
    Parameters
    ----------
    dataPath : str
        Path to csv file with star generation data
    ra0 : float
        Boresight right ascension
    dec0 : float
        Boresight declination
    roll0 : float
        Boresight roll
    fovx : float
        Camera field of view in the horizontal direction
    fovy : float
        Camera field of view in the vertical direction
    f : float
        System focal length in pixels
    h : float
        Screen height in pixels
    w : float
        Screen width in pixels
    starSize : int
        Single dimension of star spot representation
    sigma : float
        Standard deviation for Gaussian spread of stars 

    Returns
    -------
    img : numpy.ndarray
        Image array
    """

    data = pd.read_csv(dataPath)
    data_all = data.to_numpy()

    # ra in range 0 to 2pi, dec in range -pi to pi
    R = np.sqrt(fovx**2 + fovy**2) / 2 # radius of circular field of view 
    ra_min = ra0 - R / np.cos(dec0)
    ra_max = ra0 + R / np.cos(dec0)

    ra_case = 0 # case 0 = ok, case 1 = exceeds min/max 
    if ra_min < 0:
        ra_min += 2 * np.pi
        ra_case = 1
    elif ra_max > 2 * np.pi:
        ra_max -= 2 * np.pi
        ra_case = 1

    dec_min = dec0 - R
    dec_max = dec0 + R
    
    # get lists of all stars in view 
    num_entries = len(data_all) # number of entries in catalog arrays
    ra = []
    dec = []
    inten = []
    index = []
    for i in range(num_entries):
        in_view = False
        if data_all[i,1] > dec_min and data_all[i,1] < dec_max: # dec in view so check ra 
            match ra_case:
                case 0:
                    if data_all[i,0] > ra_min and data_all[i,0] < ra_max:
                        in_view = True
                case 1:
                    if data_all[i,0] > ra_min or data_all[i,0] < ra_max:
                        in_view = True
        if in_view:
            ra.append(data_all[i,0])
            dec.append(data_all[i,1])
            inten.append(data_all[i,3])
            index.append(i)

    # get rotation matrix from st to icrf
    Cbi = radec2M(ra0, dec0, roll0)
    
    # create image array with noisy background
    img = np.empty((h,w), np.uint8)
    for r in range(h):
        for c in range(w):
            img[r,c] = np.random.randint(5, 16)
    # add random spot noise
    count = np.random.randint(0, 10)
    for i in range(count):
        r = np.random.randint(0, h)
        c = np.random.randint(0, w)
        img[r,c] = 255

    # for star spot centroiding
    border = starSize // 2

    # convert ra,dec -> star tracker -> focal plane (u,v)
    num_stars = len(ra)
    for idx in range(num_stars):
        rai = ra[idx]
        deci = dec[idx]

        u_star_ICRF = np.array([[np.cos(deci) * np.cos(rai)],
                               [np.cos(deci) * np.sin(rai)],
                               [np.sin(deci)]])

        # transform to star tracker frame
        u_star_st = np.matmul(Cbi, u_star_ICRF)

        # project into image plane
        X = f * u_star_st[0,0] / u_star_st[2,0]
        Y = f * u_star_st[1,0] / u_star_st[2,0]

        # centroid in image bounds
        r = Y + h/2
        c = X + w/2
        # local centroids (center of starSize x starSize square)
        r_local = (r % 1) + border
        c_local = (c % 1) + border
        # floored centroids to identify center pixel
        r_floor = int(r // 1)
        c_floor = int(c // 1)

        # represent star spot as gaussian 
        H = gaussianKernel(starSize, np.array([[r_local], [c_local]]), sigma)
        if 0 <= r_floor-border and r_floor+border < h and 0 <= c_floor-border and c_floor+border < w:            
            for i in range(starSize):
                for j in range(starSize):
                    if img[r_floor-border+i, c_floor-border+j] + inten[idx] * H[i,j] > 255:
                        img[r_floor-border+i, c_floor-border+j] = 255
                    else:
                        img[r_floor-border+i, c_floor-border+j] += inten[idx] * H[i,j]
    
    return img

def gauss2D(sigma, mean, x):
    """
    Computes 2D Gaussian PDF value at specific point

    Parameters
    ----------
    sigma : float
        Standard deviation
    mean : numpy.ndarray
        Distribution mean
    x : numpy.ndarray
        Point of interest
    
    Returns
    -------
    p : float
        PDF evaluated at x
    """
    
    Sigma = np.array([[sigma**2, 0], [0, sigma**2]])
    t1 = 1 / (2 * np.pi * np.sqrt(np.linalg.det(Sigma)))
    arg = -0.5 * np.matmul(np.matmul((x - mean).T, np.linalg.inv(Sigma)), (x - mean))
    p = t1 * np.exp(arg)
    return p[0][0]

def gaussianKernel(size, mean, sigma):
    """
    Computes Gaussian kernel 

    Parameters
    ----------
    size : int
        Size x size of kernel
    mean : float
        Distribution mean
    sigma : float
        Standard deviation

    Returns
    -------
    H : numpy.ndarray
        Size x size array with kernel values 
    """
    
    H = np.empty((size,size))
    for i in range(size):
        for j in range(size):
            H[i,j] = gauss2D(sigma, mean, np.array([[i+0.5],[j+0.5]]))
    H = np.divide(H, H[size//2, size//2])
    return H

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

def radec2M(alpha, delta, phi):
    """
    Computes the rotation matrix from the star tracker coordinate system to the 
    celestial coordinate system

    Parameters
    ----------
    alpha : float
        Right ascension in radians
    delta : float
        Declination in radians
    phi : float
        Roll in radians

    Returns
    -------
    M : numpy.ndarray
        Rotation matrix 
    """

    M1 = np.array([[np.cos(alpha - np.pi/2), -np.sin(alpha - np.pi/2), 0], [np.sin(alpha - np.pi/2), np.cos(alpha - np.pi/2), 0], [0, 0, 1]])
    M2 = np.array([[1, 0, 0], [0, np.cos(delta - np.pi/2), -np.sin(delta - np.pi/2)], [0, np.sin(delta - np.pi/2), np.cos(delta - np.pi/2)]])
    M3 = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    Mi = np.matmul(M2, M3)
    M = np.matmul(M1, Mi)
    return M.T

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def staticImage(dataPath, ra0, dec0, roll0, fovx, fovy, f, h, w, maxStars, starSize, sigma):
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
    # print("looking at sky at RA = %f rad, DEC = %f rad" %(alpha, delta))

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

    dec_min = dec0 - R # dec doesn't wrap
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
    
    # create image array 
    img = np.zeros((h,w), np.uint8)

    # for star spot centroiding
    border = starSize // 2

    # convert ra,dec -> star tracker -> focal plane (u,v)
    # num_stars = min(len(ra), maxStars) # maybe take away the max stars thing 
    num_stars = len(ra)
    for idx in range(num_stars):
        rai = ra[idx]
        deci = dec[idx]
        
        u_star_ICRF = np.array([[np.cos(deci) * np.cos(rai)],
                               [np.cos(deci) * np.sin(rai)],
                               [np.sin(deci)]])
        # transform to star tracker framt
        u_star_st = np.matmul(Cbi, u_star_ICRF)

        # u_star_st[2,0] = -1 * u_star_st[2,0]
        # print(u_star_st)

        # project into image plane
        X = f * u_star_st[0,0] / u_star_st[2,0]
        Y = f * u_star_st[1,0] / u_star_st[2,0]

        # print("X,Y = " + str(X) + ", " + str(Y))

        # centroid in image bounds
        # r = -1 * (Y - h/2)
        r = Y + h/2
        c = X + w/2

        plt.text(c, r, str(index[idx]), color='white')

        # print("r,c = " + str(r) + ", " + str(c))
        # print("intensity = " + str(inten[i]))
        # local centroids (center of starSize x starSize square)
        r_local = (r % 1) + border
        c_local = (c % 1) + border
        # floored centroids to identify center pixel
        r_floor = int(r // 1)
        c_floor = int(c // 1)

        H = gaussianKernel(starSize, np.array([[r_local], [c_local]]), sigma)
        if 0 <= r_floor-border and r_floor+border < h and 0 <= c_floor-border and c_floor+border < w:
            for i in range(starSize):
                for j in range(starSize):
                    if img[r_floor-border+i, c_floor-border+j] + inten[i] * H[i,j] > 255:
                        img[r_floor-border+i, c_floor-border+j] = 255
                    else:
                        img[r_floor-border+i, c_floor-border+j] += inten[i] * H[i,j]

    plt.imshow(img)
    # plt.savefig("global_label.png")
    # plt.show()
    
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

# star identification pg 50
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
    # I changed M2 to have -pi/2 instead because then it rotates pointing vector to [0; 0; 1]
    M2 = np.array([[1, 0, 0], [0, np.cos(delta - np.pi/2), -np.sin(delta - np.pi/2)], [0, np.sin(delta - np.pi/2), np.cos(delta - np.pi/2)]])
    M3 = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    Mi = np.matmul(M2, M3)
    M = np.matmul(M1, Mi)
    return M.T

import cv2 as cv
import numpy as np
import pandas as pd

def static_image(dataPath, q_ECI_st, u_st_st, fov, f, h, w, maxStars):
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
    u_st_ECI = rot(u_st_st, q_ECI_st) # st boresight in ECI
    (alpha, delta) = r2radec(u_st_ECI) # ECI vector to RA and DEC
    # print("looking at sky at RA = %f rad, DEC = %f rad" %(alpha, delta))

    # find range of star ra/dec values that will be in view 
    fov = fov * np.pi / 180 # convert fov to rad

    fov_type = "book" # book or rectangular
    if fov_type == "book":
        R = np.sqrt(2 * fov**2) / 2 # radius of circular field of view from star id book 
        ra_min = alpha - R / np.cos(delta) # min/max ra 
        ra_max = alpha + R / np.cos(delta)
        dec_min = delta - R

        if dec_min < -1 * np.pi / 2: # limit dec to +/-pi/2
            dec_min = -1 * np.pi / 2
        dec_max = delta + R
        if dec_max > np.pi / 2:
            dec_max = np.pi / 2
    elif fov_type == "rectangular": # spread it out the right amount but seems to wide or something 
        R_ra = np.sqrt(2 * (66 * np.pi / 180)**2) / 2 # 66 * np.pi / 180 / 2
        R_dec = np.sqrt(2 * (41 * np.pi / 180)**2) # 41 * np.pi / 180 / 2
        ra_min = alpha - R_ra / np.cos(delta) # min/max ra 
        ra_max = alpha + R_ra / np.cos(delta)
        dec_min = delta - R_dec

        if dec_min < -1 * np.pi / 2: # limit dec to +/-pi/2
            dec_min = -1 * np.pi / 2
        dec_max = delta + R_dec
        if dec_max > np.pi / 2:
            dec_max = np.pi / 2

    # print("ra min and max %f, %f dec min and max %f, %f" %(ra_min, ra_max, dec_min, dec_max))
    
    # get lists of all stars in view 
    num_entries = len(data_all) # number of entries in catalog arrays
    ra = []
    dec = []
    inten = []
    for i in range(num_entries):
        in_view = False
        if data_all[i,1] > dec_min and data_all[i,1] < dec_max: # dec in range
            # for ra need to consider wrapping since ranges 0-2pi
            if ra_max > (2 * np.pi): # range wraps around
                if data_all[i,0] > ra_min or data_all[i,0] < (ra_max - 2 * np.pi):
                    in_view = True
            elif ra_min < 0: # wraps on other end
                if data_all[i,0] < ra_max or data_all[i,0] > (ra_min + 2 * np.pi):
                    in_view = True
            elif data_all[i,0] > ra_min or data_all[i,0] < ra_max: # regular case
                in_view = True
        if in_view:
            ra.append(data_all[i,0])
            dec.append(data_all[i,1])
            inten.append(data_all[i,3])
    
    # create image array 
    img = np.zeros((h,w), np.uint8)

    # convert ra,dec -> ECI -> star tracker -> focal plane (u,v)
    num_stars = min(len(ra), maxStars)
    for i in range(num_stars):
        rai = ra[i]
        deci = dec[i]
        u_star_ECI = np.array([[np.cos(deci) * np.cos(rai)],
                               [np.cos(deci) * np.sin(rai)],
                               [np.sin(deci)]])
        u_star_st = rot(u_star_ECI, q_star(q_ECI_st))
        u = f * u_star_st[1,0] / u_star_st[0,0] # have this be a condition depending on boresight direction
        v = f * u_star_st[2,0] / u_star_st[0,0] # what did I mean by the above 

        starSize = 9
        border = starSize // 2

        r = v + (h / 2) # unrounded centroids, directly from math in r,c coordinates 
        c = u + (w / 2)

        print("centroid = " + str(r) + ", " + str(c))

        r_local = (r % 1) + border # local centroids, center of starSize x starSize square 
        c_local = (c % 1) + border
        r_floor = int(r // 1) # floored centroids, used for finding center 
        c_floor = int(c // 1)
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

import numpy as np

def stError(C_ba, ra, dec, roll):
    """
    Computes error angles with estimated rotation matrix

    Parameters
    ----------
    C_ba : numpy.ndarray
        Calculated rotation matrix from inertial to body
    ra : float
        True right ascension
    dec : float
        True declination
    roll : float
        True roll

    Returns
    -------
    phi : float
        Rotation angle error associated with error matrix
    ex : float
        Angular error about the x-axis
    ey : float
        Angular error about the y-axis
    ez : float
        Angular error about the z-axis
    """
    
    Cbi_true = radec2M(ra, dec, roll) # true body to inertial 
    C_error = np.matmul(C_ba, Cbi_true.T) 
    # error angle 
    phi = np.rad2deg(np.arccos(0.5 * (np.trace(C_error) - 1)))
    # each axis error 
    ex = (90 - np.rad2deg(np.arccos(C_error[2,1]))) * 3600
    ey = (90 - np.rad2deg(np.arccos(C_error[2,0]))) * 3600
    ez = (90 - np.rad2deg(np.arccos(C_error[1,0]))) * 3600
    return phi, ex, ey, ez

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

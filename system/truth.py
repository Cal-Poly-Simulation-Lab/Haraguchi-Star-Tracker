import numpy as np

def stError(C_ba, ra, dec, roll):
    # C_ba is calculated rotation matrix inertial to body
    # M is true body to inertial
    Cbi_true = radec2M(ra, dec, roll) # true body to inertial 
    C_error = np.matmul(C_ba, Cbi_true.T)
    # phi from de ruiters
    phi = np.rad2deg(np.arccos(0.5 * (np.trace(C_error) - 1)))
    # each axis error from texas star tracker paper\
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
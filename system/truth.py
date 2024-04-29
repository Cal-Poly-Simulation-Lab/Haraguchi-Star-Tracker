import numpy as np

def stError(C_ba, ra, dec):
    # C_ba is calculated rotation matrix inertial to body
    # M is true body to inertial
    u_ICRF = np.array([[np.cos(ra) * np.cos(dec)], [np.sin(ra) * np.cos(dec)], [np.sin(dec)]])
    boresight = np.array([[0], [0], [-1]])
    # rotate boresight into inertial
    b_ICRF = np.matmul(C_ba.T, boresight)
    x_error = np.abs((b_ICRF[0,0] - u_ICRF[0,0]) / u_ICRF[0,0]) * 100
    y_error = np.abs((b_ICRF[1,0] - u_ICRF[1,0]) / u_ICRF[1,0]) * 100
    z_error = np.abs((b_ICRF[2,0] - u_ICRF[2,0]) / u_ICRF[2,0]) * 100
    return x_error, y_error, z_error
    
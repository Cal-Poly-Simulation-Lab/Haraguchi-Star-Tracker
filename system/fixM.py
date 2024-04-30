import numpy as np

def radec2M(alpha, delta, phi):
    M1 = np.array([[np.cos(alpha - np.pi/2), -np.sin(alpha - np.pi/2), 0], 
                   [np.sin(alpha - np.pi/2), np.cos(alpha - np.pi/2), 0], 
                   [0, 0, 1]])
    M2 = np.array([[1, 0, 0], 
                   [0, np.cos(delta - np.pi/2), -np.sin(delta - np.pi/2)], 
                   [0, np.sin(delta - np.pi/2), np.cos(delta - np.pi/2)]])
    M3 = np.array([[np.cos(phi), -np.sin(phi), 0], 
                   [np.sin(phi), np.cos(phi), 0], 
                   [0, 0, 1]])
    Mi = np.matmul(M2, M3)
    C_ICRF_b = np.matmul(M1, Mi)
    return C_ICRF_b.T

ra = np.deg2rad(195) # big dipper
dec = np.deg2rad(55)
roll = np.deg2rad(180)

s_ICRF = np.array([[np.cos(ra) * np.cos(dec)], [np.sin(ra) * np.cos(dec)], [np.sin(dec)]])
C_b_ICRF = radec2M(ra, dec, roll)
s_b = np.matmul(C_b_ICRF, s_ICRF)
print(s_b)


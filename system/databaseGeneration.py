import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def generationCatalog(path, minMag, maxMag):
    """
    Parses bs5_brief.csv file for use in image generation and saves as csv file

    Parameters
    ----------
    path : str
        Path to bs5_brief.csv file location
    minMag : float
        Minimum star magnitude displayable by screen
    maxMag : float
        Maximum star magnitude displayable by screen
    Ls : flat
        Maximum luminance of screen in lm/(m2*sr)
    Ap : float
        Square area of pixel in m
    """

    # calculate displayable magnitudes - fix this stuff!! 
    # do this in a different function and pass them in like I'm doing
    # jpl specs
    Ep_min = 5.7303e-08
    Ep_max = 1.4670e-05
    # all stars in catalog
    # Ep_min = 1.3677e-9
    # Ep_max = 8.0168e-6
    E = np.linspace(Ep_min, Ep_max, 156)
    mag = illuminance2magnitude(E)
    px = np.linspace(150, 255, 156)

    # read in file at path as pandas dataframe
    catalogDF = pd.read_csv(path)
    # drop columns other than ra, dec, and vmag
    catalogDF.drop(catalogDF.columns[[0,1,2,3,4,5,14,15,16,17,18,19,20]], axis=1, inplace = True) # 0 is hr star number, might want to keep it
    # delete rows that contain a null
    catalogDF.dropna(inplace=True)
    # filter by displayable magnitude
    catalogDF.drop(catalogDF[catalogDF.Vmag > minMag].index, inplace=True)
    catalogDF.drop(catalogDF[catalogDF.Vmag < maxMag].index, inplace=True)
    # sort by magnitude in descending order (negative is high magnitude so techinally in ascending)
    catalogDF.sort_values(by=['Vmag'], inplace=True)

    numEntries = len(catalogDF)
    print(numEntries)

    data_arr = np.empty([numEntries,4])

    for i in range(numEntries):
        data_arr[i][0] = hours2rad(catalogDF.iat[i,0], catalogDF.iat[i,1], catalogDF.iat[i,2])
        data_arr[i][1] = degmin2rad(catalogDF.iat[i,4], catalogDF.iat[i,5], catalogDF.iat[i,6], catalogDF.iat[i,3])
        data_arr[i][2] = catalogDF.iat[i,7]
        # find closest value in mag 
        idx = np.argmin(np.abs(mag - data_arr[i][2]))
        data_arr[i][3] = px[idx]

    # save as csv file 
    data = pd.DataFrame(data_arr, columns=['RA', 'DEC', 'VMAG', 'INTEN'])
    data.to_csv('star_generation_data.csv', index=False)

    return

def stDatabase(path, minMag, maxMag):
    """
    Parse bs5_brief.csv file for use in star identification and saves as csv file 

    Parameters
    ----------
    path : str
        Path to bs5_brief.csv file location
    minMag : float
        Minimum star magnitude displayable by screen
    maxMag : float
        Maximum star magnitude displayable by screen
    """

    # read in file at path as pandas dataframe
    catalogDF = pd.read_csv(path)
    # drop columns other than ra, dec, and vmag
    catalogDF.drop(catalogDF.columns[[0,1,2,3,4,5,14,15,16,17,18,19,20]], axis=1, inplace = True) # 0 is hr star number, might want to keep it
    # delete rows that contain a null
    catalogDF.dropna(inplace=True)
    # filter by displayable magnitude
    catalogDF.drop(catalogDF[catalogDF.Vmag > minMag].index, inplace=True)
    catalogDF.drop(catalogDF[catalogDF.Vmag < maxMag].index, inplace=True)
    # sort by magnitude in descending order (negative is high magnitude so techinally in ascending)
    catalogDF.sort_values(by=['Vmag'], inplace=True)

    numEntries = len(catalogDF)
    v_arr = np.empty([numEntries,3]) # ra, dec, v

    for i in range(numEntries):
        ra = hours2rad(catalogDF.iat[i,0], catalogDF.iat[i,1], catalogDF.iat[i,2])
        dec = degmin2rad(catalogDF.iat[i,4], catalogDF.iat[i,5], catalogDF.iat[i,6], catalogDF.iat[i,3])
        v_arr[i] = radec2v(ra, dec)

    # save as csv file
    data = pd.DataFrame(v_arr, columns=['x', 'y', 'z'])
    data.to_csv('v_unit_vectors.csv', index=False)
    
    return

# based on K-vector paper 
def K_vector(path):
    """
    Creates K-Vector 
    """
    v_array = pd.read_csv(path)
    v = v_array.to_numpy()

    cosTheta = np.cos(75 * np.pi / 180) # cos thetaFOV for visibility condition 
    numEntries = len(v)
    # m = numEntries * numEntries - numEntries # number of admissible star pairs - m is actually the number that fit the criteria
    P = [] # P, I, J
    m = 0
    for i in range(numEntries):
        for j in range(i+1, numEntries): # change this to i+1 to numEntries to get rid of j,i duplicates 
            if i != j:
                dot = np.dot(v[i], v[j])
                if dot >= cosTheta:
                    P.append([dot, i, j])
                    m += 1
    P = np.array(P) # convert to numpy array 
    print("P =")
    print(P)

    # sort P to get S 
    S = P[P[:,0].argsort()] # S, I, J
    print("S =")
    print(S)

    plt.plot(S[:,0], linestyle='None', marker='.')
    plt.xlabel("Progressive Index")
    plt.ylabel("S-Vector")
    plt.grid(True)

    plt.show()

    # shifted best fit line
    D = (S[m-1][0] - S[0][0]) / (m - 1)
    a1 = m * D / (m - 1)
    a0 = S[0][0] - a1 - (D / 2)

    # build K-vector
    S_only = S[:,0]
    K = np.zeros([m,1]) # <---- really need to verify this, why are they only even? 
    K[-1] = m # set last element to m 
    for k in range(1,m-1): # K(0) = 0 so can start at 1 (check -1 instead of -2)
        val = a1 * k + a0
        # print("val = " + str(val))
        # idx_less = np.where(S_only <= val)
        # print("idx = " + str(idx_less))
        idx_greater = np.where(S_only > val)
        # print(idx_less[0])
        # # print("more = " + str(idx_greater))
        K[k] = idx_greater[0][0]
        # need to think about what to do here 

    plt.plot(K, S_only, linestyle='None', marker='.')
    plt.xlabel("K-vector")
    plt.ylabel("S-vector")
    plt.grid(True)

    plt.show()

    # # plot cosTheta line 

    # print("K = ")
    # print(K)
    # print("SIJ = ")
    # print(S)
    KSIJ = np.concatenate((K,S), axis=1)
    # print(KSIJ)

    # convert K, S, I, J to csv
    data = pd.DataFrame(KSIJ, columns=['K', 'S', 'I', 'J'])
    data.to_csv('KSIJ_arrays.csv', index=False)

    return a0, a1


def hours2rad(hr, min, sec):
    """
    Converts right ascension hours, minutes, seconds to radians
    
    Parameters
    ----------
    hr : float
        RA hours
    min : float
        RA minutes
    sec : float
        RA seconds

    Returns
    -------
    rad : float
        RA angular value in radians 
    """

    hr += (min / 60)
    hr += (sec / 3600)
    rad = hr * 15 * np.pi / 180 # verify this math that it's ok Johan says it is
    return rad

def degmin2rad(deg, min, sec, dir):
    """
    Converts declination degrees, minutes, seconds to radians

    Parameters
    ----------
    deg : float
        DEC degrees
    min : float
        DEC minutes
    sec : float
        DEC seconds
    dir : str
        DEC direction, N or S

    Returns
    -------
    rad : float
        DEC angular value in radians
    """
    deg += (min / 60)
    deg += (sec / 3600)
    rad = deg * np.pi / 180
    if dir == "N":
        return rad
    elif dir == "S":
        return -1 * rad
    else:
        raise Exception("invalid declination direction")
    
def illuminance2magnitude(E):
    """
    Calculates star magnitude given illuminance.

    Parameters
    ----------
    E : float
        Illuminance in lm/m2 = lux

    Returns
    -------
    m : float
        Magnitude 
    """
    
    m = -14.2 - 2.5 * np.log10(E)
    return m

def cosTheta(a0, a1, k):
    return a1 * k + a0

# converts ra and dec to unit vector in celestial sphere 
# from star id pg 6 
def radec2v(ra, dec):
    """
    Converts right ascension and declination to unit vector in celestial sphere

    Parameters
    ----------
    ra : float
        Right ascension in rad
    dec : float
        Declination in rad

    Returns
    -------
    v : list
        [x, y, z] unit vector 
    """

    x = np.cos(ra) * np.cos(dec)
    y = np.sin(ra) * np.cos(dec)
    z = np.sin(dec)
    norm = np.sqrt(x**2 + y**2 + z**2)
    v = [x, y, z] / norm
    return v
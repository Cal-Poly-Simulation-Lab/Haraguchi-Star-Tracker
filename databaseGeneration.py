import numpy as np
import pandas as pd
import bisect

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
    """

    mag = np.linspace(minMag, maxMag, 156) 
    px = np.linspace(150, 255, 156)

    # read in file at path as pandas dataframe
    catalogDF = pd.read_csv(path)
    # drop columns other than ra, dec, and vmag
    catalogDF.drop(catalogDF.columns[[0,1,2,3,4,5,14,15,16,17,18,19,20]], axis=1, inplace = True)
    # delete rows that contain a null
    catalogDF.dropna(inplace=True)
    # filter by displayable magnitude
    catalogDF.drop(catalogDF[catalogDF.Vmag > minMag].index, inplace=True)
    catalogDF.drop(catalogDF[catalogDF.Vmag < maxMag].index, inplace=True)
    # sort by magnitude in descending order (negative is high magnitude so techinally in ascending)
    catalogDF.sort_values(by=['Vmag'], inplace=True)

    numEntries = len(catalogDF)
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
    Creates K-Vector for rapid searching

    Parameters
    ----------
    path : str
        Contains path to star unit vectors file 
    """

    v_array = pd.read_csv(path)
    v = v_array.to_numpy()

    cosTheta = np.cos(75 * np.pi / 180) # cos thetaFOV for visibility condition 
    numEntries = len(v)
    PIJ = [] # P, I, J
    m = 0
    for i in range(numEntries):
        for j in range(i+1, numEntries): 
            if i != j:
                dot = np.dot(v[i], v[j])
                if dot >= cosTheta:
                    PIJ.append([dot, i, j])
                    m += 1
    PIJ = np.array(PIJ) # convert to numpy array 

    # sort P to get S 
    SIJ = PIJ[PIJ[:,0].argsort()] # S, I, J
    S = SIJ[:,0]

    # parameters of modified linear approximation
    D = (S[-1] - S[0]) / (m - 1)
    a0 = S[0] - D/2 # intercept
    a1 = (S[m-1] - S[0] + D) / (m - 1) # slope

    # build K-vector
    K = []
    for k in range(m):
        val = a1 * k + a0
        num = bisect.bisect_left(S, val)
        K.append([num])

    KSIJ = np.concatenate((K,SIJ), axis=1)
    a_values = np.array([[a0], [a1]])

    # convert K, S, I, J to csv
    data = pd.DataFrame(KSIJ, columns=['K', 'S', 'I', 'J'])
    data.to_csv('KSIJ_arrays.csv', index=False)
    values = pd.DataFrame(a_values, columns=['a'])
    values.to_csv('a_values.csv', index=False)

    return
    
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
    rad = hr * 15 * np.pi / 180
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

def cosTheta(a0, a1, k):
    """
    Computes value of cosTheta line in K-vector creation

    Parameters
    ----------
    a0 : float
        Slope of cosTheta line
    a1 : float
        y-intercept of cosTheta line
    k : float
        Value of interest

    Returns
    -------
    float
        Value of cosTheta line at k
    """

    return a1 * k + a0

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

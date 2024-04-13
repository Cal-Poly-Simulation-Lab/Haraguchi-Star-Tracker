import numpy as np
import pandas as pd

def parseCatalog(path, minMag, maxMag):
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

    data_arr = np.empty([numEntries,3])

    for i in range(numEntries):
        data_arr[i][0] = hours2rad(catalogDF.iat[i,0], catalogDF.iat[i,1], catalogDF.iat[i,2])
        data_arr[i][1] = degmin2rad(catalogDF.iat[i,4], catalogDF.iat[i,5], catalogDF.iat[i,6], catalogDF.iat[i,3])
        data_arr[i][2] = catalogDF.iat[i,7]

    # save as csv file 
    data = pd.DataFrame(data_arr, columns=['RA', 'DEC', 'VMAG'])
    data.to_csv('star_generation_data.csv', index=False)

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

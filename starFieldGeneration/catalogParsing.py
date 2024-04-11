import csv
import numpy as np
from timeit import default_timer

def parseCatalog(path, num_entries):

    data_arr = np.empty([num_entries, 3]) # create empty array to hold catalog data 
    count = 0 # for use in indexing into array 

    with open(path, mode='r') as file: # open file at provided path in read mode 
        csvFile = csv.DictReader(file) # reading as dictionary makes file headers dict keys
       
        for lines in csvFile: # iterate through each line

            # read in RA values and convert to rad 
            data_arr[count, 0] = hours2rad(lines.get("RA"), lines.get("m"), lines.get("s"))
            
            # read in DEC values and convert to rad with proper sign 
            udec = minsec2rad(lines.get("Dec deg"), lines.get("Dec m"), lines.get("Dec s"))
            if lines.get("Dec dir") == "N":
                data_arr[count, 1] = udec
            elif lines.get("Dec dir") == "S":
                data_arr[count, 1] = -1 * udec
            else:
                raise Exception("Declination direction not specified")
            
            # read in visual magnitude values
            data_arr[count, 2] = float(lines.get("Vmag"))
            count += 1 # increment counter

    # sort contents of data_arr by mag column values in descending order (negative means high magnitude)
    data_arr = data_arr[data_arr[:, 2].argsort()]
    print("highest magnitude = " + str(data_arr[0,2]))
    print("lowest magnitude = " + str(data_arr[-1,2]))
    print(data_arr)
    return data_arr

def hours2rad(hr, min, sec):
    """Converts ra or dec string to equivalent value in radians
    
    Parameters
    ----------
    hrs_str : str
        Angular value in HH:MM:SS as output from star catalog

    Returns
    -------
    rad : double
        Angular value in radians 
    """

    hours = float(hr)
    minutes = float(min)
    sec = float(sec)
    hours += (minutes / 60)
    hours += (sec / 3600)
    rad = hours * 15 * np.pi / 180 # verify this math that it's ok Johan says it is
    return rad

def minsec2rad(deg, min, sec):
    deg = float(deg)
    min = float(min)
    sec = float(sec)
    deg += (min / 60)
    deg += (sec / 3600)
    rad = deg * np.pi / 180
    return rad

# ra = np.empty((9096, 1))
# dec = np.empty((9096,1))
# mag = np.empty((9096, 1))
# parseCatalog("bs5_brief.csv", 9096)
# print(ra)
# print(dec)
# print(mag)

# timing stuff I should remove 
# total = 0
# for i in range(100):
#     start = default_timer()
#     parseCatalog("bs5_brief.csv", 9096)
#     stop = default_timer()
#     total += stop - start
# total = total / 100
# print(total)
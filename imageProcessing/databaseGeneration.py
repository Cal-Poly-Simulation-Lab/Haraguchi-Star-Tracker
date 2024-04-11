import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# converts hours to rad for use in ra calcs
def hours2rad(hr, min, sec):
    hr += (min / 60)
    hr += (sec / 3600)
    rad = hr * 15 * np.pi / 180 # verify this math that it's ok Johan says it is
    return rad

# converts deg min to rad for use in dec calcs
def degmin2rad(deg, min, sec, dir):
    deg += (min / 60)
    deg += (sec / 3600)
    rad = deg * np.pi / 180
    if dir == "N":
        return rad
    elif dir == "S":
        return -1 * rad
    else:
        raise Exception("invalid declination direction")
    
# converts ra and dec to unit vector in celestial sphere 
# from star id pg 6 
def radec2v(ra, dec):
    x = np.cos(ra) * np.cos(dec)
    y = np.sin(ra) * np.cos(dec)
    z = np.sin(dec)
    norm = np.sqrt(x**2 + y**2 + z**2)
    v = [x[0]/norm[0], y[0]/norm[0], z[0]/norm[0]] # see if there's a better way to do this 
    return v

# read in csv file as pandas dataframe
catalogDF = pd.read_csv('bs5_brief.csv')
print(catalogDF.info())

# drop columns that are not relevant
catalogDF.drop(catalogDF.columns[[1,2,3,4,5,14,15,16,17,18,19,20]], axis=1, inplace = True)
print(catalogDF.info())

# delete rows where one of the data rows is null
catalogDF.dropna(inplace = True)
print(catalogDF.info())

# filter out rows with magnitudes not detectable 
# preliminary calcs, can detect magnitudes -1.5829 to 4.4377
catalogDF.drop(catalogDF[catalogDF.Vmag > 4.4377].index, inplace=True)

# convert columns to numpy arrays
catalog = catalogDF.to_numpy()

numEntries = len(catalogDF)
print(catalogDF)

# store caclulated data 
ra = np.empty([numEntries,1])
dec = np.empty([numEntries,1])
v = np.empty([numEntries,3])

for i in range(numEntries):
    ra[i] = hours2rad(catalog[i][1], catalog[i][2], catalog[i][3])
    dec[i] = degmin2rad(catalog[i][5], catalog[i][6], catalog[i][7], catalog[i][4])
    v[i] = radec2v(ra[i], dec[i])

# from k-vector paper 

cosTheta = np.cos(75 * np.pi / 180) # cos thetaFOV for visibility condition 

m = numEntries * numEntries - numEntries # number of admissible star pairs 
# P = np.empty([m,3]) # columns are P, I, J, respectively 
P = []
count = 0
for i in range(numEntries):
    for j in range(numEntries):
        if i != j:
            dot = np.dot(v[i], v[j])
            if dot >= cosTheta:
                P.append([dot, i, j])
                # P[count,0] = dot
                # P[count,1] = i
                # P[count,2] = j
                count += 1
P = np.array(P)
print("P =")
print(P)

# sort P to get S 
S = P[P[:,0].argsort()]
print("S =")
print(S)

plt.plot(S[:,0], linestyle='None' ,marker=',')
plt.xlabel("progressive index")
plt.ylabel("S-vector")
plt.grid(True)
plt.show()

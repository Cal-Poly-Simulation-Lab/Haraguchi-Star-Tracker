import numpy as np

# attitude parameters 
ra0 = np.deg2rad(195) # big dipper
dec0 = np.deg2rad(55)
roll0 = np.deg2rad(180)
# ra0 = np.deg2rad(249.2104) # orion
# dec0 = np.deg2rad(-12.0386)
# roll0 = np.deg2rad(13.3845)
# ra0 = np.deg2rad(101.29) # sirius
# dec0 = np.deg2rad(-16.716111)
# roll0 = np.deg2rad(0)

# physical system parameters 
fovx = np.deg2rad(66)
fovy = np.deg2rad(41)
h = 600 # pixels, screen height
w = 1024 # pixels, screen width

h_cm = 8.988617 # screen height in cm
l = 13.06 # camera / screen separation in cm - should it be lens / screen?
f = h / h_cm * l # system focal length in pixels

# minMag = 4.4377 # minimum star magnitude displayable by screen
# maxMag = -1.5829 # maximum displayable magnitude 
# using jpl specs for now
minMag = 3.9045
maxMag = -2.1161
# all stars in catalog
# minMag = 7.96
# maxMag = -1.46

# catalog parameters
catalogPath = "bs5_brief.csv" # path to catalog from where code will be run
dataPath = "star_generation_data.csv" # path to csv file generated from catalog
intensityPath = "magnitude2pixel.csv" # path to csv file with magnitude and pixel intensity data 

# generation parameters
maxStars = 50 # maximum number of stars to display in image
starSize = 5
sigma = 1.2 # for gaussian spread of stars 
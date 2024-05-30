import numpy as np

# set to true to regenerate databases
regenerateDatabase = False

# physical system parameters ---------------------------------------------------
fovx = np.deg2rad(66) # horizontal fov in rads
fovy = np.deg2rad(41) # vertical fov in rads 
h = 600 # screen height in pixels
w = 1024 # screen width in pixels 
h_cm = 8.7 # screen height in cm
l = 13.47 # camera / screen separation in cm - should it be lens / screen?
f_sys = h / h_cm * l # system focal length in pixels\
f_cam = 1946.51409462 # calculated based on sizing routine

# file locations ---------------------------------------------------------------
catalogFile = "bs5_brief.csv"
generationDataFile = "star_generation_data.csv"
stDatabaseFile = "v_unit_vectors.csv"
stLookupFile = "KSIJ_arrays.csv"

# operational parameters -------------------------------------------------------
minMag = 3.9045 # minimum star magnitude displayable by screen
maxMag = -2.1161 # maximum displayable magnitude 
starSize = 5 # star spot size in pixels (nxn)
sigma = 1.2 # standard deviation for gaussian spread of stars 
minAreaSoftware = 5 # minimum pixel area to consider as star
maxAreaSoftware = 25 # maximum pixel area
minAreaHardware = 50 # minimum pixel area when running with hardware
maxAreaHardware = 200 # maximum pixel area

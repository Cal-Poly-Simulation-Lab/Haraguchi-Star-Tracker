import numpy as np

regenerateDatabase = False

# physical system parameters ---------------------------------------------------
fovx = np.deg2rad(66) # horizontal fov in rads
fovy = np.deg2rad(41) # vertical fov in rads 
h = 600 # screen height in pixels
w = 1024 # screen width in pixels 
h_cm = 8.7 # 8.988617 # 8.7 # (real) screen height in cm
l = 13.06 # camera / screen separation in cm - should it be lens / screen?
f = h / h_cm * l # system focal length in pixels

# database locations -----------------------------------------------------------
catalogFile = "bs5_brief.csv"
generationDataFile = "star_generation_data.csv"
stDatabaseFile = "v_unit_vectors.csv"
stLookupFile = "KSIJ_arrays.csv"

# operational parameters -------------------------------------------------------
# minMag = 4.4377 # minimum star magnitude displayable by screen
# maxMag = -1.5829 # maximum displayable magnitude 
# using jpl specs for now
minMag = 3.9045 
maxMag = -2.1161
# all stars in catalog
# minMag = 7.96
# maxMag = -1.46
maxStars = 50 # maximum number of stars to display in image - do I still want this?? 
starSize = 5 # star spot size in pixels (nxn)
sigma = 1.2 # standard deviation for gaussian spread of stars 
minArea = 5 # minimum pixel area to consider as star
maxArea = 25 # maximum pixel area to consider as star

# attitude ---------------------------------------------------------------------
# ra0 = np.deg2rad(195) # big dipper
# dec0 = np.deg2rad(55)
# roll0 = np.deg2rad(180)
ra0 = np.deg2rad(np.random.random_sample() * 360)
dec0 = np.deg2rad(180 * np.random.random_sample() - 90)
roll0 = np.deg2rad(np.random.random_sample() * 360)
# ra0 = 5.50703331947639
# dec0 = -0.5024724211135204
# roll0 = 3.902365232373933

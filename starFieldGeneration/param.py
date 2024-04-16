import numpy as np

# attitude parameters 
q_ECI_st = np.array([[0], [0], [0], [1]]) # st frame to ECI
u_st_st = np.array([[1], [0], [0]]) # boresight direction of st in st frame 
# value of u_st_st affects conversion to u,v coordinates 

# physical system parameters 
fov = 40 # degrees, smaller dimension field of view
h = 600 # pixels, screen height
w = 1024 # pixels, screen width
h_cm = 12 # cm, image height on screen
l = 5 * 2.54 # cm, separation between camera and screen
f = h / h_cm * l # pixels, system focal length
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

import numpy as np

# attitude parameters 
q_ECI_b = np.array([[0], [0], [0], [1]]) # body to ECI
q_b_st = np.array([[0], [0], [0], [1]]) # star tracker to body
u_st_st = np.array([[1], [0], [0]]) # boresight direction of st in st frame 
# value of u_st_st affects conversion to u,v coordinates 

# physical system parameters 
fov = 40 # degrees, smaller dimension field of view
h = 600 # pixels, screen height
w = 1024 # pixels, screen width
h_cm = 12 # cm, image height on screen
l = 5 * 2.54 # cm, separation between camera and screen
f = h / h_cm * l # pixels, system focal length
minMag = 4.4377 # minimum star magnitude displayable by screen
maxMag = -1.5829 # maximum displayable magnitude 

# image generation parameters
num_stars = 50 # number of stars to include in image - don't think I want to have this actually 

# catalog parameters
catalogPath = "bs5_brief.csv" # path to catalog from where code will be run
dataPath = "star_generation_data.csv" # path to csv file generated from catalog
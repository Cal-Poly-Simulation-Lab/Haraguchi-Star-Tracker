import numpy as np
import pandas as pd

FOVx = np.deg2rad(66) # rad
FOVy = np.deg2rad(41) # rad
row = 600 # pixels
col = 1024 # pixels

# focal length calculated from FOV
fx = 1 / np.tan(FOVx / 2)
fy = 1 / np.tan(FOVy / 2)

data = pd.read_csv("star_generation_data.csv")
data_all = data.to_numpy()


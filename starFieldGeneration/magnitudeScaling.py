import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def mag(E):
    return -14.2 - 2.5 * np.log10(E)

# screen parameters
Ap = 2.2443e-8 # square area of one pixel in m
Ip_max = 400 * Ap # using JPL screen luminance
Ip_min = 400 / (2**8) * Ap

E = np.linspace(Ip_min, Ip_max, 256)
m = mag(E)
px = np.linspace(0, 255, 256)

plt.plot(px, m)
plt.grid(True)
plt.xlabel("pixel intensity")
plt.ylabel("magnitude")
plt.show()

data = pd.DataFrame(data=[m, px]).T
data.to_csv("magnitude2pixel.csv", index=False, header=None)

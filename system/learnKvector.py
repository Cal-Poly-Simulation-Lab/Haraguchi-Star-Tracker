import numpy as np
from matplotlib import pyplot as plt
import bisect
from systemParameters import *

def getUnitVector(X, Y, f):
    s = np.array([X, Y, f]) # as row vector for appending to list 
    s = 1 / np.sqrt(X**2 + Y**2 + f**2) * s
    return s

# vectors for testing ----------------------------------------------------------
local_unit = np.array([[-0.05183503, -0.03149954,  0.99815876],
                       [ 0.28189683, -0.02969848,  0.95898497],
                       [ 0.0204207,  -0.00789933,  0.99976027],
                       [-0.16071304,  0.00829916,  0.98696628],
                       [0.11247896, 0.01407085, 0.99355448],
                       [0.27750887, 0.06396917, 0.95859104],
                       [0.14417867, 0.08614454, 0.98579492]])
global_unit = np.array([[-0.54429276, -0.1307443, 0.82864427],
                        [-0.45911157, 0.11504759, 0.88089762],
                        [-0.58145959, -0.29479991, 0.75828607],
                        [-0.53654398, -0.20575851, 0.81840332],
                        [-0.5359151, 0.13899214, 0.83275218],
                        [-0.59187286, 0.01593808, 0.80587375],
                        [-0.54293093, -0.03660081, 0.83897937]])

# generate P-vector ------------------------------------------------------------
n = len(global_unit)

PIJ = []

m = 0
for i in range(n):
    for j in range(i+1,n):
        if i != j:
            vi = np.atleast_2d(global_unit[i,:]).T
            vj = np.atleast_2d(global_unit[j,:]).T
            dot = np.matmul(vi.T, vj)[0,0]
            PIJ.append([dot, i, j])
            m += 1

PIJ = np.array(PIJ)

# sort for S-vector ------------------------------------------------------------
SIJ = PIJ[PIJ[:,0].argsort()]
S = SIJ[:,0]

idx = np.linspace(0, 20, 21)
plt.plot(idx, S, '.')
plt.xlabel("progressive index")
plt.ylabel("S-vector")
plt.grid(True)

# line connecting extreme points
x0 = [0, m-1]
y0 = [S[0], S[m-1]]
plt.plot(x0, y0)

# create K-vector --------------------------------------------------------------
D = (S[-1] - S[0]) / (m - 1)
a0 = S[0] - D/2 # intercept
a1 = (S[m-1] - S[0] + D) / (m - 1) # slope

# steeper line
x1 = [0, m-1]
y1 = [a0, a1 * (m-1) + a0]
plt.plot(x1, y1)

K = []
for k in range(m):
    val = a1 * k + a0
    num = bisect.bisect_left(S, val)
    plt.axhline(val)
    K.append(num)

# indexing into K --------------------------------------------------------------
print(K)
print(SIJ)

# test cosines
cosl = 0.99
cosh = 1

l_bot = int(np.floor((cosl - a0) / a1))
l_top = int(np.ceil((cosh - a0) / a1)) - 1
plt.plot(l_bot, cosl, '.')
plt.plot(l_top, cosh, '.')
print(l_bot)
print(l_top)

k_start = K[l_bot]
k_end = K[l_top] - 1
print(k_start)
print(k_end)

possible = 0
for k in range(k_start, k_end+1):
    i = SIJ[k,1]
    j = SIJ[k,2]
    print("possible star pair " + str(i) + ", " + str(j))
    possible += 1

print(str(possible) + " possible star pairs")
plt.show()

# angle error calculation
s1 = np.atleast_2d(getUnitVector(0.5, 0.5, f)).T
s2 = np.atleast_2d(getUnitVector(1, 1, f)).T
epsilon = np.arccos(np.matmul(s1.T, s2))
print("angle error " + str(epsilon) + " rad")

import numpy as np
from matplotlib import pyplot as plt
import bisect
from systemParameters import *
import tikzplotlib as tp

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

def getUnitVector(X, Y, f):
    s = np.array([X, Y, f]) # as row vector for appending to list 
    s = 1 / np.sqrt(X**2 + Y**2 + f**2) * s
    return s

# vectors for testing ----------------------------------------------------------
global_unit = np.array([[-0.54429276, -0.1307443, 0.82864427],
                        [-0.45911157, 0.11504759, 0.88089762],
                        [-0.58145959, -0.29479991, 0.75828607],
                        [-0.53654398, -0.20575851, 0.81840332],
                        [-0.5359151, 0.13899214, 0.83275218],
                        [-0.59187286, 0.01593808, 0.80587375],
                        [-0.54293093, -0.03660081, 0.83897937]])

# global_unit = np.array([[-0.54429276, -0.1307443, 0.82864427], 
#                         [-0.45911157, 0.11504759, 0.88089762],
#                         [-0.58145959, -0.29479991, 0.75828607],
#                         [-0.53654398, -0.20575851, 0.81840332],
#                         [-0.5359151, 0.13899214, 0.83275218],
#                         [-0.59187286, 0.01593808, 0.80587375],
#                         [-0.76124922, -0.18990071, 0.62003011],
#                         [-0.69613262, 0.15540211, 0.70089197],
#                         [-0.61817602, -0.48331239, 0.61989317],
#                         [-0.32281449, -0.40196695, 0.85686252],
#                         [-0.54293093, -0.03660081, 0.83897937],
#                         [-0.65977965, 0.31789696, 0.68090552],
#                         [-0.53397053, -0.54311949, 0.64799436],
#                         [-0.6802629, -0.5306675, 0.505603 ],
#                         [-0.37031785, -0.2233643, 0.9016502 ],
#                         [-0.36124153, 0.27338244, 0.89149683],
#                         [-0.67074186, 0.04087745, 0.74056356],
#                         [-0.6107742, 0.51776805, 0.59905854],
#                         [-0.35022272, 0.04392777, 0.93563583],
#                         [-0.34180864, -0.05029613, 0.93842269]])

# global_unit = np.array([[-0.78378556, -0.52698821, 0.32857819],
#                         [-0.54429276, -0.1307443, 0.82864427],
#                         [-0.45911157, 0.11504759, 0.88089762],
#                         [-0.58145959, -0.29479991, 0.75828607],
#                         [-0.45138849, -0.77792958, 0.43712013],
#                         [-0.20072794, -0.18507241, 0.9620065 ],
#                         [-0.52916997, -0.71964245, 0.44954854],
#                         [-0.53654398, -0.20575851, 0.81840332],
#                         [-0.5359151, 0.13899214, 0.83275218],
#                         [-0.59187286, 0.01593808, 0.80587375], 
#                         [-0.91781488, 0.18627979, 0.35059335],
#                         [-0.85245493, 0.39763457, 0.33942206],
#                         [-0.83254224, -0.45526074, 0.31561223],
#                         [-0.66948593, -0.58705451, 0.45514348],
#                         [-0.19400541, -0.43570064, 0.87893507],
#                         [-0.35525517, -0.85999589, 0.36633432],
#                         [-0.07867132, -0.60642616, 0.79123836],
#                         [-0.28680045, -0.80195802, 0.52403133],
#                         [-0.76124922, -0.18990071, 0.62003011],
#                         [-0.76279647, 0.50559426, 0.40313272],
#                         [-0.69613262, 0.15540211, 0.70089197],
#                         [-0.61817602, -0.48331239, 0.61989317],
#                         [-0.6819708, 0.30961285, 0.66261279],
#                         [-0.1996438, -0.23946763, 0.95015662],
#                         [-0.63456524, 0.52750477, 0.56485899],
#                         [-0.17691964, -0.89007781, 0.42007253],
#                         [-0.47112519, 0.47439819, 0.74363123],
#                         [-0.15603634, -0.78528333, 0.5991517 ],
#                         [-0.09114183, -0.40105428, 0.91150899],
#                         [-0.49661812, 0.37132634, 0.78452992],
#                         [-0.32281449, -0.40196695, 0.85686252],
#                         [-0.54293093, -0.03660081, 0.83897937],
#                         [-0.29819833, 0.38769062, 0.87222345], 
#                         [-0.82597242, 0.39977986, 0.39742374], 
#                         [-0.65977965, 0.31789696, 0.68090552],
#                         [-0.54961188, -0.62949574, 0.54923756],
#                         [-0.82406255, 0.15095265, 0.54601667],
#                         [-0.53397053, -0.54311949, 0.64799436],
#                         [-0.84411738, 0.45197968, 0.28840981],
#                         [-0.25683031, -0.73438509, 0.62826485],
#                         [-0.6802629, -0.5306675, 0.505603 ],
#                         [-0.48837465, 0.47316467, 0.73321579], 
#                         [-0.37031785, -0.2233643, 0.9016502 ],
#                         [-0.36124153, 0.27338244, 0.89149683],
#                         [-0.53843297, -0.68809977, 0.48642434],
#                         [-0.67074186, 0.04087745, 0.74056356],
#                         [-0.39203814, -0.85945403, 0.32809278],
#                         [-0.85246957, 0.39761876, 0.33940382],
#                         [-0.43508143, 0.27454248, 0.85751419],
#                         [-0.6107742, 0.51776805, 0.59905854],
#                         [-0.79217109, 0.23724206, 0.56229989],
#                         [-0.35022272, 0.04392777, 0.93563583],
#                         [-0.50539593, -0.74048936, 0.44300165],
#                         [-0.34180864, -0.05029613, 0.93842269],
#                         [-0.76375755, 0.47371893, 0.43848008],
#                         [-0.2926184, -0.62566805, 0.7231279 ]])

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
            if dot >= np.cos(75 * np.pi / 180):
                PIJ.append([dot, i, j])
                m += 1

PIJ = np.array(PIJ)

# sort for S-vector ------------------------------------------------------------
SIJ = PIJ[PIJ[:,0].argsort()]
S = SIJ[:,0]

idx = np.linspace(0, m-1, m)
fig = plt.figure()
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

plt.legend(('dot products','linear approximation', 'modified approximation'))
# plt.show()
tikzplotlib_fix_ncols(fig)
tp.clean_figure()
tp.save("s_vector.tex")

K = []
for k in range(m):
    val = a1 * k + a0
    num = bisect.bisect_left(S, val)
    # plt.axhline(val)
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
print("angle error " + str(epsilon * 180 / np.pi) + " deg")

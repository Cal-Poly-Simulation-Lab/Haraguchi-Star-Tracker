from param import *
from catalogParsing import * 
import catalogParsing as cp
import staticImage as si

# only happens once or if parameters have changed 
cp.parseCatalog(catalogPath, minMag, maxMag)

si.static_image(dataPath, q_ECI_b, q_b_st, u_st_st, fov, f, h, w, num_stars)

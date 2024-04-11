from param import *
from catalogParsing import * 
import catalogParsing as cp
import staticImage as si

# do this outside function so it only needs to happen once 
data_all = cp.parseCatalog(path, num_entries)
si.static_image(data_all, q_ECI_b, q_b_st, u_st_st, fov, f, h, w, num_stars)

import cv2 as cv
import numpy as np

def static_image(data_all, q_ECI_b, q_b_st, u_st_st, fov, f, h, w, num_stars): # ra, dec, box, num_stars, r, c, C_camECI):
    """Generates a star field image in a given location
    
    Parameters
    ----------
    ra : str
        Right ascencion of center of image in J2000 coordinates
        Formatted HH:MM:SS
    dec : str
        Decination of center of image in J2000 coordinates
        Formatted HH:MM:SS
    box : int
        Size of box to search for stars in degrees
    num_stars : int
        Number of top brightest stars to seach catalog for
    r : int
        Pixel rows in the output image
    c : int
        Pixel columns in the output image

    Returns
    -------
    displays image
    """

    # convert st boresight direction to celestial ra and dec 
    u_st_b = rot(u_st_st, q_b_st) # st boresight in body frame
    u_st_ECI = rot(u_st_b, q_ECI_b) # st boresight in ECI
    (alpha, delta) = r2radec(u_st_ECI) # ECI vector to RA and DEC
    print("looking at sky at RA = %f rad, DEC = %f rad" %(alpha, delta))

    # find range of star ra/dec values that will be in view 
    fov = fov * np.pi / 180 # convert fov to rad

    fov_type = "rectangular" # book or rectangular
    if fov_type == "book":
        R = np.sqrt(2 * fov**2) / 2 # radius of circular field of view from star id book 
        ra_min = alpha - R / np.cos(delta) # min/max ra 
        ra_max = alpha + R / np.cos(delta)
        dec_min = delta - R

        if dec_min < -1 * np.pi / 2: # limit dec to +/-pi/2
            dec_min = -1 * np.pi / 2
        dec_max = delta + R
        if dec_max > np.pi / 2:
            dec_max = np.pi / 2
    elif fov_type == "rectangular": # spread it out the right amount but seems to wide or something 
        R_ra = np.sqrt(2 * (66 * np.pi / 180)**2) / 2 # 66 * np.pi / 180 / 2
        R_dec = np.sqrt(2 * (41 * np.pi / 180)**2) # 41 * np.pi / 180 / 2
        ra_min = alpha - R_ra / np.cos(delta) # min/max ra 
        ra_max = alpha + R_ra / np.cos(delta)
        dec_min = delta - R_dec

        if dec_min < -1 * np.pi / 2: # limit dec to +/-pi/2
            dec_min = -1 * np.pi / 2
        dec_max = delta + R_dec
        if dec_max > np.pi / 2:
            dec_max = np.pi / 2

    print("ra min and max %f, %f dec min and max %f, %f" %(ra_min, ra_max, dec_min, dec_max))
    
    # get lists of all stars in view 
    num_entries = data_all.shape[0] # number of entries in catalog arrays
    ra = []
    dec = []
    mag = []
    for i in range(num_entries):
        in_view = False
        if data_all[i,1] > dec_min and data_all[i,1] < dec_max:
            if ra_max > (2 * np.pi): # explain this! 
                if data_all[i,0] > ra_min or data_all[i,0] < (ra_max - 2 * np.pi):
                    # print("star %d found! from max too big" %i)
                    in_view = True
            elif ra_min < 0:
                if data_all[i,0] < ra_max or data_all[i,0] > (ra_min + 2 * np.pi):
                    # print("star found! from min too small")
                    in_view = True
            elif data_all[i,0] > ra_min or data_all[i,0] < ra_max:
                # print("star found! from regular conditions")
                in_view = True
        if in_view:
            ra.append(data_all[i,0])
            dec.append(data_all[i,1])
            mag.append(data_all[i,2])
 
    # goofy normalizing mag stuff that definitely needs to be changed
    # max_mag = max(mag)
    
    # create image array 
    img = np.zeros((h,w), np.uint8)

    # convert ra,dec -> ECI -> body -> star tracker -> focal plane (u,v)
    stars_added = 0
    for i in range(num_stars):
        rai = ra[i]
        deci = dec[i]
        u_star_ECI = np.array([[np.cos(deci) * np.cos(rai)],
                               [np.cos(deci) * np.sin(rai)],
                               [np.sin(deci)]])
        u_star_b = rot(u_star_ECI, q_star(q_ECI_b))
        u_star_st = rot(u_star_b, q_star(q_b_st))
        u = f * u_star_st[1,0] / u_star_st[0,0] # have this be a condition depending on boresight direction
        v = f * u_star_st[2,0] / u_star_st[0,0]
        r = int(round(v) + (h / 2))
        c = int(round(u) + (w / 2))
        if 0 <= r-1 and r+1 < h and 0 <= c-1 and c+1 < w:
            # print("point added at (%d, %d)" %(r,c))
            img[r,c] = 255 # * (2.512**(-1 * mag[i])) / (2.512**1.46)
            img[r,c+1] = 255
            img[r,c-1] = 255
            img[r+1,c] = 255
            img[r+1,c+1] = 255
            img[r+1,c-1] = 255
            img[r-1,c] = 255
            img[r-1,c+1] = 255
            img[r-1,c-1] = 255
            stars_added += 1
            if stars_added >= num_stars: # limit stars included in the image 
                break 

    cv.imshow("static image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# def dynamic_image():
#     cap = cv.VideoCapture("input.mp4")
#     output = cv.VideoWriter( 
#         "output.avi", cv.VideoWriter_fourcc(*'MPEG'), 30, (1080, 1920)) 
  
#     while(True): 
#         ret, frame = cap.read() 
#         if(ret): 
              
#             # adding rectangle on each frame 
#             cv.rectangle(frame, (100, 100), (500, 500), (0, 255, 0), 3) 
              
#             # writing the new frame in output 
#             output.write(frame) 
#             cv.imshow("output", frame) 
#             if cv.waitKey(1) & 0xFF == ord('s'): 
#                 break
  
#     cv.destroyAllWindows() 
#     output.release() 
#     cap.release()

def q_star(q):
    q_star = np.array([[-1 * q[0,0]], [-1 * q[1,0]], [-1 * q[2,0]], [q[3,0]]])
    return q_star

def q_mult_cross(p, q):
    q_ret = np.array([p[0] * q[3] - p[1] * q[2] + p[2] * q[1] + p[3] * q[0],
                      p[0] * q[2] - p[2] * q[0] + p[1] * q[3] + p[3] * q[1],
                      p[1] * q[0] - p[0] * q[1] + p[2] * q[3] + p[3] * q[2],
                      p[3] * q[3] - np.vdot(p[0:3], q[0:3])])
    return q_ret

def rot(v, q):
    q_s = q_star(q)
    v_q = np.array([v[0], v[1], v[2], [0]])
    v_rot = q_mult_cross(q_mult_cross(q, v_q), q_s)
    v_rot = v_rot[0:3]
    return v_rot

def r2radec(r):
    u = r / np.linalg.norm(r)
    l = u[0]
    m = u[1]
    n = u[2]
    dec = np.arcsin(n)
    ra = np.arccos(l / np.cos(dec))
    if m <= 0:
        ra = 2 * np.pi - ra
    return ra[0], dec[0]


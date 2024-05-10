import cv2 as cv
import numpy as np
from systemParameters import * 
import databaseGeneration as dg
import imageGeneration as ig
import starTrackerTest as st
import truth
from picamera2 import Picamera2, Preview
import os

# attitude ---------------------------------------------------------------------
# ra0 = np.deg2rad(np.random. random_sample() * 360)
# dec0 = np.deg2rad(180 * np.random.random_sample() - 90)
# roll0 = np.deg2rad(np.random.random_sample() * 360)

# database generation ----------------------------------------------------------
if regenerateDatabase:
    dg.generationCatalog(catalogFile, minMag, maxMag)
    dg.stDatabase(catalogFile, minMag, maxMag)
    dg.K_vector(stDatabaseFile)

# image generation -------------------------------------------------------------
newImage = False
if newImage:
    img = ig.staticImage(generationDataFile, ra0, dec0, roll0, fovx, fovy, f, h, w, 
                        maxStars, starSize, sigma)
    cv.imwrite("display_img.png", img)
# print(ra0)
# print(dec0)
# print(roll0)

# star tracker -----------------------------------------------------------------
takeImage = False                                                                          
if takeImage:
    os.system("chmod 0700 /run/user/1000/")
    os.environ["LIBCAMERA_LOG_LEVELS"] = "3"
    picam = Picamera2()
    config = picam.create_preview_configuration(main={"size": (2304,1296)})
    picam.configure(config)
    picam.start_preview(Preview.QTGL)
    picam.start()             
    input("waiting for alignment")
    picam.capture_file("st_image.png")
    picam.close()
img = cv.imread("st_image.png", 0)
res = st.stOperation(img, minArea, maxArea, 1296, 2304, f)

# # truth ------------------------------------------------------------------------
if res == 0:
    print("identification failed")
else:
    q = res[0]
    C = res[1]
    errorAngle, errorX, errorY, errorZ = truth.stError(C, ra0, dec0, roll0)
    if errorAngle > 0.1:
        print("error greater than 0.1 = " + str(errorAngle))
    print("error angle = " + str(errorAngle) + " degrees")
    # print("error about x, y, z axis = " + str(errorX) + ", " + str(errorY) + ", " + str(errorZ) + " arcseconds")


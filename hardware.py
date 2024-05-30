import cv2 as cv
import numpy as np
from systemParameters import * 
import databaseGeneration as dg
import imageGeneration as ig
import starTracker as st
import truth
from picamera2 import Picamera2
import os

failed = 0
results = []
numTrials = 100

for i in range(numTrials):
    # attitude -----------------------------------------------------------------
    ra0 = np.deg2rad(np.random. random_sample() * 360)
    dec0 = np.deg2rad(180 * np.random.random_sample() - 90)
    roll0 = np.deg2rad(np.random.random_sample() * 360)

    # database generation ------------------------------------------------------
    if regenerateDatabase:
        dg.generationCatalog(catalogFile, minMag, maxMag)
        dg.stDatabase(catalogFile, minMag, maxMag)
        dg.K_vector(stDatabaseFile)

    # image generation ---------------------------------------------------------
    newImage = True
    if newImage:
        img = ig.staticImage(generationDataFile, ra0, dec0, roll0, fovx, fovy, 
                             f_sys, h, w, starSize, sigma)
        cv.imwrite("display_img.png", img)

    # star tracker -------------------------------------------------------------
    os.system("chmod 0700 /run/user/1000/")
    os.environ["LIBCAMERA_LOG_LEVELS"] = "3"
    picam = Picamera2()
    config = picam.create_preview_configuration(main={"size": (2304,1296)})
    picam.configure(config)
    picam.start()             
    picam.capture_file("st_image.png")
    picam.close()
    img = cv.imread("st_image.png", 0)
    res = st.stOperationHardware(img, minAreaHardware, maxAreaHardware, f_cam)

    # truth --------------------------------------------------------------------
    if res == 0:
        failed += 1
    else:
        q = res[0]
        C = res[1]
        errorAngle, errorX, errorY, errorZ = truth.stError(C, ra0, dec0, roll0)
        results.append(errorAngle)

results = np.asarray(results)
print(str(failed) + " failed identifications")
print("successful error angle = " + str(np.mean(results)))
print("successful error standard deviation = " + str(np.std(results)))

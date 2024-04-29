import cv2 as cv
import numpy as np
from systemParameters import * 
import databaseGeneration as dg
import imageGeneration as ig
import starTracker as st
import truth

numFail = 0
avg_x = 0
avg_y = 0
avg_z = 0

for i in range(100):

    # attitude ---------------------------------------------------------------------
    ra0 = np.deg2rad(np.random.random_sample() * 360)
    dec0 = np.deg2rad(180 * np.random.random_sample() - 90)
    roll0 = np.deg2rad(np.random.random_sample() * 360)

    # database generation ----------------------------------------------------------
    if regenerateDatabase:
        dg.generationCatalog(catalogFile, minMag, maxMag)
        dg.stDatabase(catalogFile, minMag, maxMag)
        dg.K_vector(stDatabaseFile)

    # image generation -------------------------------------------------------------
    img = ig.staticImage(generationDataFile, ra0, dec0, roll0, fovx, fovy, f, h, w, 
                        maxStars, starSize, sigma)
    # cv.imshow("img", img)
    # cv.waitKey()
    # print(ra0)
    # print(dec0)
    # print(roll0)

    # star tracker -----------------------------------------------------------------
    res = st.stOperation(img, minArea, maxArea, h, w, f)

    # truth ------------------------------------------------------------------------
    if res == 0:
        # print("identification failed")
        numFail += 1
    else:
        q = res[0]
        C = res[1]
        x_error, y_error, z_error = truth.stError(C, ra0, dec0)
        avg_x += x_error
        avg_y += y_error
        avg_z += z_error
        # print("Percent error in x, y, z dir:")
        # print(x_error)
        # print(y_error)
        # print(z_error)

print("Number of failed identifications: " + str(numFail))
print("Average x error = " + str(avg_x / (100 - numFail)))
print("Average y error = " + str(avg_y / (100 - numFail)))
print("Average z error = " + str(avg_z / (100 - numFail)))

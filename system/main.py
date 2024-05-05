import cv2 as cv
import numpy as np
from systemParameters import * 
import databaseGeneration as dg
import imageGeneration as ig
import starTracker as st
import truth
from matplotlib import pyplot as plt

# random tests
failed = 0
error = 0

numTrials = 1
for i in range(numTrials):
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
    img = ig.staticImage(generationDataFile, ra0, dec0, roll0, fovx, fovy, f, h, w, 
                        maxStars, starSize, sigma)
    # cv.imwrite("orion.png", img)
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
        failed += 1
    else:
        q = res[0]
        C = res[1]
        errorAngle, errorX, errorY, errorZ = truth.stError(C, ra0, dec0, roll0)
        if errorAngle > 0.1:
            print("error greater than 0.1 = " + str(errorAngle))
        # print("error angle = " + str(errorAngle) + " degrees")
        # print("error about x, y, z axis = " + str(errorX) + ", " + str(errorY) + ", " + str(errorZ) + " arcseconds")
        error += errorAngle

print(str(failed) + " failed identifications")
print("successful error angle = " + str(error / (numTrials - failed)))

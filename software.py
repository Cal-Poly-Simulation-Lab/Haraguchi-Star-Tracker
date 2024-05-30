import numpy as np
from systemParameters import * 
import databaseGeneration as dg
import imageGeneration as ig
import starTracker as st
import truth

failed = 0
results = []
numTrials = 10

for i in range(numTrials):
    # attitude -----------------------------------------------------------------
    ra0 = np.deg2rad(np.random.random_sample() * 360)
    dec0 = np.deg2rad(180 * np.random.random_sample() - 90)
    roll0 = np.deg2rad(np.random.random_sample() * 360)

    # database generation ------------------------------------------------------
    if regenerateDatabase:
        dg.generationCatalog(catalogFile, minMag, maxMag)
        dg.stDatabase(catalogFile, minMag, maxMag)
        dg.K_vector(stDatabaseFile)

    # image generation ---------------------------------------------------------
    img = ig.staticImage(generationDataFile, ra0, dec0, roll0, fovx, fovy, 
                         f_sys, h, w, starSize, sigma)

    # star tracker -------------------------------------------------------------
    res = st.stOperation(img, minAreaSoftware, maxAreaSoftware, h, w, f_sys)

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This program launches random walks of increasing lengths and
computes the distance between the origin point and the endpoint of the walk.
The length of the walk represents time (the longer walk time, the longer walk length)


"""
import numpy as np
from random import seed
import matplotlib.pyplot as plt

from openalea import lpy

seed(3)


#model_filename = '5-rw-pseudosphere-geodesic-dist.lpy'
model_filename = 'rw-sphere.lpy'

##############################
# Input variables
##############################

MAXWALKLENGTH = 40
WALKS = 30      # At each length

meter = 1.0  # An object of 1 meter is mapped onto a graphic unit size of 1.0
             # (F(1) for instance corresponds to a forward move of 1 meter)
cm = 0.01 * meter

R = 1 * meter
slen = 0.05 * meter
dt = 1

D = (slen**2)/(4*dt)

###### Arrays for result:
# Array of geodesic lengths for each walk. This array is updated at each timestep.
# It contains the geodesic distance of each walker to the origin at the current time step
# geodesicLen = np.zeros(WALKS)

# At each timestep, the mean, Std and Variance of the lengths of the walks are computed
# using a generic algorithm for curved surfaces to compute the geodesic distance of each walker to the origin.
# this make 3 values at each time step:
geodesicMean = np.zeros(MAXWALKLENGTH)
geodesicStd = np.zeros(MAXWALKLENGTH)
geodesicVar = np.zeros(MAXWALKLENGTH)

# same computation with the use of an analytical formula on the sphere to compute the geodesic distances.
# At each timestep, the mean, Std and Variance of the lengths of the walks are computed
# this make 3 values at each time step:
geodesicMeanTrue = np.zeros(MAXWALKLENGTH)
geodesicStdTrue = np.zeros(MAXWALKLENGTH)
geodesicVarTrue = np.zeros(MAXWALKLENGTH)

# walk length represent the progression of time
for wlen in range(1,MAXWALKLENGTH):

    print ('---', wlen)

    # reconstruct the lsystem model at each time to rerun a ful set of independent walkers
    # (and independent from the walkers at the previous step)
    #geodesicLen = np.zeros(WALKS)

    vardic = {'NBSTEPS':wlen, 'WALKS':WALKS, 'R':R, 'slen':slen}

    lsys = lpy.Lsystem(model_filename, vardic)
    lstring = lsys.derive()
    lsys.sceneInterpretation(lstring)  # interpretation of the final lstring (this computes the geodesic distance)

    # Retrieves computations
    # retreives geodesic length computed using LineTo-based distance
    geodesicLen = lsys.geodesicLen
    # 1. with LineTo function
    posvals = geodesicLen >= 0.0
    #print(posvals)

    # Filter out error vals (as LineTo may fail to compute certain distances)
    geodesicLen = lsys.geodesicLen[posvals]
    #print("Iteration: ", wlen, "geodesic lengths= ", geodesicLen)
    geodesic_dist2 = np.power(geodesicLen, 2)
    geodesicMean[wlen] = np.mean(geodesic_dist2)
    geodesicStd[wlen] = np.std(geodesic_dist2)
    geodesicVar[wlen] = np.var(geodesic_dist2)

    # 2. retreives geodesic length computed using analytic formula
    geodesicLenTrue = lsys.geodesicLenTrue

    geodesicMeanTrue[wlen] = np.mean(geodesic_dist2)
    geodesicStdTrue[wlen] = np.std(geodesic_dist2)
    geodesicVarTrue[wlen] = np.var(geodesic_dist2)

    # reset the array geodesicLen for the next step
    #geodesicLen = np.zeros(WALKS)

print("END REACHED")
print(geodesicMean)


t = np.arange(0,MAXWALKLENGTH,1)
Kg = (1. / R) ** 2  # Courbure de Gauss d'une sphere de rayon R
flat_theo = 4*D*t
curved_theo = 4 * D * t - 4 / 3. * Kg * (D * t) ** 2 * 1
print(4 / 3. * Kg * (D * t) ** 2)
plt.plot(t,flat_theo, c = 'b')
plt.plot(t, curved_theo, c='r')
plt.errorbar(t,geodesicMean, yerr=geodesicStd, fmt='.k')
plt.xlabel('Time')
plt.ylabel('Mean squared length')
plt.title(f'MSGD. Sphere RW,N={WALKS:3d},Time={MAXWALKLENGTH:3d},$\delta$ ={slen:1.2f}m,dt={dt:0.2f},D = {D:0.4f}')
plt.grid(True)
plt.figure(1).canvas.draw() # to force the drawing of the new figure
plt.show()                  # to show the window containing the figure

#print(squared_dists)
#with open('squared_dists.txt', 'wb') as f:
#    np.save(f, squared_dists, allow_pickle=False)

'''
with open('squared_dists.txt', 'rb') as f:
    dists = np.load(f)
'''

#squared_dists = squared_dists * 0.0001

#BoxName = [walk_len for walk_len in range(10, MAXWALKLENGTH)]
#plt.boxplot(squared_dists.T)
#plt.ylim(0,10)
#pylab.xticks([1,2,3], BoxName)

#plt.savefig('geodesic_distances_negative_curvature.png')
#plt.show()
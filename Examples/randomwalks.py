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
import pylab

from openalea import lpy

seed(1)

model_filename = '5-rw-pseudosphere-geodesic-dist.lpy'

lsys = lpy.Lsystem(model_filename)

##############################
# Input variables
##############################

MINWALKLENGTH = 150
MAXWALKLENGTH = 200
NBWALKS = 10      # At each length

# Array with 2 dimensions: walk length x walk index (w)
squared_dists = np.empty((MAXWALKLENGTH-MINWALKLENGTH,NBWALKS))

for walk_len in range(MINWALKLENGTH, MAXWALKLENGTH):
    for w in range(NBWALKS):
        lsys.derivationLength = walk_len
        lsys.NB_STEPS = walk_len
        lstring = lsys.derive()             # Equivalent to a run
        lsys.sceneInterpretation(lstring)   # Execute an interpretation of the final lstring (this computes the geodesic distance)
        if lsys.distance == None:
            print ("*** Distance could not be computed")
            squared_dists[walk_len-MINWALKLENGTH][w] = np.inf
        else:
            squared_dists[walk_len-MINWALKLENGTH][w] = lsys.distance**2
        print(walk_len, w, squared_dists[walk_len-MINWALKLENGTH][w])

print(squared_dists)
with open('squared_dists.txt', 'wb') as f:
    np.save(f, squared_dists, allow_pickle=False)

'''
with open('squared_dists.txt', 'rb') as f:
    dists = np.load(f)
'''

#squared_dists = squared_dists * 0.0001

#BoxName = [walk_len for walk_len in range(10, MAXWALKLENGTH)]
plt.boxplot(squared_dists.T)
plt.ylim(0,2.5)
#pylab.xticks([1,2,3], BoxName)

#plt.savefig('geodesic_distances_negative_curvature.png')
plt.show()
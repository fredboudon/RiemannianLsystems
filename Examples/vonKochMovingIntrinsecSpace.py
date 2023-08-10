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
from openalea.plantgl.all import Viewer

simulation_id = 2
model_filename = '11-Abst2DRiemannSpace-snowflake.lpy'

##############################
# Input variables
##############################

STEPS = np.linspace(-1.9,1.,300)

i = 0
for x in STEPS:
    lsys = lpy.Lsystem(model_filename, {'startx':x})
    lstring = lsys.derive()             # Equivalent to a run, computes the final lstring
    scene = lsys.plot(lstring)          # Executes an interpretation of the final lstring and plots the result in the current instance of the viewer
    Viewer.saveSnapshot('snapshots/test' + str(simulation_id) + '_' + str(i) + '.png')
    i += 1

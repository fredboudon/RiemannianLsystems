#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Increments the gravity felt by the plant

"""
import numpy as np
from random import seed
import matplotlib.pyplot as plt
import pylab

from openalea import lpy
from openalea.plantgl.all import Viewer

simulation_id = 1
model_filename = '11-Abst2DRiemannSpace-hyperbolic-tree.lpy'
PrefixDir = 'HyperbolicTreePtSource/tree-gravity-'

##############################
# Input variables
##############################
Viewer.show()

STEPS = np.linspace(0.0,1.0,200)
#STEPS = np.linspace(100,0.,20)
i = 0
for x in STEPS:
    lsys = lpy.Lsystem(model_filename, {'TEST':2, 'TREE': True, 'FIELD_STRENGTH':x})
    lstring = lsys.derive()             # Equivalent to a run, computes the final lstring
    scene = lsys.plot(lstring)          # Executes an interpretation of the final lstring and plots the result in the current instance of the viewer
    Viewer.saveSnapshot('snapshots/' + PrefixDir + str(simulation_id) + '_' + str(i) + '.png')
    i += 1

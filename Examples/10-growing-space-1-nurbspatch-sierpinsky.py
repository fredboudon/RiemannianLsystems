#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulate the growth of a space which feedsback on the drawing of the Sierpinski form
"""

from openalea import lpy
from openalea.plantgl.all import Viewer

simulation_id = 1
model_filename = '10-growing-space-1-nurbspatch-sierpinski.lpy'
PrefixDir = 'GrowingSpaceNurbspatchSierpinski/sierpinski-'

##############################
# Input variables
##############################
Viewer.show()

total_simulation_time = 200

i = 0
for t in range(0,total_simulation_time):
    lsys = lpy.Lsystem(model_filename, {'FRACTAL_DEPTH': 4, 'T':t, 'dt':0.02})
    lstring = lsys.derive()             # Equivalent to a run, computes the final lstring
    scene = lsys.plot(lstring)          # Executes an interpretation of the final lstring and plots the result in the current instance of the viewer
    Viewer.saveSnapshot('snapshots/' + PrefixDir + str(simulation_id) + '_' + str(i) + '.png')
    i += 1

import os, fnmatch
import numpy as np
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

import imageio.v2 as imageio #to avoid deprecation
def gif_from_images(filenames, gif_name):
    with imageio.get_writer(gif_name+'.gif', mode='I', loop=1, fps=20) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
"""
    This file calls lpy file 3-tree-pseudosphere-crawling.lpy
    and increment the basis of the tree to make it crawl on the
    pseudosphere

    Author: C. Godin, Inria
    Date: 2019-2022
    Lab: RDP ENS de Lyon, Mosaic Inria Team

    call with
    > python 3-tree-pseudosphere-crawling.py 3-tree-pseudosphere-crawling.lpy -t1

"""

# just lauch this file with the command:
# shell> python plant_snapshots

from openalea.plantgl.all import Vector3
from openalea import lpy
from lscene_snapshot import simulate_and_shoot, grid_simulate_and_shoot, plot_images, build_suffix

import argparse

##############################
# Scanning command arguments
##############################

# Scanning argument to define test number
# Instantiate the parser
parser = argparse.ArgumentParser(description='Launching l-system simulations')
# Required positional argument
parser.add_argument('lpymodel_filename', help='please indicate the test number as an integer ...')
parser.add_argument('-t','--test', type=int, required=True, help='please indicate the test number as an integer ...')
# optional arguments:
# ap.add_argument("-n", "--name", required=True, help="name of the user")
args = vars(parser.parse_args())
#print(args)
TEST = args['test']
print ("Simulation of Lpy model:",args['lpymodel_filename'], " for test --> ", TEST)


##############################
# Input variables
##############################

#model_filename = "geometric-model-automated.lpy"
model_filename = args['lpymodel_filename']

###########################################
# Simulate the plant and generate the scans
###########################################

# def build_suffix(var):
#
#     return "O"+str(var["MAX_ORDER"])+"-D"+str(var["NB_DAYS"])+"-MD"+str(var["MD"])+'+' if str(var["SHOW_MERISTEMS"]) else '-'

# Definition of shortnames for variables (used for generating small image names)
variable_short_names = {'root_altitude_max':'max', 'root_altitude_min':'min', 'alt_nb':'n', 'alt_index':'H', 'treedepth' : 'O'}

if TEST == 1: # Snapshot of a grid of simulations (NB_DAYS x MD)

    # fixed variables input to the L-system for all the simulations
    root_altitude_min = -3.5
    root_altitude_max = 3.5
    alt_nb = 50  # the interval [root_altitude_min, root_altitude_max] is divided into alt_nb intervals of equal sizes
    alt_index = 15  # alt_index defines the altitude of the simulated tree
    treedepth = 6

    # Plots a grid of results for different values of m_delay and nb_days = (md, dnb)
    # this is made using a dictionary providing for each md the list daynb to simulate
    # y = dict key is md,
    # x = dict values giving for each key the list of days numbers for which a
    # simulation sim(x,y) must be computed
    fixed_variables_dict = {'root_altitude_max':root_altitude_max, 'root_altitude_min':root_altitude_min, 'alt_nb':alt_nb}
    free_variable_list = ['alt_index','treedepth'] # Should contain the two variables which will vary (simpoints) to make the grid
    rootalt_indexes = list(range(alt_nb))
    simpoints = {6:rootalt_indexes}

    # Building of the image
    # setting camera options
    target_point = Vector3(0,0,1.) # looks a bit above z = 0
    zoomcoef = 7.   # increase to zoom out
    camdist = 1.    # increase to widen the observation window
    camdict = {'camdist':camdist, 'zoomcoef':zoomcoef, 'bb':None, 'target_point':target_point}

    def cam_setting_fun1(x,y):
        '''
        Function that defines the camera setting values for each pair of parameter values (x,y)
        here
        x represents time
        y represents meristem delay
        '''
        t = target_point
        z = zoomcoef
        c = camdist

        return {'camdist':c, 'zoomcoef':z, 'bb':None, 'target_point':t, 'elevation':0.0}

    grid_simulate_and_shoot(simpoints, model_filename, free_variable_list, fixed_variables_dict, cam_settings = cam_setting_fun1, short_names = variable_short_names)



'''
    Library of tools to take snapshots of the result of a l-system simulations

    Authors: Christophe Godin, Inria, 2021
    Licence: Open source LGPL

    The lscene_snapshot library offers 2 main functions:
      simulate_and_shoot(model_filename, variables, suffix, cam_settings = camdict)
      grid_simulate_and_shoot(simpoints, model_filename, free_variable_list, fixed_variables_dict, cam_settings = cam_setting_fun1, short_names = variable_short_names)

'''

import os
from numpy import arange
from openalea.plantgl.all import *
from openalea import lpy
from math import pi

from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_bb(viewer):
  '''
  gets the bounding box of a scene contained in a LPy viewer.
  '''
  sc = viewer.getCurrentScene()
  return BoundingBox(sc)

# scan la scene courante
def take_snapshot(viewer, filename_prefix= ".", suffix = "", cam_settings = {'camdist' : 1., 'zoomcoef' : 1., 'bb' : None, 'target_point' : None}):
  '''
  take a snapshot of a computed lpy_scene, with camera attributes defined in cam_settings:
  - target_point is the point at which the camera is looking (Vector3)
  - camdist fixes the distance of the camera to the target point (in screen units)
  - zoomcoef is a multiplicative zoom factor (higher means zooming out)
  - azimuth is an angle in degrees
  - elevation is a height in screen units with respect to the (x,y) plane (applied after having zoomed in/out)

  filename_prefix: A prefix directory may be defined (default is current directory '.')
  suffix: name of the image
  '''

  if 'camdist' in cam_settings:
    camdist = cam_settings['camdist']
  else:
    camdist = None
  if 'bb' in cam_settings:
    bb = cam_settings['bb']
  else:
    bb = None
  if 'zoomcoef' in cam_settings:
      zoomcoef = cam_settings['zoomcoef']
  else:
    zoomcoef = 1.0
  if 'target_point' in cam_settings:
    target_point = cam_settings['target_point']
  else:
    target_point = None
  if 'elevation' in cam_settings:
    elevation = cam_settings['elevation']
  else:
    elevation = 0.0 # elevation in screen units along the z-axis
  if 'azimuth' in cam_settings:
    azimuth = cam_settings['azimuth']
  else:
    azimuth = 0.0

  # Determine the point to look at
  if target_point == None:
      # define bounding box used to take the picture
      if bb == None:
          # computes the bounding box of this scene
          bbx = get_bb(viewer)
      else:
          # computes the bounding box of this scene
          bbx = bb
      c = bbx.getCenter()
  else:
      c = target_point     # target point to look at is given as an argument

  # computes the  distance of the camera to the target point.
  if camdist == None:
    #p = position of the camera
    #h = head (shooting direction)
    #u = up (upward direction of the shooting)
    p,h,u = viewer.camera.getPosition()

    dist = norm(p-c) #* camdist_factor
    #print ("scene center = ", c)
    #print ("camera position = ", p,h,u)
    #print ("camera distance = ", camdist)
  else:
    dist = camdist

  # moves the position of the camera from (1,0,0) with origin in (0,0,0)
  # to (r,theta,z)=(dist*zoomceof,azimuth, elevation) with origin in c
  np = c + Matrix3.axisRotation((0,0,1),azimuth*pi/180)*Vector3(1,0,0)*dist*zoomcoef
  np = np + Vector3(0,0,elevation)

  # Defines the new camera position and orientation as
  # shooting direction = c-np
  # up vector of the camera keeps oriented towards Z
  viewer.camera.lookAt(np,c)
  iname = filename_prefix+'/'+str(suffix)+'.png'
  # print("2:image name = ", iname)
  viewer.saveSnapshot(iname)
  return iname


# !!!!!!!!!!! NEW VERSION NOT YET TESTED --> TO BE TESTED
def take_circular_snapshots(viewer, delta_a = 36, filename_prefix= ".", suffix = "", cam_settings = {'camdist':1., 'zoomcoef':1., 'bb':None, 'target_point':None}):
  '''
  take a snapshot of a computed lpy_scene contained in the viewer.

  - delta_a is the increment angle in degrees
  '''

  if 'camdist' in cam_settings:
    camdist = cam_settings['camdist']
  else:
    camdist = None
  if 'bb' in cam_settings:
    bb = cam_settings['bb']
  else:
    bb = None
  if 'zoomcoef' in cam_settings:
    zoomcoef = cam_settings['zoomcoef']
  else:
    zoomcoef = 1.0
  if 'target_point' in cam_settings:
    target_point = cam_settings['target_point']
  else:
    target_point = None
  if 'elevation' in cam_settings:
    elevation = cam_settings['elevation']
  else:
    elevation = 0.0 # elevation in screen units along the z-axis
  if 'azimuth' in cam_settings:
    azimuth = cam_settings['azimuth']
  else:
    azimuth = 0.0


  if target_point == None:
  # define bounding box used to take the picture
    if bb == None:
      # computes the bounding box of this scene
      bbx = get_bb(viewer)
    else:
      # computes the bounding box of this scene
      bbx = bb
    c = bbx.getCenter()
  else:
    c = target_point     # target point to look at is given as an argument

  # computes the  distance of the camera to the target point.
  if camdist == None:
    #p = position of the camera
    #h = head (shooting direction)
    #u = up (upward direction of the shooting)
    p,h,u = viewer.camera.getPosition()

    dist = norm(p-c) #* camdist_factor
    #print ("scene center = ", c)
    #print ("camera position = ", p,h,u)
    #print ("camera distance = ", camdist)
  else:
     dist = camdist

  for angle in arange(0,360,delta_a):
  	# Rotate the X-direction (1,0,0) of the scene of angle degrees around the Z direction (0,0,1)
  	# and assigns the initial camera dist (camdist) as the norm of the vector
    # warning: the angle in axisRotation should be given in radians.
    np = c + Matrix3.axisRotation((0,0,1),angle*pi/180)*Vector3(1,0,0)*camdist
    np = np + Vector3(0,0,elevation)

    # Defines the new camera position and orientation as
    # shooting direction = c-np
    # up vector of the camera keeps oriented towards Z
    viewer.camera.lookAt(np,c)
    iname = filename_prefix+'/snapshot_'+str(suffix)+'-'+str(angle)+'.png'
    viewer.saveSnapshot(iname)

  return iname


def simulate_and_shoot(model_filename, variables, suffix, cam_settings):

    # build the L-system with its input variables
    lsys = lpy.Lsystem(model_filename, variables)

    lstring = lsys.derive()
    lsys.plot(lstring) # uses interpretation rules to plot using the viewer
    # Note: there is no way for the moment to get the viewer activated on the lstring without plotting it

    # scans the plant iteratively starting at X = 0 and with view angles increments
    # passed in argument
    # the procedure writes on the disk the different snapshots (.png files)
    # in a file

    # create a directory to store the snapshots
    # 1. current_filename without suffix
    filename = model_filename.split('.')[0]

    # 2. build the new dir name
    newdir = 'snapshots' #+filename+'-'+str(suffix)
    if not os.path.exists(newdir):
      os.makedirs(newdir)

    # 3. creates a snapshot
    lpy_viewer = lsys.Viewer
    # second argument is azimuth
    iname = take_snapshot(lpy_viewer, newdir, suffix, cam_settings = cam_settings)

    # Explicitly delete this L-system
    # (not clear why needed, as this should be done from the destruction of variables local to this function, but indeed needed)
    del lsys

    return iname


def build_suffix(var, short_names = None):
    '''
    takes a dict of variables and constructs a str suffix identifying uniquely this set of parameter
    (for further defining a name corresponding to this set of parameters for storage on disk of corresponding results)
    '''
    sep = ' ' # could also be ' ', '_' or '#'
    stg = ''
    index = 0
    for name in var:
        if short_names != None:
            shname = short_names[name]
        else:
            shname = str(index)
        if shname != '': # if shorname empty, do not use theis variable for suffix
            if index != 0:
                stg += sep
            val = var[name]
            if type(val) == bool:
                stg += shname+':'+('+' if val==True else '-')
            else:
                stg += shname+':'+str(val)
        index += 1
    return stg


def build_variables_dict(x,y,free_variable_list, fixed_variables):
    '''
    builds a dictionary of variables by blending free variables (made of two variables)

    - x,y are the values of the two free free_variables
    - free_variable_list contains the names of the free variables
    - fixed_variables is a dict of variables with already defined values (i.e. fixed)

    Precondition: len(free_variable_list) >= 2
    '''
    assert len(free_variable_list) >= 2 #(at list names for x and y should be defined)

    vardict = fixed_variables.copy()
    vardict.update({free_variable_list[0]:x})
    vardict.update({free_variable_list[1]:y})

    return vardict

def grid_simulate_and_shoot(simpoints, model_filename, free_variable_list, fixed_variables, cam_settings = {'camdist':1., 'zoomcoef':1., 'bb':None, 'target_point':None}, short_names = None):
    '''
    launches simulations of the lpy model model_filename for free parameters values defined in simpoints,
    - simpoints is a dict whose keys are the y values and the values are lists of x-values for each y key
    - free variable names are given in free_variable_list and
    - fixed variables are defined with their values in the fixed_variables dict (parameters fixed for all simulations)

    - cam_setting can be a simple dict
        cam_settings = {'camdist':1., 'zoomcoef':1., 'bb':None, 'target_point':None}
      or a function returning a dict for args = (x,y) = pair of values of the free parameters
        cam_settings(x,y) --> {'camdist':val1, 'zoomcoef':val2, 'bb':val3, 'target_point':val4}

    '''

    # create list of variable names
    fixed_variable_list = fixed_variables.keys()

    # computes the list of all time points for all md, and sorts it.
    total_simpoints = set()
    for y in simpoints:
        total_simpoints = total_simpoints.union(total_simpoints, set(simpoints[y]))
    total_simpoints = sorted(total_simpoints)
    XMAX = len(total_simpoints) # length of the list of all the simulated timepoints merged together
    print("total_simpoints(",XMAX,") = ", total_simpoints)

    # Now make the simulations
    YMAX = len(simpoints)
    image_name_list = []  # will contain the final image names as saved on the disk.
    coords = []           # will store for each resulting image the pairs (md,dnb) corresponding to it
    cnt = 0
    # the grid corresponds to the variation of two varibales x and y (free variables)
    # in general x may represent time (and x0,x1,...,xk) a trajectory of a given value of y.
    # y represents the value of another parameter that is studied in the model with respect to the first one (x, possibly = time).
    for y in simpoints:
        for x in simpoints[y]:
            variables = build_variables_dict(x,y,free_variable_list, fixed_variables)

            suffix = build_suffix(variables,short_names)
            print(x,y,suffix)
            if type(cam_settings) == dict:
                camdict = cam_settings # setting is constant for every pair (x,y)
            else: # should be a function
                assert callable(cam_settings)
                camdict = cam_settings(x,y)
            iname = simulate_and_shoot(model_filename, variables, suffix, camdict)
            image_name_list.append(iname)
            y_index = cnt
            x_index = total_simpoints.index(x)
            coords.append((y_index,x_index))
        cnt += 1
    print("image_name_list(",len(image_name_list),") = ", image_name_list)
    print("Coords list(",len(coords),") = ", coords)

    # Computes the "flattened" index of an image in the array (this index is used to position the image in )
    index_list = []
    for k in range(len(image_name_list)):
        xi = coords[k][0]
        yi = coords[k][1] # index in list total_simpoints
        index_list.append(xi * XMAX + yi +1) # these indexes must be shifted by 1 as they start at 1
    print("array dim = ", YMAX,  XMAX)
    print("index_list(",len(index_list),") = ", index_list)
    plot_images(image_name_list, index_list, dir =".", size = (YMAX, XMAX))

def plot_images(image_name_list, index_list = None, dir = '.', size = (1,1)):
    '''
    Plots the grid of png images @ grid positions defined by index_list

    - index_list = 1 dimentional position = flattened 2D position in a grid of size[0]xsize[1]
    - dir specifies the directory where to print
    - size is the dimensions of the underlying 2D grid
    '''

    # A figure is a canvas that contains axes (= places on which to plot)
    # by default the figure has one axis
    # plot functions can be called on axes only (and not on figures)
    # but a different setting of axes can be created wth subplots
    #fig = plt.figure(figsize=(10, 15))

    dim2 = min(size[0]*3,10)
    dim1 = min(size[1]*3,15)
    print("Image: grid size= ", size[1],size[0], " image --> width,heigth = ", dim1,dim2)
    fig = plt.figure(figsize=(dim1,dim2))

    i = 1

    for iname in image_name_list:

        image = mpimg.imread(dir + '/' + iname)

        if not index_list == None:
          k = index_list[i-1]
          axis = fig.add_subplot(size[0], size[1], k)
        else:
          axis = fig.add_subplot(size[0], size[1], i)
        #axis.add_artist(ab)
        axis.imshow(image)
        axis.axis('off')

        # remove stuff before last '/' and suffix .png from the name
        a = iname.split('/')
        title = a[-1]
        a = title.split('.') # to remove ending .png
        title = a[0] # without .png
        axis.set_title(title)

        i += 1

    #plt.grid()
    #plt.draw()

    #plt.savefig('image_array.png',bbox_inches='tight')
    #fig.suptitle("Simple geometric model simulations")
    fig.tight_layout() # organises figure axes to minimize spacing
    plt.show()

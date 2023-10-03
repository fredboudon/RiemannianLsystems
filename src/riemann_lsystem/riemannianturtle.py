"""
    Classes and functions for manipulating a Riemannian Turtle in LP-y

    Author: C. Godin, Inria
    Date: 2019-2022
    Lab: RDP ENS de Lyon, Mosaic Inria Team

"""
#import logging
from math import dist
from .surfaces import *

from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from openalea.plantgl.all import triangulation,eOptimalTriangulation, eStarTriangulation

""" Pb print messages twice: check this ...
logger = logging.getLogger("Logger")
c_handler = logging.StreamHandler()
logger.addHandler(c_handler)
logger.removeHandler(consoleHandler)

logging.basicConfig(level=logging.DEBUG)
#logger.debug('***** raised a debug message')
#logger.error('***** raised an error')
"""

PERIODIC_COORD_NORMALIZATION = True

def flatten(t):
  return [item for sublist in t for item in sublist]


def midpoint(edge):
  '''
  edge is a pair of two points (uO,v0), (u1,v1)
  '''
  # print("edge=",edge)
  u0, v0 = edge[0]
  u1, v1 = edge[1]

  return (u0 + u1) / 2., (v0 + v1) / 2.


def center(polyline):
  '''
  polyline is a list of n points (u_i,v_i)
  '''
  # print('Polyline = ', polyline)
  dim = len(polyline)
  c = np.zeros(2)
  for i in range(dim):
    c += np.array(polyline[i])
  return tuple(c / dim)


def quadsFromPolyline(polyline, opttrilist):
  '''
  returns a list of quads corresponding to a submesh of the polygone
  '''

  # print("opttrilist=",opttrilist)

  quadlist = []

  mind = np.inf
  maxd = 0

  for k in range(len(opttrilist)):

    t = (polyline[opttrilist[k][0]], polyline[opttrilist[k][1]],
         polyline[opttrilist[k][2]])  # triangle = list of 3 points (u,v)

    # print("t=",t)

    c = center(t)

    m = [midpoint((t[i % 3], t[((i + 1) % 3)])) for i in range(len(t))]

    a = len(t)
    for i in range(a):
      quadlist.append((m[i % a], t[((i + 1) % a)], m[((i + 1) % a)], c))
      d = dist(c, m[i % a])
      if d < mind: mind = d
      if d > maxd: maxd = d

  return quadlist, mind, maxd


def quadsFromQuad(quad):
  '''
  returns a list of quads corresponding to a submesh of the quad
  - quad is a list of 4 points (u_i,v_i) i = 0..3
  '''

  mind = np.inf
  maxd = 0

  c = center(quad)
  m = [midpoint((quad[i % 4], quad[((i + 1) % 4)])) for i in range(len(quad))]

  a = len(quad)
  quadlist = []
  for i in range(a):
    quadlist.append((m[i % a], quad[((i + 1) % a)], m[((i + 1) % a)], c))
    d = dist(c, m[i % a])
    if d < mind: mind = d
    if d > maxd: maxd = d

  return quadlist, mind, maxd


def meshPolygon(polyline, resolution):
  '''
  test:

   polyline = ((0, 0), (1, 0), (1, 2), (-1,1), (0, 2))
   meshPolygon(polyline, .05)

  '''

  # Computes an initial and optimal first triangle optimisation of the polygon
  # as a list of index 3-uples pointing into the original list of points (polyline)
  # This uses CGAL function through PlantGL wrapping
  opttrilist = triangulation(polyline, eOptimalTriangulation)
  #opttrilist = triangulation(polyline, eStarTriangulation)

  # from this returns a first list of quads
  # note that the function returns also the min and max values of the edge length in the quad list
  quadlist, minseg, maxseg = quadsFromPolyline(polyline, opttrilist)

  # Then, if more details are required iterate on each quad (independently)
  # and flatten the resulting lists of quads in a single big list

  midseg = (minseg + maxseg) / 2.
  STOP = True if midseg <= resolution else False

  while not STOP:

    quadlist2 = []  # This list will contain the list of all subquads at the desired resolution
    # Compute a quad list at the next resolution level
    minseg = np.inf
    maxseg = 0

    for quad in quadlist:
      quadsublist, minlen, maxlen = quadsFromQuad(quad)
      quadlist2.append(quadsublist)
      # update the min and max lengths of encountered segments
      if minlen < minseg: minseg = minlen
      if maxlen > maxseg: maxseg = maxlen

    quadlist = flatten(quadlist2)

    midseg = (minseg + maxseg) / 2.
    STOP = True if midseg <= resolution else False

  return quadlist, minseg, maxseg



def riemannian_turtle_init(uvpq, space):
    u, v, p, q = uvpq

    # Compute shitf tensor at u,v, then use it to transform p,q vector
    # representing the coordinates of the covariant basis on the curve
    # expressed in the surface covariant basis
    h = space.shift_vector(u, v, p, q)
    head = h / np.linalg.norm(h)
    up = np.array(space.normal(u, v))
    # print('head,up=',(head,up))
    return head, up


def riemannian_turtle_move_forward(p_uvpq, surf, delta_s, SUBDIV=10, SPEED1 = True, INT_METHOD = 1):
    uu, vv, pp, qq = p_uvpq
    #print ("riemannian_turtle_move_forward 0: input uvpq = ", [uu,vv,pp,qq] )

    # print("Norm pq", np.linalg.norm(np.array([pp,qq])))
    #  print ("pq norm before rot = ", surf.norm(uu,vv,np.array([pp,qq])))
    #  print ("pq norm after rot  = ", surf.norm(uu,vv,npq[0:2]) )

    # NEXT POS: compute turtle's next position and orientation
    # and update the turtle's current state

    # Two slightly different methods to perform the integration:
    # In both cases the norm of the surface velocity does not impact the result (as it should be)
    #
    # to express the distance to move forward from coords expressed in the curve covariant basis
    # The surface vector (p,q) defines the initial velocity of the turtle on its geodesic path.
    # Let us call v the norm of this velocity. s is assumed to be the curvilinear abscissa on the curve
    # Hence |dC/ds| = 1 (rate of change of the distance on the curve with a change in the curvilinear abscissa
    # is constant and = 1).
    # Let us call C_v(s) the same curve traversed at a speed v. This curve results from a reparametrerization
    # of C: r = vs. The initial curve is then C(s) = C_1(s) and we have C_v(s) = C_1(sv)
    # Then, progressing from Cv(s0) to Cv(s) (i.e. at speed v on C) can be done in two ways
    # 1. by going to C_v(s0 + delta_s), with delta_s = s-s0
    # 2. by going to C_1(s0 + delta_s*v)

    # In the instruction F(delta_s) the fact that we get to point C(s0 + delta_s) must be independent of the speed v
    # this can be done in two ways:
    #
    # 1. normalizing the velocity first to a speed v = 1, and then move to C_1(delta_s)
    # 2. keeping the original velocity v, but then moving to C_1(delta_s/v) (one goes artificially faster, then one must go less far)

    # 1. First method (v is set to 1, original delta_s is kept)
    if SPEED1:
        v = surf.norm(uu, vv, [pp,qq])
        npp = pp/v
        nqq = qq/v

        # normalized velocity (v = 1)
        npq = np.array([npp, nqq])

        # Now we can compute the time tics at which the geodesic equation must be integrated
        # over the entire delta_s

        s = np.linspace(0, delta_s, SUBDIV)
        #print('riemannian_turtle_move_forward 1: delta = ', delta_s, 'SUBDIV = ', SUBDIV, 's = ', s)

    # 2. second method (v != 1, delta_s is scaled by 1/v)
    else :
        npq = np.array([pp, qq])
        v = surf.norm(uu, vv, npq[0:2])

        # the distance is reduced  if the speed is increased (and vice versa)
        ds = delta_s / v
        # print("*** ds = ",ds, " dt = ", dt)

        # Now we can compute the time tics at which the geodesic equation must be integrated
        s = np.linspace(0, ds, SUBDIV)

    # computes the new state by integration of the geodesic equation
    # the equation takes into account the original velocity and
    # integrates a trajectory travelled with speed v = |velocity|

    # print("BEFORE: ", np.array([uu,vv,npq[0],npq[1]]))
    # print("s (len = ", len(s),"):", s)

    # We tested two different scipy methods for integration: odeint and solve_ivp
    # solve_ivp seems to be more rapid TODO: compare efficiency of both method in time.
    # however when degenerated points are encountered apparently odeint behaves better in some cases than solve_ivp
    # (see for instance file: 1-geodesic-sphere.lpy, AXIOMTEST = 1 and TEST = 3 (spherical triangle))
    if INT_METHOD == 1:    # Default integration method = odeint
        # scipy.integrate.odeint(func, y0, t, args=(), Dfun=None, col_deriv=0, full_output=0, ml=None, mu=None, rtol=None, atol=None, tcrit=None, h0=0.0, hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12, mxords=5, printmessg=0, tfirst=False)

        uvpq_s, resdict = odeint(surf.geodesic_eq, np.array([uu, vv, npq[0], npq[1]]), s, rtol = 1.49012e-5, atol = 1.49012e-5, full_output = 1)
        #uvpq_s = odeint(surf.geodesic_eq, np.array([uu, vv, npq[0], npq[1]]), s)
        #print("riemannian_turtle_move_forward 2: computed uvpq_s: ")
        #if not resdict['imxer'] == -1:
        #print("*******  ******  ", resdict['imxer'] )
        #print (uvpq_s)
    else:                 # integration method = solve_ivp
        # scipy.integrate.solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, **options)
        # by default uses RungeKutta of order 4/5
        # Warning, signature of fun must be fun(t,y) (and not fun(y,t)), and the result is transposed compared to that returned by odeint.
        t_span = [s[0],s[-1]]
        # to get the right signature for the geodesic equation
        def fun(t,y):
            return surf.geodesic_eq(y,t)

        y0 = [uu, vv, npq[0], npq[1]]
        sol = solve_ivp(fun, t_span, y0, t_eval = s)
        #print(sol.y)
        uvpq_s = np.transpose(sol.y)

    # The problem is that as a result there may be discontinuous coordinates that are not properly handled in the
    # Intersection module

    if PERIODIC_COORD_NORMALIZATION:
      surf.normalize_periodic_coords(uvpq_s)

    return uvpq_s

def parameterspace_turtle_move_forward(p_uvpq, surf, delta_s, SUBDIV=10):
    u, v, p, q = p_uvpq

    assert(SUBDIV > 1)

    theta = np.arctan2(q,p)

    u_new = np.zeros(SUBDIV)
    v_new = np.zeros(SUBDIV)
    p_new = np.full(SUBDIV, p)
    q_new = np.full(SUBDIV, q)

    # computes intermediate points
    dl = delta_s / SUBDIV

    for k in range(SUBDIV):
        u_new[k] = u + dl * (k+1) * np.cos(theta)
        v_new[k] = v + dl * (k+1) * np.sin(theta)

    # combine these 1D array as a (SUBDIV,4) array [[u0,v0,p0,q0],[u1,v1,p1,q1], ...]
    uvpq_s = np.vstack((u_new, v_new, p_new, q_new)).T

    #print('FORWARD --> u0, v0 =', u, v, 'new u, v =', u_new[-1], v_new[-1])

    return uvpq_s

def riemannian_turtle_turn(p_uvpq, surf, deviation_angle):
    u, v, p, q = p_uvpq
    # print("Norm pq", np.linalg.norm(np.array([pp,qq])))

    # ROTATION : Turn turtle with deviation angle
    angle_rad = np.radians(deviation_angle)

    # a surface vector is defined by (pp,qq) at point (uu,vv)
    # the primitive returns
    uu, vv, pp, qq = surf.rotate_surface_vector(u, v, p, q, angle_rad)

    uvpq = np.array([uu, vv, pp, qq])

    return uvpq

def parameterspace_turtle_turn(p_uvpq, surf, deviation_angle):
    u, v, p, q = p_uvpq
    # print("Norm pq", np.linalg.norm(np.array([pp,qq])))

    pq = np.array([p,q])
    # ROTATION : Turn turtle with deviation angle
    angle_rad = np.radians(deviation_angle)

    # a surface vector is defined by (pp,qq) at point (uu,vv)
    # the primitive returns
    rpq = surf.rotation_mat(angle_rad).dot(pq)

    pp, qq = rpq

    #print('TURN --> u, v =', u, v, 'angle =', angle_rad, 'p,q before(',p, q, '), p,q after(', pp, qq, ')')

    uvpq = np.array([u, v, pp, qq])

    return uvpq

def geodesic_to_point(space,uv,uvt,nb_points, max_iter):
    '''
    Computes initial sequences of coords (u,v) to pass to the newton method solver of the class.

    nb_points includes uv and ut,vt, meaning that for nb_points = 10 for instance, 8 intermediary points will be computed
    in addition to both u,v and ut,vt.
    '''
    #print("ENTERING geodesic_to_point", flush=True)
    # Checks that (ut,vt) = coords of the target point are valid
    ut,vt = uvt

    if not space.check_coords_domain(ut, vt):
        print("geodesic_to_point: target point out of space coord domain: ", ut, vt)
        return None

    # the returned value may be None if the preconditions are not respected
    # e.g. (u,v) must be different from (ut,vt)
    uvpq_s, error_array, errorval = space.geodesic_to_target_point(uv, uvt, nb_points, max_iter)
    #uvpq_s = space.parameterspace_line_to_target_point(uv, uvt, nb_points)
    #errorval =0
    #print("Error = ", errorval)

    if PERIODIC_COORD_NORMALIZATION:
      space.normalize_periodic_coords(uvpq_s)

    return uvpq_s, errorval

def geodesic_to_point_LS(space,uv,uvt,nb_points):
    '''
    Computes initial sequences of coords (u,v) to pass to the newton method solver of the class.

    nb_points includes uv and ut,vt, meaning that for nb_points = 10 for instance, 8 intermediary points will be computed
    in addition to both u,v and ut,vt.
    '''

    # Checks that (ut,vt) = coords of the target point are valid
    ut,vt = uvt

    if not space.check_coords_domain(ut, vt):
        print("geodesic_to_point: target point out of space coord domain: ", ut, vt)
        return None

    # the returned value may be None if the preconditions are not respected
    # e.g. (u,v) must be different from (ut,vt)
    uvpq_s  = space.geodesic_to_target_point_LS(uv, uvt, nb_points)

    if True:
      space.normalize_periodic_coords(uvpq_s)

    return uvpq_s

def geodesic_to_point_shoot(space,uv,uvt,nb_points):
    '''
    Computes initial sequences of coords (u,v) to pass to the newton method solver of the class.

    nb_points includes uv and ut,vt, meaning that for nb_points = 10 for instance, 8 intermediary points will be computed
    in addition to both u,v and ut,vt.
    '''

    # Checks that (ut,vt) = coords of the target point are valid
    ut,vt = uvt

    if not space.check_coords_domain(ut, vt):
        print("geodesic_to_point: target point out of space coord domain: ", ut, vt)
        return None

    # the returned value may be None if the preconditions are not respected
    # e.g. (u,v) must be different from (ut,vt)
    uvpq_s  = space.geodesic_to_target_point_shoot(uv, uvt, nb_points)

    if PERIODIC_COORD_NORMALIZATION:
      space.normalize_periodic_coords(uvpq_s)

    return uvpq_s

def parameterspace_line_to_point(space, uv, uvt, nb_points):
    '''
    Computes sequences of coords (u,v) to go to uvt within the parameter space

    nb_points includes uv and ut,vt, meaning that for nb_points = 10 for instance,
    8 intermediary points will be computed in addition to both u,v and ut,vt.
    '''

    #print("ENTERING parameterspace_line_to_point", flush=True)
    # Checks that (ut,vt) = coords of the target point are valid
    ut,vt = uvt

    if not space.check_coords_domain(ut, vt):
        print("geodesic_to_point: target point out of space coord domain: ", ut, vt)
        return None
    #print("ENTERING parameterspace_line_to_point, after checking coords")
    # the returned value may be None if the preconditions are not respected
    # e.g. (u,v) must be different from (ut,vt)
    uvpq_s = space.parameterspace_line_to_target_point(uv, uvt, nb_points)

    # NOTE: Periodic coordinates ARE NOT normalized here as this function is
    # used to initialize geodesic_to_point functions.

    return uvpq_s

def geodesic_distance_to_point(space,uv,uvt,nb_points, max_iter=20):
    '''
    Computes initial sequences of coords (u,v) to pass to the newton method solver of the class.

    nb_points includes uv and ut,vt, meaning that for nb_points = 10 for instance, 8 intermediary points will be computed
    in addition to both u,v and ut,vt.
    '''

    # Checks that (ut,vt) = coords of the target point are valid
    ut,vt = uvt

    if not space.check_coords_domain(ut, vt):
        print("geodesic_to_point: target point out of space coord domain: ", ut, vt)
        return None

    # the returned value may be None if the preconditions are not respected
    # e.g. (u,v) must be different from (ut,vt)
    dist, error_array, errorval = space.geodesic_distance(uv, uvt,nb_points, max_iter)
    '''
    try:
    dist, error_array = space.geodesic_distance(uv, uvt)
except RuntimeError as error:
    print(error)
    error_array = []
    '''


    #np.set_printoptions(precision=3, suppress = True)
    #print("Cumulated error sequence throughout iterations: ",error_array)


    return dist, error_array, errorval
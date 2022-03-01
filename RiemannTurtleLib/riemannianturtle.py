"""
    Classes and functions for manipulating a Riemannian Turtle in LP-y

    Author: C. Godin, Inria
    Date: 2019-2022
    Lab: RDP ENS de Lyon, Mosaic Inria Team

"""
from surfaces import *

from scipy.integrate import odeint
from scipy.integrate import solve_ivp


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
    # print("Norm pq", np.linalg.norm(np.array([pp,qq])))

    # if endflag and deviation_angle != 0 :
    #  print ("uvpq just after turning = ", [uu,vv,npq[0:2]] )
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
    # of C: r = vs. The initial curve is then C(s) = C_1(s) and we have Cv(s) = C_1(sv)
    # Then, progressing from Cv(s0) to Cv(s) (i.e. at speed v on C) can be done in two ways
    # 1. by going to C_v(s0 + delta_s), with delta_s = s-s0
    # 2. by going to C_1(s0 + delta_s*v)

    # In the instruction F(delta_s) the fact that we get to point C(s0 + delta_s) must be independent of the speed v
    # this can be done in two ways. In the geodesic this displacement is made at rate v = 1.
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
    if INT_METHOD == 1:    # integration method = odeint
        # scipy.integrate.odeint(func, y0, t, args=(), Dfun=None, col_deriv=0, full_output=0, ml=None, mu=None, rtol=None, atol=None, tcrit=None, h0=0.0, hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12, mxords=5, printmessg=0, tfirst=False)
        uvpq_s = odeint(surf.geodesic_eq, np.array([uu, vv, npq[0], npq[1]]), s)
        #print(uvpq_s)
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

def geodesic_to_point(space,uv,uvt,nb_points, max_iter=20, mu=0.2):
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
    #uvpq_s = space.geodesic_to_target_point_swapped(uv, uvt, nb_points, max_iter, mu)
    uvpq_s = space.geodesic_to_target_point(uv, uvt, nb_points, max_iter, mu)


    return uvpq_s
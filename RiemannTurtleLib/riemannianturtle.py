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


def riemannian_turtle_move_forward(p_uvpq, surf, delta_s, SUBDIV=10, SPEED1 = True):
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
    # scipy.integrate.odeint(func, y0, t, args=(), Dfun=None, col_deriv=0, full_output=0, ml=None, mu=None, rtol=None, atol=None, tcrit=None, h0=0.0, hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12, mxords=5, printmessg=0, tfirst=False)
    uvpq_s = odeint(surf.geodesic_eq, np.array([uu, vv, npq[0], npq[1]]), s)

    # scipy.integrate.solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, **options)
    # t_eval = s
    # t = [s[0],s[-1]]
    # sol = solve_ivp(surf.geodesic_eq, s, np.array([uu, vv, npq[0], npq[1]]))
    #uvpq_s = sol.y

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

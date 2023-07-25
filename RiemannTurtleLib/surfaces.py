"""
    Classes and functions for manipulating a Riemannian Turtle in LP-y

    Author: C. Godin, Inria
    Date: 2019-2022
    Lab: RDP ENS de Lyon, Mosaic Inria Team

TODO: add Extrusions (i.e. generalized cylinders) to list of available surface classes
TODO: compute principal curvatures, gauss curvature for the mother class
"""
import math
import numpy as np
import numpy.linalg as linalg

from importlib import reload

import geodesic_between_2points
geodesic_between_2points = reload(geodesic_between_2points)

from geodesic_between_2points import *

# To install: pip install pynverse
from pynverse import inversefunc

def derivatives(surf, u, v, order):
    return surf.derivatives(u,v,order)

# Fred's code :
def bruteforce_derivatives(surf, u, v, order):
    du = 0.01
    from scipy.misc import derivative
    def upt(ui):
        return np.array(surf.evaluate_single((ui,v)))

    def vpt(vi):
        return np.array(surf.evaluate_single((u,vi)))

    def dvpt(vi):
        def lupt(ui):
            return np.array(surf.evaluate_single((ui,vi)))
        return derivative(lupt, u, du)

    if order == 1:
        return [[0,  derivative(vpt, v, du) ],
                [derivative(upt, u, du), derivative(dvpt, v, du) ]]

    elif order == 2:
        return [[0,  0,  derivative(vpt, v, du, n=2)],
                [0, derivative(dvpt, v, du)],
                [derivative(upt, u, du, n=2)] ]

# computation of generic derivatives using scipy
################################################
from scipy.misc import derivative

# Functions to generate derivatives (with a single x argument) automatically from a function with several arguments
# Note that the original function func, may have optional arguments

# Create a function with a single argument from a known list of arguments (function parameters)
# given as a list: eg. gen_func(f,(2,3,10))
# means f(x,2,3,10)
def gen_func(func, *args):
    def ff(x):
        return func(x, *args)
    return ff

# function defined as
def gen_prime_deriv(func, *args):
    def prime_derive(x):
        return derivative(func, x, dx = 1e-6, n = 1, args=args)
    return prime_derive

def gen_second_deriv(func, *args):
    def second_derive(x):
        return derivative(func, x, dx = 1e-6, n = 2, args=args)
    return second_derive

# TODO: Split this class in a purely abstract class and a class IntrinsicRiemannianSpace2D that creates a branch different from the Riemaniann 2D surface one
class RiemannianSpace2D:
    '''
    This class represents an intrinsic 2D space with coordinates u,v
    Each point of the coordinate space (u,v) is mapped in the R3 euclidean space to the point (x=0,y=u,z=v)

    The distance between nearby points separated by du,dv is given by a metric:
    ds2 = g11 du^2 + 2 g12 du dv + g22 dv^2

    where g11, g12 and g22 are functions of u,v and are passed to the constructor of the class
    (with possibly arguments as a list in metric_tensor_params)

    All other classes (for parametric surfaces) will be inherited from this class,
    (as the was to compute geodesics, RC sympbols may be inherited)
    but usually they will redefine the some of them.
    '''

    def __init__(self, g11 = None, g12 = None, g22 = None, g11s = None, g12s = None, g22s = None, umin = 0, umax = 1.0, vmin = 0, vmax = 1.0, metric_tensor_params = (), STOP_AT_BOUNDARY_U = False, STOP_AT_BOUNDARY_V = False):
      # functions to compute the metric
      self.g11 = g11
      self.g12 = g12
      self.g22 = g22
      self.g11s = g11s
      self.g12s = g12s
      self.g22s = g22s

      self.metric_tensor_params = metric_tensor_params

      # sets domain bounds
      self.umin = umin
      self.umax = umax
      self.vmin = vmin
      self.vmax = vmax

      self.metric_tensor_params = metric_tensor_params

      self.STOP_AT_BOUNDARY_U = STOP_AT_BOUNDARY_U
      self.STOP_AT_BOUNDARY_V = STOP_AT_BOUNDARY_U

    def S(self,u,v):
      """
      by default (u,v) = (y,z) and x = 0
      """
      x = 0.
      y = u
      z = v
      return np.array([x,y,z])

    def check_coords_domain(self,u,v):
        '''
        Checks whether domain boundary was reached at u,v
        Do something only if the boundary reached in the parameter space
        and if the flag to stay on boundary is on (by defaut, it is off).
        '''
        boundary_reached = False

        if self.STOP_AT_BOUNDARY_U:
            if u > self.umax:
                boundary_reached = True
                u = self.umax
            elif u < self.umin:
                boundary_reached = True
                u = self.umin
        if self.STOP_AT_BOUNDARY_V:
            if v > self.vmax:
                boundary_reached = True
                v = self.vmax
            elif v < self.vmin:
                boundary_reached = True
                v = self.vmin

        return u,v, boundary_reached

    def check_coords(self, uvpq):
        '''
        Verification of the impact of degenerated points
        by default returns upvq
        '''
        return uvpq

    def sets_boundary_flag(self, STOP_AT_BOUNDARY_U = False, STOP_AT_BOUNDARY_V=False):
        """
        By default the class contains a flag that restricts the values of (u,v)
        to stay in the domain (umin,umax,vmin,vmax) while the turtle is moving .

        This functions may be used to change this default behaviour:
        if set to True, check_coord function will returned bounded values
        when called to check specific (u,v)

        This can be used to stop the turtle movement
        """
        self.STOP_AT_BOUNDARY_U = STOP_AT_BOUNDARY_U  # boolean value
        self.STOP_AT_BOUNDARY_V = STOP_AT_BOUNDARY_V  # boolean value


    def Shift(self,u,v):
      """
      Also called: Pushforward operator.

      the Shift tensor may be viewed as the coordinates of the surface
      covariant basis expressed in the ambiant basis (3x2) = 2 column vectors in 3D.
      It is a (3,2 matrix) to transform the coordinates u,v in x,y,z
      Here u,v, are mapped on the y,z plane
      It can also be used to transform a vector in the u,v tangent plane into a 3D vector
      in the ambient space if any.
      """
      xu = 0.
      yu = 1.
      zu = 0.
      xv = 0.
      yv = 0.
      zv = 1.

      return np.array([[xu,xv],[yu,yv],[zu,zv]])

    def normal(self,u,v):
      # the normal is always borne by the x axis (as the space is the plane y,z)
      nx = 1.
      ny = 0.
      nz = 0.
      return np.array([nx,ny,nz])

    def uv_domain(self):
        #print("Domain = ", self.umin,self.umax,self.vmin,self.vmax)
        return self.umin,self.umax,self.vmin,self.vmax

    def covariant_basis(self,u,v):
      A = self.Shift(u,v) # Compute shitf tensor at the new u,v,
      S1 = A[:,0] # first colum of A = first surface covariant vector in ambiant space
      S2 = A[:,1] # second column of A = second surface covariant vector in ambiant space
      return S1,S2

    def covariant_basis_and_velocity(self,u,v,p,q):
      A = self.Shift(u,v) # Compute shitf tensor at the new u,v,
      S1 = A[:,0] # first colum of A = first surface covariant vector in ambiant space
      S2 = A[:,1] # second column of A = second surface covariant vector in ambiant space

      velocity = A.dot(np.array([p, q]))

      return S1,S2,velocity

    def shift_vector(self,u,v,p,q):
      '''
      This is the Jacobian of the transformation passing from the parameter space to the ambiant space.
      If f denotes tis mapping, it is also called the differential mapping df, or the push-forward operator
      '''
      A = self.Shift(u,v) # Compute shitf tensor at the new u,v,

      # Computes the 3D vector corresponding to the vector p,q
      # in the tangent plane at point u,v.
      vector = A.dot(np.array([p, q]))

      return vector

    def metric_tensor(self, u, v):
        args = self.metric_tensor_params # args is a store in the object as a tuple
        guu = self.g11(u, v, *args)      # args is unpacked before calling the function
        guv = gvu = self.g12(u, v, *args)
        gvv = self.g22(u, v, *args)
        return np.array([[guu, guv], [gvu, gvv]])

    def inverse_metric_tensor(self,u,v):
      """
      tensor = inverse of the metric tensor
      """
      return np.linalg.inv(self.metric_tensor(u,v))

    def norm(self,u,v,S):
      """
      uses the metric tensor at point u,v to compute the norm of a surface vector S
      of components S1,S2 in the local covariant basis
      """
      #print("norm: ******")
      #print("S = ", S)
      S1 = S[0]
      S2 = S[1]
      g = self.metric_tensor(u,v)
      return np.sqrt(S1*S1*g[0,0]+S1*S2*g[0,1]+S2*S1*g[1,0]+S2*S2*g[1,1])

    def ds(self,u,v,du,dv):
      """
      uses the metric tensor at point u,v to compute the ds at a point (u,v) in direction (du,dv)
      """
      #print("norm: ******")
      #print("S = ", S)
      g = self.metric_tensor(u,v)
      return np.sqrt(du*du*g[0,0]+du*dv*g[0,1]+dv*du*g[1,0]+dv*dv*g[1,1])

    def firstFundFormCoef(self,u,v):
      """
      Returns the coefficients of the fondamental forms I and II at point (u,v)
      - E,F,G: First fundamental form = metric (I = ds^2 = E du^2 + 2F dudv + G dv2)
      """
      g = self.metric_tensor(u,v)

      E = g[0,0]
      F = g[1,0]
      G = g[1,1]

      return (E,F,G)

    def passage_matrix_cb2ortho(self,u,v):
      """
      defines the passage matrix to pass from the covariant basis (cv) to an orthomormal
      basis. The latter is constructed using Gram-Schmidt method starting with S1
      (useful for example to make rotation in the tangent plane defined by u,v)
      """
      S1,S2 = self.covariant_basis(u,v)
      len1 = np.linalg.norm(S1)

      if np.isclose(len1,0.):
        # Build the orthonormal basis using the normal vector and
        print("length basis vector 1 is 0 !!!")

      a = 1/len1 # also defined by 1/sqrt(g11) as s1.s1 = len1**2 = g11
      g11 = self.metric_tensor(u,v)[0,0]
      g12 = self.metric_tensor(u,v)[0,1]
      #S22 = S2 - g12 * S1
      coef = g12/g11
      S22 = S2 - coef * S1
      b = 1/np.linalg.norm(S22)

      # print("J2orthonormal")
      # print("covariant S1 = ", S1)
      # print("covariant S2 = ", S2)
      # print("         S22 = ",S22)
      # print("g12 = ", g12, " a = ", a, " b = ", b)

      return np.array([[a,-coef*b],[0,b]])

    def passage_matrix_cb2ortho_inverse(self,u,v):
      '''
      Inverse of the Matrix to pass from the orthog basis to the covariant basis
      '''
      S1,S2 = self.covariant_basis(u,v)
      a = np.linalg.norm(S1)
      g11 = self.metric_tensor(u,v)[0,0]
      g12 = self.metric_tensor(u,v)[0,1]
      #S22 = S2 - g12 * S1
      coef = g12 / g11
      S22 = S2 - coef * S1
      b = np.linalg.norm(S22)
      #print("Jfromorthonormal")
      #print("covariant S1 = ", S1)
      #print("covariant S2 = ", S2)
      #print("         S22 = ",S22)
      #print("g12 = ", g12, " a = ", a, " b = ", b)
      return np.array([[a,coef*a],[0,b]])

    def is_degenerate_point(self, u,v,p,q):
        """
        Procedure to detect a degenerated point. By default checks if
        none of the basis vectors is a null vector
        """
        S1, S2 = self.covariant_basis(u, v)
        len1 = np.linalg.norm(S1)
        len2 = np.linalg.norm(S2)

        return np.isclose(len1, 0.) or np.isclose(len2, 0.)

    def rotation_mat(self, a):
        """
        returns a matrix rotation for an angle a.
        a should be given in radians
        """
        # note that the matrix is a set of line vectors
        # first array = first line of the matrix,
        # second array = second line.
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    def rotate_surface_vector_degenerate_basis(self, u, v, vectpq,angle):
        '''
        rotation in case of a degenerated covariant basis (eg. at the poles of a sphere.
        This class must be implemented by the daughter classes depending on their specificities
        '''

        print("Abstract class: RiemannianSpace2D. Method should be implemented on the daugther classes that need to handle degenerated covariant basis")

        vectpqy = [u,v,vectpq[0],vectpq[1]]

        return vectpqy

    def rotate_surface_vector_apply_matrixes(self,u, v, vectpq,angle_rad):
        '''
        Assuming that the covariant basis is sound (independent vectors),
        this function applies the matrix R = P R' P^-1 to vector vectpq = (p,q) to get rotated vector vectpqy.
        - R' is the known rotation matrix in the orthonormal basis (angle is in radians)
        - P is the passage matrix from the covariant basis to the orthonormal basis

        IMPORTANT: This assumes that the covariant basis exists at point u,v (i.e. u,v is
        not a degenerated point).
        '''

        vectpq1 = self.passage_matrix_cb2ortho_inverse(u, v).dot(vectpq)   # application of P^-1
        vectpq2 = self.rotation_mat(angle_rad).dot(vectpq1)                # application of R'
        vectpqy = self.passage_matrix_cb2ortho(u, v).dot(vectpq2)          # application of P

        # P = surf.passage_matrix_cb2ortho(u,v)
        # Pinv = surf.passage_matrix_cb2ortho_inverse(u,v)
        # Id = np.dot(P,Pinv)
        # print ("P :", P)
        # print ("Pinv :", Pinv)
        # print ("P*Pinv = ", Id)
        return [u,v,vectpqy[0],vectpqy[1]]

    def rotate_surface_vector(self,u,v,p,q,angle):
        '''
        This function rotates a vector X whose coordinates are given in the covariant
        basis at (u,v) and transforms it into a vector Y whose coordinates (py,qy) are given
        in the covariant basis.

        IMPORTANT: This assumes that the covariant basis exists at point u,v (i.e. u,v is
        not a degenerated point). If this is not the case, e.g. poles of the sphere,
        the function must be overloaded in the corresponding daughter class.
        '''
        if not np.isclose(angle, 0.):
            vectpq = np.array([p, q])
            # 1. computes an orthogonal frame from the local covariant basis using Gram-Schmidt orthog. principle)
            # 2. computes the components of the direction vector (given in the local covariant basis as [pp,qq]),
            # in the orthonormal frame
            # 3. then perform the rotation of the vector by an angle deviation_angle
            # 4. finally, transforms back the resulting vector in the covariant basis

            if self.is_degenerate_point(u,v,p,q):
                # determine new p,q using geodesic equ.
                # print("length basis vector is 0 !!!")
                vectuvpqy = self.rotate_surface_vector_degenerate_basis(u, v, vectpq, angle)
            else:
                # the returned vector is of the form [u,v,p,q]
                vectuvpqy = self.rotate_surface_vector_apply_matrixes(u, v, vectpq, angle)
        else:
            # if no turn, nothing changes
            vectuvpqy = np.array([u,v,p,q])

        return vectuvpqy


    def ChristoffelSymbols(self, u, v, printflag = False):
        """
        defined as the scalar product of S^a . dS_b / dS^c, with a,b,c in {u,v}
        """
        #u = min(self.umax,max(self.umin,u))
        #v = min(self.vmax,max(self.vmin,v))

        # covariant basis
        #S_u, S_v = self.covariant_basis(u,v)

        #g = self.metric_tensor(u,v)

        # inverse metric tensor ig
        ig = self.inverse_metric_tensor(u,v)

        # Christoffel symbols (CS) that is an array of 2x2x2 = 8 numbers
        # Gamma^a_bc  = CS[a,b,c]

        # Computation of the derivatives of the metric to prepare the RC coef computation
        args = self.metric_tensor_params

        # computation of derivative functions of the metric
        d1g11f = gen_prime_deriv(self.g11,u,*args)
        d1g12f = gen_prime_deriv(self.g12,u,*args)
        d1g22f = gen_prime_deriv(self.g22,u,*args)
        d2g11f = gen_prime_deriv(self.g11s,v,*args)
        d2g12f = gen_prime_deriv(self.g12s,v,*args)
        d2g22f = gen_prime_deriv(self.g22s,v,*args)

        d1g11 = d1g11f(u)
        d2g11 = d2g11f(v)
        d1g12 = d1g12f(u)
        d1g21 = d1g12       # not used below due to symmetry
        d2g12 = d2g12f(v)
        d2g21 = d2g12
        d1g22 = d1g22f(u)
        d2g22 = d2g22f(v)

        CS = np.zeros((2, 2, 2))
        CS[0, 0, 0] = 0.5 * ig[0][0] * d1g11 + 0.5 * ig[0][1] * (2 * d1g12 - d2g11)
        CS[1, 0, 0] = 0.5 * ig[1][0] * d1g11 + 0.5 * ig[1][1] * (2 * d1g12 - d2g11)
        CS[0, 1, 0] = 0.5 * ig[0][0] * d2g11 + 0.5 * ig[0][1] * d1g22
        CS[1, 1, 0] = 0.5 * ig[1][0] * d2g11 + 0.5 * ig[1][1] * d1g22
        CS[0, 0, 1] = CS[0, 1, 0]
        CS[1, 0, 1] = CS[1, 1, 0]
        CS[0, 1, 1] = 0.5 * ig[0][0] * (2 * d2g21 - d1g22) + 0.5 * ig[0][1] * d2g22
        CS[1, 1, 1] = 0.5 * ig[1][0] * (2 * d2g21 - d2g22) + 0.5 * ig[1][1] * d2g22

        if printflag:
            print("Christoffel coefs:")
            print(CS)

        return CS

    def geodesic_eq(self,uvpq,s):
        """
        definition of a geodesic at the surface starting at point u,v and heading in direction p,q
        expressing the tangent components in the surface covariant basis.
        """
        u,v,p,q = uvpq
        Gamma = self.ChristoffelSymbols(u, v)
        lhs1 = - Gamma[0,0,0]*p**2 - 2*Gamma[0,0,1]*p*q - Gamma[0,1,1]*q**2
        lhs2 = - Gamma[1,0,0]*p**2 - 2*Gamma[1,0,1]*p*q - Gamma[1,1,1]*q**2
        return [p,q,lhs1,lhs2]

    def homogeneize_discretization(self,X):
        """
        Homogeneizes the discretization of a polyline given as a sequence of (u,v,p,q) coords
        by replacing the original segments with segments of equal length.

        X: discrete polyline in the form of a 1D array as a (m,4) array [u0,v0,p0,q0,u1,v1,p1,q1, ...]
        Hout: returned homogeneized polyline in the form of a 1D array as a (m,4) array [uu0,vv0,pp0,qq0,uu1,vv1,pp1,qq1, ...]
        """
        #1. Transform the input in a m-dimensional array of uvpq-vectors (4-vectors): Xinput
        m = int(X.shape[0] / 4)
        Xin = X.reshape(m,4)
        Xout = np.array(Xin) # duplicate Xin and in particular sets up extremity points A and B in Xout

        # Array of segment length in X the (u,v) domain (contains m-1) segments
        seg_length_array = [np.sqrt((Xin[k][0]-Xin[k-1][0])**2  +  (Xin[k][1]-Xin[k-1][1])**2) for k in range(1,m)]
        tot_length = sum(seg_length_array)
        ds = tot_length / (m-1) # new length between equidistant points in the parameter domain (u,v)
        #print(f"curve length: {tot_length:.3f}, nb points: {m:2d}, homogeneized ds: {ds:.3f}")

        # Computes the cumulated distance from the origin on each m-1 segment
        cum_lengths = np.cumsum(seg_length_array) # m elements
        arr = np.empty(m-1) # array of m elements
        arr.fill(ds)        # filled with ds
        cum_homogeneous_lengths = np.cumsum(arr) # cumulated array of ds

        # find the indexes of the elements immediately after the given index
        # the given indexes are given in the form of an array cum_length
        insertion_indexes = np.searchsorted(cum_lengths, cum_homogeneous_lengths)

        for k in range(0,m-2): # loop on m-2 segments (the last one is no use as B is already stored in Xout)
            #for segment k the index of the point after its insertion is stored in
            index2 = insertion_indexes[k]   # gets the index of the point after
            index1 = index2-1               # gets the index of the point before
            u1,v1,p1,q1 = Xin[index1+1]     # +1 to take into account the shift in segment arrays and points arrays like Xin
            u2,v2,p2,q2 = Xin[index2+1]
            dold = cum_lengths[index2]      # cumulated distance to just before point
            dnew = cum_homogeneous_lengths[k]
            deltad = dold - dnew            # should be always >=0 as this was obtained by inex sorting
            # Then use this difference in path length to ponderate the interpolation to compute the new point
            doldseg = np.sqrt((u1-u2)**2+(v1-v2)**2)
            rho = deltad / doldseg
            u = rho * u1 + (1-rho) * u2
            v = rho * v1 + (1-rho) * v2
            p = rho * p1 + (1-rho) * p2
            q = rho * q1 + (1-rho) * q2
            Xout[k+1] = [u,v,p,q]

        Xout = Xout.reshape(4*m,)

        return Xout

    def parameterspace_to_target_point(self, uv, uvt, m):
        # Initialization : computes a first the sequence of u,v indexes and their velocity values p,q .
        u,v = uv
        ut,vt = uvt

        # print('Got to function geodesic_to_target_point !!!!! ')
        # at least the two endpoints
        assert( m >= 2 )

        # u,v values: an efficient initialization consists of taking the straight line joining u,v to ut,vt in the parameter space
        u_initseq = np.zeros(m)
        v_initseq = np.zeros(m)
        delta_u = (ut-u) / (m-1)
        delta_v = (vt-v) / (m-1)
        #print('uv,utvt, m,delta_u,delta_v:', uv, uvt, m,delta_u,delta_v)
        for k in range(m):
            u_initseq[k] = u + delta_u * k
            v_initseq[k] = v + delta_v * k

        # p,q values: use the first fundamental form to compute du/ds and then deduce dv/ds = uv_slope * du/ds
        if np.isclose(u,ut) and np.isclose(v,vt):
            return None # the two points should be different

        p_initseq = np.zeros(m)
        q_initseq = np.zeros(m)

        # test if u = ut
        UEQ = np.isclose(u, ut)

        # Initialize the value (p,q) for each point m (u,v) points.
        for k in range(m):
            Ek,Fk,Gk = self.firstFundFormCoef(u_initseq[k],v_initseq[k])
            if not UEQ:
                uv_slope = (vt - v) / (ut - u)
                duds = 1/np.sqrt(Ek + 2 * Fk * uv_slope + Gk * uv_slope**2)
                p_initseq[k] = duds
                q_initseq[k] = uv_slope * duds
            else:
                p_initseq[k] = 0
                q_initseq[k] = 1/np.sqrt(Gk)

        # combine these 1D array as a (m,4) array [[u0,v0,p0,q0],[u1,v1,p1,q1], ...]
        uvpq_init_seq = np.vstack((u_initseq, v_initseq, p_initseq, q_initseq)).T

        # convert to a single dim 4*m array [u0,v0,p0,q0,u1,v1,p1,q1, ...]
        return uvpq_init_seq

    def geodesic_to_target_point(self, uv, uvt, m, max_iter, epsilon_conv=1e-3):
        '''
        Computes the geodesic path from point (u,v) to target point (ut,vt) using Newton's method described
        in (Maekawa 1996). Journal of Mechanical design, ASME Transactions, Vol 118, No 4, p 499-508
        - uv are the u,v coordinates of the source point
        - uvt are the u,v coordinates of the target point
        - m is the number of discretization points nb_points (including endpoints (u,v) and (ut,vt))
        - max_iter is the maximum number of iteration of the newton method
        '''

        X0 = self.parameterspace_to_target_point(uv, uvt, m).reshape((4*m,))
        X = np.array(X0)

        # --> Initialization completed here.

        # 2. Newton method: Loop on improving initial path to reach a geodesic using the jacobian
        end_test = False
        ERROR = -1  # Initializes the ERROR type (-1 is not meaningful and should always be replaced by another value)
        i = 0
        last_average_delta_X_norm = np.inf
        average_delta_X_norm_array = []
        while not end_test:
            #print('##############  MAIN LOOP i = ', i)
            # 1. Evaluate Delta_s = (distance) between curve points.
            # X being defined (a set of (uvpq) values along the path of size m
            # we approximate the distance between consecutive points
            # in the Riemannian space by the euclidean chord between these points:
            # There are m points and m-1 segments indexed 0 .. m-2 in delta_s
            # with delta_s[k] being the distance P_k+1 - P_k on the curve.

            delta_s = np.zeros(m-1)
            for k in range(m-1):
                # extracts u,v coords of consecutive points on the curve
                u1, v1 = X[4 * k], X[4 * k + 1]
                u2, v2 = X[4 * (k+1)], X[4*(k+1) +1]
                # computes the corresponding points (np arrays) in the physical space
                P1 = self.S(u1,v1)
                P2 = self.S(u2,v2)
                # the norm is euclidean as a proxy, relying on the fact
                # that if the points are close enough, we can consider them
                # almost in a locally euclidean space.
                # Note that delta_s[k] contains P_k+1 - P_k on the curve
                delta_s[k] = np.linalg.norm(P2-P1)

            # 2. Compute the residual vector R corresponding to state X

            #print('delta_s = ', delta_s)
            R = compute_residual_vec(self, X, delta_s)
            #print(R)
            #print(' ----> R norm = ', standardized_L1norm(R))

            # 3. Compute deltaX of the Newton method
            # This method builds the Jacobian to perform the Newton's method and computes

            delta_X = compute_deltaX(self, X, R, delta_s)

            # 4. Check deltaX norm
            # The standardization might be tuned using MU,MV,MP,MQ parameters
            average_delta_X_norm = standardized_L1norm(delta_X)
            #print(' ----> DELTA_X NORM :', average_delta_X_norm)

            # Store the average error.
            average_delta_X_norm_array.append(average_delta_X_norm)

            # 5. Test the norm of delta_X
            # If sufficiently small exit the loop (or if maximum iteration number is reached)
            # note if maxiter = 0, this means that the user wants the initial path.
            if average_delta_X_norm < epsilon_conv:
                #print("SMALL ERROR REACHED !!!!!!!! ")
                ERROR = 0           # Convergence
                end_test = True
            elif average_delta_X_norm > 10 * average_delta_X_norm_array[0]:
                # We estimate that the solution diverges if before reaching maxiter
                # the error becomes 100 x the error corresponding to the initial solution.
                ERROR = 1           # Diverges before end of iterations
                end_test = True
                #raise RuntimeError(f"ERROR: from ({u:.3f},{v:.3f}) to ({ut:.3f},{vt:.3f}), solution diverges after {i:d}/{max_iter:d} steps.")
            elif i >= max_iter:
                ERROR = 2           # Did not converge before end of iterations, but error kept bounded
                end_test = True

            #elif last_average_delta_X_norm < average_delta_X_norm:
            #    print(f"******* INSTABILITY DETECTED: error could not be decreased at step {i:d} / {max_iter:d} ")
            #    end_test = True
            else:

                i += 1

                # Estimation of mu as a function of the  error
                if average_delta_X_norm > 0.5: #0.5
                    mu = 0.02 #0.02
                elif average_delta_X_norm > 0.3: #0.3
                    mu = 0.04 #0.04
                else:
                    mu = 0.06 #0.06

                last_X = np.array(X)
                last_average_delta_X_norm = average_delta_X_norm

                X = X + (mu * delta_X)
                # Below: optional step to homogeneize the segment lengths in the (u,v) domain
                # This step can be commented out (and keep non-homogeneized X only)
                # Tests show that this seems more robust with length homogeneization
                X = self.homogeneize_discretization(X)

        # In any case, if the final solution is poorer than the initial solution
        # restores the initial path with its initial error
        # Note: in this case, the average_delta_X_norm_array will have identical start and end values
        if average_delta_X_norm > average_delta_X_norm_array[0]:
            #print("SOLUTION FOUND NO BETTER THAN INITIAL SOLUTION --> INITIAL SOLUTION KEPT")
            X = X0
            ERROR = 3
            average_delta_X_norm_array.append(average_delta_X_norm_array[0])

        # The resulting value X is a vector of size 4m make an array of upvq values
        # of size m.

        # the first 4 terms
        geodesic_path = X.reshape((m,4))

        return geodesic_path, np.array(average_delta_X_norm_array), ERROR

    def path_distance(self, uvpqs):
        """
        Compute the geometric distance between set of points given by uvpqs
        """
        P1 = self.S(uvpqs[0][0], uvpqs[0][1])
        dist = 0.
        for uu, vv, _, _ in uvpqs:
            P2 = self.S(uu, vv)
            dist += np.linalg.norm(P2-P1)
            P1 = P2
        return dist

    def geodesic_distance(self, uv, uvt, nb_points = 20, max_iter= 20):
        """
        Compute the geodesic distance between two points given as u,v and ut,vt
        """
        nb_points  = 20  # To be better estimated
        max_iter = 20    # To be better estimated

        uvpq, errarray, errorval = self.geodesic_to_target_point(uv, uvt, nb_points, max_iter)

        dist = self.path_distance(uvpq)

        #print("Dist(A,B) = ", dist)
        return dist, errarray, errorval


# Base class for the definition of parametric surfaces
class ParametricSurface(RiemannianSpace2D):
    ''''# Surface position vector
    def S(self,u,v):
       print("S: base ParametricSurface class - ABSTRACT: NO IMPLEMENTATION")

    def uv_domain(self):
        #print("Domain = ", self.umin,self.umax,self.vmin,self.vmax)
        return self.umin,self.umax,self.vmin,self.vmax

    def Shift(self,u,v):
      """ Shit tensor (3,2 matrix) to transform the coordinates u,v in x,y,z
      It is derived from the partial derivatives of the surface equations wrt u,v
      DS(u,v)/du, DS(u,v)/dv
      (may be could be computed automatically from S(u,v) ... see that later)
      """
      print("Shift tensor: base ParametricSurface class - ABSTRACT: NO IMPLEMENTATION")
    '''
    def __init__(self, umin,umax,vmin,vmax, STOP_AT_BOUNDARY_U = False, STOP_AT_BOUNDARY_V = False):
        super(ParametricSurface, self).__init__(umin=umin,umax=umax,vmin=vmin,vmax=vmax,STOP_AT_BOUNDARY_U = STOP_AT_BOUNDARY_U, STOP_AT_BOUNDARY_V = STOP_AT_BOUNDARY_V)

    def secondsuu(self,u,v):
      """ Vectors of the second derivatives of the point S(u,v):
      D2S(u,v)/du^2, D2S(u,v)/du2dv2, D2S(u,v)/dv^2 (the fourth /dvdu is symmetric to /dudv)
      tensor (3,3 matrix))

      Used for instance to compute the coefficients of the second fundamental form (LMN)
      """
      print("Seconds tensor: base ParametricSurface class - ABSTRACT: NO IMPLEMENTATION")

    def secondsuv(self,u,v):
      print("Seconds tensor: base ParametricSurface class - ABSTRACT: NO IMPLEMENTATION")

    def secondsvv(self,u,v):
      print("Seconds tensor: base ParametricSurface class - ABSTRACT: NO IMPLEMENTATION")

    def normal(self,u,v):
      '''
      the choice of a + sign must be make consistent parameterizing of the surface
      and its orientation (if orientable)
      Here we assume
      - a + sign is consistent with the parameterization
      - both covariant vectors are not null
      if len1 becomes to close to 0, a specific method must be written
      in the corresponding daughter class
      '''
      S_u, S_v = self.covariant_basis(u,v)

      N1 = + np.cross(S_u,S_v)
      len1 = np.linalg.norm(N1)
      assert(not np.isclose(len1, 0))

      return N1/len1

    '''
    def covariant_basis(self,u,v):
      A = self.Shift(u,v) # Compute shitf tensor at the new u,v,
      S1 = A[:,0] # first colum of A = first surface covariant vector in ambiant space
      S2 = A[:,1] # second column of A = second surface covariant vector in ambiant space
      return S1,S2
    '''

    def metric_tensor(self,u,v):
        """
          Via the dot product for parametric surfaces=
          scalar product of covariant basis
        """
        S_u, S_v = self.covariant_basis(u,v)
        guu = np.dot(S_u,S_u)
        guv = np.dot(S_u,S_v)
        gvu = np.dot(S_v,S_u)
        gvv = np.dot(S_v,S_v)
        if (np.isclose(guu,0.0) and np.isclose(guv,0.0) and np.isclose(gvv,0.0) ):
            print ("NULL METRIC TENSOR DETECTED ")
        return np.array([[guu,guv],[gvu,gvv]])

    def fundFormCoef(self,u,v):
      """
      Returns the coefficients of the fondamental forms I and II at point (u,v)
      - E,F,G: First fundamental form = metric (I = ds^2 = E du^2 + 2F dudv + G dv2)
      - L,M,N: Second funcdamental form = curvature (II = L du^2 + 2M dudv + N dv2)
      The second fondamental form represents for given du,dv how the surface
      gets quadratically away from the tangent plane.
      if d(du,dv) is the distance of a point P(du,dv) to the tangent compliance
      II = 2 * d
      (See for instance A. Gray p282)
      """
      E,F,G = self.firstFundFormCoef(u,v)

      Suu = self.secondsuu(u,v)
      Suv = self.secondsuv(u,v)
      Svv = self.secondsvv(u,v)

      Normal = self.normal(u,v)

      L = np.dot(Suu,Normal)
      M = np.dot(Suv,Normal)
      N = np.dot(Svv,Normal)

      return (E,F,G,L,M,N)

    def localCurvatures(self,u,v):
        """
        Returns Gaussian (K), mean (H), kappa_min, kappa_max local curvatures at point u,v
        """
        E,F,G,L,M,N = self.fundFormCoef(u,v)

        K = (L*N-M**2)/(E*G-F**2)
        H = (E*N+G*L-2*F*M)/(2*(E*G-F**2))

        # TODO: check whether (H**2 - K) can become negative
        discriminant = H ** 2 - K
        if np.isclose(discriminant,0):
            k1 = k2 = H
        elif discriminant < 0:
            print("discriminant < 0")
            k1 = float('nan')
            k2 = float('nan')
        else:
            k1 = H + np.sqrt(discriminant)
            k2 = H - np.sqrt(discriminant)

        #if abs(k1) > abs(k2):
        if k1 > k2:
            kappa_min = k2
            kappa_max = k1
        else:
            kappa_min = k1
            kappa_max = k2

        return K, H, kappa_min, kappa_max

    def shapeOperator(self,u,v):
        """
        Returns the shape operator at point u,v expressed in the covariant basis (Gray 1993, p275)
        (Warning, this matrix might not be symmetric as the covariant basis is not orthogonal in general.)
        returns empty arrays if frame is undefined
        """
        E,F,G,L,M,N = self.fundFormCoef(u,v)

        A11 = (M*F-L*G)/(E*G-F**2)
        A12 = (L*F-M*E)/(E*G-F**2)
        A21 = (N*F-M*G)/(E*G-F**2)
        A22 = (M*F-N*E)/(E*G-F**2)

        # Shape operator expressed in the covariant basis
        shape_op = np.array([[A11,A12],[A21,A22]])

        if np.isnan(A11) or np.isnan(A12) or np.isnan(A21) or np.isnan(A22) \
        or A11 == np.inf or A12 == np.inf or A21 == np.inf or A22 == np.inf:
            #raise ValueError('Shape operator contains inf values',shape_op)
          return [], []


        # Computation of the shape operator in an orthonormal basis, to obtain a symmetric matrix representation
        # of the symmetric shape operator: Ssym = P^-1 S P

        '''
        shape_op_2 = shape_op.dot(self.passage_matrix_cb2ortho(u,v))                    # S P
        shape_op_sym = self.passage_matrix_cb2ortho_inverse(u, v).dot(shape_op_2)   # P^-1 S P

        if shape_op_symprint("shape operator ----> ")
        print(shape_op_sym)

        # eigen values, eigen vectors of the shape operator
        # note: the ith eigen vector is the ith column of evv: u = vv[:,i]
        ev_sym, evv_sym = linalg.eig(shape_op_sym)

        # transport back the eigen vectors in the covariant basis: evv = P evv_sym
        # Note: the eigen values keep the same in the two bases.
        evv = self.passage_matrix_cb2ortho(u, v).dot(evv_sym)
        ev = ev_sym
        #print("fundFormLocalFrame: ", ev, evv)
        '''

        return shape_op

    def principalDirections(self,u,v):
        """
        Returns the principal directions of curvature at point u,v
        cf Struiks 1961 Dover p 80
        """
        E,F,G,L,M,N = self.fundFormCoef(u,v)
        #print("\tE,F,G\tL,M,N : ", E,F,G,L,M,N)

        K, H, kappa_min, kappa_max = self.localCurvatures(u, v)
        #print(" K, H, kappa_min, kappa_max : ", K, H, kappa_min, kappa_max)

        # computation of the normal curvatures in the direction dvdu
        # There are 2 Equations verified by the normal curvature in direction dvdu
        # 1. A * du + B dv = 0
        # 2. B * du + C dv = 0
        #
        Amax = L-kappa_max*E
        Bmax = M-kappa_max*F
        Cmax = N-kappa_max*G

        Amin = L-kappa_min*E
        Bmin = M-kappa_min*F
        Cmin = N-kappa_min*G

        #print("equs max : ", Amax, Bmax, Cmax)
        #print("equs min : ", Amin, Bmin, Cmin)

        # Equations for max curvature
        if not np.isclose(Amax,0) and not np.isclose(Bmax,0) :
          dudvmax = - Bmax/ Amax
          dirmax = [dudvmax,1]
        elif not np.isclose(Amax,0) and np.isclose(Bmax,0) :
          # du = 0
          if np.isclose(Cmax,0):
            # dv can be any
            dirmax = [0, 1]
          else:
            # i.e. A!=0, B = 0 and C != 0
            # no principal direction can be detected
            dirmax = []
        elif np.isclose(Amax,0) and not np.isclose(Bmax,0) :
          # i.e. A =0, B != 0
          # means that B dv = 0 => dv = 0
          # means that B du = 0 => du = 0
          # the point is degenerated (no principal direction)
          dirmax = []
        else:
          assert(np.isclose(Amax,0) and np.isclose(Bmax,0))
          # means that C dv = 0
          # du can be any
          if not np.isclose(Cmax,0):
            # means that dv = 0
            # du may be any (no constraint from eqs 1 or 2)
            dirmax = [1,0]
          else:
            # C = 0
            # dv can be any
            # du can be any
            # all directions are principal directions
            # => choose one along u
            dirmax = [1,0]

        # Equations for min curvature
        if not np.isclose(Amin,0) and not np.isclose(Bmin,0) :
          dudvmax = - Bmin/ Amin
          dirmin = [dudvmax,1]
        elif not np.isclose(Amin,0) and np.isclose(Bmin,0) :
          # du = 0
          if np.isclose(Cmin,0):
            # dv can be any
            dirmin = [0, 1]
          else:
            # i.e. A!=0, B = 0 and C != 0
            # no principal direction can be detected
            dirmin = []
        elif np.isclose(Amin,0) and not np.isclose(Bmin,0) :
          # i.e. A =0, B != 0
          # means that B dv = 0 => dv = 0
          # means that B du = 0 => du = 0
          # the point is degenerated (no principal direction)
          dirmin = []
        else:
          assert(np.isclose(Amin,0) and np.isclose(Bmin,0))
          # means that C dv = 0
          # du can be any
          if not np.isclose(Cmin,0):
            # means that dv = 0
            # du may be any (no constraint from eqs 1 or 2)
            dirmin = [1,0]
          else:
            # C = 0
            # dv can be any
            # du can be any
            # all directions are principal directions
            # => choose one along u
            dirmin = [1,0]

        # if
        if abs(kappa_min) > abs(kappa_max): # swap min and max
            kappa_tmp = kappa_min
            kappa_min = kappa_max
            kappa_max = kappa_tmp
            dirtmp = dirmin
            dirmin = dirmax
            dirmax = dirtmp

        # Max corresponds to the curvature with maximal ABSOLUTE VALUE
        return kappa_min, kappa_max, dirmin, dirmax

    def ChristoffelSymbols(self, u, v, printflag = False):
        """
        defined as the scalar product of S^a . dS_b / dS^c, with a,b,c in {u,v}
        """
        #u = min(self.umax,max(self.umin,u))
        #v = min(self.vmax,max(self.vmin,v))

        # covariant basis
        S_u, S_v = self.covariant_basis(u,v)

        # inverse metric tensor ig
        ig = self.inverse_metric_tensor(u,v)

        # contravariant basis
        Su = ig[0][0] * S_u + ig[0][1] * S_v
        Sv = ig[1][0] * S_u + ig[1][1] * S_v

        S_uu = self.secondsuu(u,v)
        S_uv = S_vu = self.secondsuv(u,v)
        S_vv = self.secondsvv(u,v)

        # compute the dot product ...
        # Christoffel symbols (RCS) that is an array of 2x2x2 = 8 numbers
        # Gamma^a_bc  = RCS[a,b,c]

        RCS = np.zeros((2,2,2))
        RCS[0,0,0] = np.dot(Su,S_uu)
        RCS[1,0,0] = np.dot(Sv,S_uu)
        RCS[0,1,0] = np.dot(Su,S_vu)
        RCS[1,1,0] = np.dot(Sv,S_vu)
        RCS[0,0,1] = np.dot(Su,S_uv)
        RCS[1,0,1] = np.dot(Sv,S_uv)
        RCS[0,1,1] = np.dot(Su,S_vv)
        RCS[1,1,1] = np.dot(Sv,S_vv)

        if printflag:
            print("Riemann Christoffel coefs:")
            print(RCS)

        return RCS

class Sphere(ParametricSurface):

    def __init__(self, R = 1.0):
        super(Sphere, self).__init__(umin=0, umax=2 * np.pi, vmin=-np.pi/2., vmax=np.pi/2.)
        self.R = R # radius of the sphere
        self.CIRCUM = 2*np.pi*self.R  # Circumference

    # Surface position vector
    # uvpq is an 1x4 array of reals where:
    # - u,v are the coordinates on the surface of the moving point
    # - p,q are the coordinates of the initial vector corresponding to
    # the initial direction in the local covariant basis (at point [u,v]

    def S(self,u,v):
      """ Returns the coordinates (x,y,z) of a position vector restricted to the sphere surface
      as a function of the two surface coordinates (u,v)
      u = azimuth (u in [0,2Pi], counted from the x-axis, where u = 0)
      v = elevation (v in [-Pi/2,+Pi/2] )
      """
      x = self.R*np.cos(u)*np.cos(v)
      y = self.R*np.sin(u)*np.cos(v)
      z = self.R*np.sin(v)
      return np.array([x,y,z])

    def check_coords(self,uvpq):
        '''
        Verification of the impact of degenerated points
        On a sphere, trajectories that go through the poles need
        to have uvpq possibly corrected after passing the pole.
        Here, we don't know whether
        the trajectory went through the poles, but if it is
        the case, upvq have to be checked, and possibly corrected
        '''

        # First checks whether u,v are in the domain
        # For a sphere and its parameterization,
        # u may be any real number (representing an angle between 0 and 2 pi
        # v must be between -pi/2 and pi
        u, v, p, q = uvpq
        w = v
        if v < 0:  # if negative apply the rule to -v
            w = -v
        # Now test the rule on w (representing either v or -v)
        # Compute the floor division and modulo of w by pi/2
        k, r = divmod(w, np.pi / 2.)
        if int(k) % 2 != 0:  # floor division is an odd number
            if np.isclose(r, 0.):
                # consider that w is either pi/2 or -pi/2
                pass
            else:
                #print("*******          angles =", u, v, p, q)
                if v >= 0:
                    v = np.pi - v
                else:
                    v = - np.pi - v
                u = u + np.pi
                q = -q
                #print("******* changing angles =", u, v, p, q)

        # No needs of additional check for u,v on the boundary
        # Poles are considered as boundaries, but as they are degenerate,
        # they are not a true boundary and we don't want to stop there
        # is not terminal point.

        return [u,v,p,q]

    # the Shift tensor may be wieved as the coordinates of the surface
    # covariant basis expressed in the ambiant basis (3x2) = 2 column vectors in 3D.
    def Shift(self,u,v):
      """
      Shit tensor (3,2 matrix) to transform the coordinates u,v in x,y,z
      It is derived from the partial derivatives of the surface equations wrt u,v
      (may be could be computed automatically from S(u,v) ... see that later)
      """
      xu = -self.R*np.sin(u)*np.cos(v)
      yu =  self.R*np.cos(u)*np.cos(v)
      zu = 0
      xv = -self.R*np.cos(u)*np.sin(v)
      yv = -self.R*np.sin(u)*np.sin(v)
      zv = self.R*np.cos(v)

      return np.array([[xu,xv],[yu,yv],[zu,zv]])

    def secondsuu(self,u,v):
      """
      second derivatives of the position vector at the surface
      """
      xuu = -self.R*np.cos(u)*np.cos(v)
      yuu = -self.R*np.sin(u)*np.cos(v)
      zuu = 0
      return np.array([xuu,yuu,zuu])

    def secondsuv(self,u,v):
      xuv =  self.R*np.sin(u)*np.sin(v)
      yuv = -self.R*np.cos(u)*np.sin(v)
      zuv = 0
      return np.array([xuv,yuv,zuv])

    def secondsvv(self,u,v):
      xvv = -self.R*np.cos(u)*np.cos(v)
      yvv = -self.R*np.sin(u)*np.cos(v)
      zvv = -self.R*np.sin(v)

      return np.array([xvv,yvv,zvv])

    def metric_tensor(self,u,v):
      guu = ((self.R*np.cos(v)))**2
      guv = gvu = 0.
      gvv = self.R**2
      return np.array([[guu,guv],[gvu,gvv]])

    def normal(self,u,v):
      '''
      Note: this normal is well defined everywhere despite the fact that the poles
      are degenerated. This comes that in the computation cos(v) appears as a global
      factor, that cancels out in the normed vector.
      '''
      nx = np.cos(u)*np.cos(v)
      ny = np.sin(u)*np.cos(v)
      nz = np.sin(v)

      nvec = np.array([nx,ny,nz])
      lennvec = np.linalg.norm(nvec)
      return nvec/lennvec

    def rotate_surface_vector_degenerate_basis(self, u, v, vectpq, angle):
        '''
        rotation in case of a degenerated covariant basis (eg. at the poles of a sphere.
        This class must be implemented by the daughter classes depending on their specificities
        '''

        #print("!!! Degenerated covariant basis detected ...")

        # At the poles, (u,v) = (u,+/-pi/2), u being the value of the azimuth
        # of the great circle passing at the pole that conducted to the pole
        # therefore, at the poles, turning means changing the value of u by the
        # turning angle!
        # The value of the parameter u is the value when arriving at the pole from previous
        # points on the trajectory. It therefore represents the heading of the turtle
        # Then turning by an angle  means adding an angle to this heading direction
        #print ("BEFORE: u,v = ", u,v,vectpq)
        u += angle
        v = np.pi/2 if v >=0 else -np.pi/2
        #print ("AFTER : u,v = ", u,v,vectpq)


        # In adition, the velocity needs be reinitiated in the new direction of u
        # FIXME: at speed one by default, but should be able to use any figure
        vectpqy = [u, np.pi/2, 0., 1.]

        # Note that u has been changed by this rotation in the retured vector
        return vectpqy

    def geodesic_eq(self,uvpq,s):
      u,v,p,q = uvpq

      X = self.S(u, v)              # current 3D point on the trajectory
      P1 = self.S(0, np.pi / 2)     # pole 1
      P2 = self.S(0, -np.pi / 2)    # pole 2

     # the following test check in the 3D space (and not in the parameter space)
      # whether the current point is close to either poles.
      if np.allclose(X, P1) or np.allclose(X, P2):      # np.isclose(np.sin(v), 1.): #(v = pi/2)
          # If yes slightly approximates the equation to avoid divergence at the pole
          # by pretending that p keeps constant while the point is in the narrow
          # degenerated region.

          #print("PI/2 REACHED ...", np.allclose(X, P1), np.allclose(X, P1) )

          # makes the approximation that p does not change close to this very narrow region
          # this avoids the divergence of p at the poles

          return [p, q, p, -np.cos(v) * np.sin(v) * p ** 2]
      else: # standard equation
          return [p, q, 2*np.tan(v)*p*q, -np.cos(v)*np.sin(v)*p**2]

# function defined to compute its inverse
def pseudo_sphere_z(x, R):
    return R * (x - np.tanh(x))

class PseudoSphere(ParametricSurface):

    def __init__(self, R = 1.0,zmin=-5,zmax=5):
      # Need to infer min and max values of u from that of z
      # warning: inverse function returns an array for a given input value
      min = inversefunc(pseudo_sphere_z, y_values=zmin, args=(R))
      max = inversefunc(pseudo_sphere_z, y_values=zmax, args=(R))
      super(PseudoSphere, self).__init__(umin=min,umax=max,vmin=0,vmax=2*np.pi,STOP_AT_BOUNDARY_V=True) # only keep z in the domain

      self.R = R # radius of the sphere
      self.CIRCUM = 2*np.pi*self.R  # Circumference

    # Surface position vector
    # uvpq is an 1x4 array of reals where:
    # - u,v are the coordinates on the surface of the moving point
    # - p,q are the coordinates of the initial vector corresponding to
    # the initial direction in the local covariant basis (at point [u,v]

    def S(self,u,v):
      """ Returns the coordinates (x,y,z) of a position vector restricted to the sphere surface
      as a function of the two surface coordinates (u,v)
      u = azimuth (u in [0,2Pi], counted from the x-axis, where u = 0)
      v = elevation (v in [-Pi/2,+Pi/2] )
      """
      x = self.R*np.cos(v)/np.cosh(u)
      y = self.R*np.sin(v)/np.cosh(u)
      z = self.R * (u - np.tanh(u))

      return np.array([x,y,z])

    # the Shift tensor may be wieved as the coordinates of the surface
    # covariant basis expressed in the ambiant basis (3x2) = 2 column vectors in 3D.
    def Shift(self,u,v):
      """ Shit tensor (3,2 matrix) to transform the coordinates u,v in x,y,z
      It is derived from the partial derivatives of the surface equations wrt u,v
      (may be could be computed automatically from S(u,v) ... see that later)
      """
      sech = 1 / np.cosh(u)
      xu = -self.R*sech*np.tanh(u)*np.cos(v)
      yu = -self.R*sech*np.tanh(u)*np.sin(v)
      zu = self.R*np.tanh(u)**2
      xv = -self.R*sech*np.sin(v)
      yv =  self.R*sech*np.cos(v)
      zv = 0

      return np.array([[xu,xv],[yu,yv],[zu,zv]])

    def secondsuu(self,u,v):
      """
      second derivatives of the position vector at the surface
      """
      sech = 1/np.cosh(u)
      xuu = -self.R*np.cos(v)*sech*(sech**2 - np.tanh(u)**2)
      yuu = -self.R*np.sin(v)*sech*(sech**2 - np.tanh(u)**2)
      zuu = 2*self.R*np.tanh(u)*sech**2
      return np.array([xuu,yuu,zuu])

    def secondsuv(self,u,v):
      sech = 1 / np.cosh(u)
      xuv =  self.R*sech*np.tanh(u)*np.sin(v)
      yuv = -self.R*sech*np.tanh(u)*np.cos(v)
      zuv = 0
      return np.array([xuv,yuv,zuv])

    def secondsvv(self,u,v):
      sech = 1 / np.cosh(u)
      xvv = -self.R*sech*np.cos(v)
      yvv = -self.R*sech*np.sin(v)
      zvv = 0

      return np.array([xvv,yvv,zvv])

    def metric_tensor(self,u,v):
      sech = 1 / np.cosh(u)

      guu = (self.R**2) * np.tanh(u)**2
      guv = gvu = 0.
      gvv = (self.R**2) * sech**2
      return np.array([[guu,guv],[gvu,gvv]])

    def normal(self,u,v):
      nx = -np.tanh(u)*np.cos(v)
      ny = -np.tanh(u)*np.sin(v)
      nz = - 1/np.cosh(u)

      nvec = np.array([nx,ny,nz])
      lennvec = np.linalg.norm(nvec)
      return nvec/lennvec


    def geodesic_eq(self,uvpq,s):
      u,v,p,q = uvpq
      sech = 1 / np.cosh(u)
      tanh = np.tanh(u)
      pdot = -((sech**2-tanh**2)/(np.sinh(u)*np.cosh(u)))*p**2 - (sech/np.sinh(u))*q**2
      qdot = 2*tanh*p*q

      X = self.S(u, v)

      # check if z is close to 0
      if np.isclose(X[2], 0):  # np.isclose(np.sin(v), 1.): #(v = pi/2)
          # If yes slightly approximates the equation to avoid divergence at the pole
          # by pretending that p keeps constant while the point is in the narrow
          # degenerated region.

          # print("RIM REACHED ...")

          # makes the approximation that p does not change close to this very narrow region
          # this avoids the divergence of p at the rim (z = 0)

          return [p, q, p,qdot]
      else:
          return [p,q,pdot,qdot]


# For the moment the implementation is not done (copied from Sphere)
class EllipsoidOfRevolution(ParametricSurface):

    def __init__(self, a = 1.0, b = 0.5):
      super(EllipsoidOfRevolution, self).__init__(umin=0,umax=2*np.pi,vmin=-np.pi/2.,vmax=np.pi/2)
      self.a = a # radius of the circle at equator
      self.b = b # other radius

    # Surface position vector
    # uvpq is an 1x4 array of reals where:
    # - u,v are the coordinates on the surface of the moving point
    # - p,q are the coordinates of the initial vector corresponding to
    # the initial direction in the local covariant basis (at point [u,v]

    def S(self,u,v):
      """ Returns the coordinates (x,y,z) of a position vector restricted to the sphere surface
      as a function of the two surface coordinates (u,v)
      u = azimuth (u in [0,2Pi], counted from the x-axis, where u = 0)
      v = elevation (v in [-Pi/2,+Pi/2] )
      """
      x = self.a*np.cos(u)*np.cos(v)
      y = self.a*np.sin(u)*np.cos(v)
      z = self.b*np.sin(v)
      return np.array([x,y,z])

    def check_coords(self,uvpq):
        '''
        Verification of the impact of degenerated points
        On a sphere, trajectories that go through the poles need
        to have upvq possibly corrected. Here, we don't know whether
        the trajectory went through the poles, but if it is
        the case, upvq have to be checked
        '''

        # First checks whether u,v are in the domain
        # For a an ellipsoid of revolution and its parameterization,
        # u may be any real number (representing an angle between 0 and 2 pi
        # v must be between -pi/2 and pi
        u, v, p, q = uvpq
        w = v
        if v < 0:  # if negative apply the rule to -v
            w = -v
        # Now test the rule on w (representing either v or -v)
        # Compute the floor division and modulo of w by pi/2
        k, r = divmod(w, np.pi / 2.)
        if int(k) % 2 != 0:  # floor division is an odd number
            if np.isclose(r, 0.):
                # consider that w is either pi/2 or -pi/2
                boundary = True
            else:
                #print("*******          angles =", u, v, p, q)
                if v >= 0:
                    v = np.pi - v
                else:
                    v = - np.pi - v
                u = u + np.pi
                q = -q
                #print("******* changing angles =", u, v, p, q)

        # No needs of additional check for u,v on the boundary
        # Poles are considered as boundaries, but as they are degenerate,
        # they are not a true boundary and we don't want to stop there
        # is not terminal point.

        return [u,v,p,q]

    # the Shift tensor may be wieved as the coordinates of the surface
    # covariant basis expressed in the ambiant basis (3x2) = 2 column vectors in 3D.
    def Shift(self,u,v):
      """ Shit tensor (3,2 matrix) to transform the coordinates u,v in x,y,z
      It is derived from the partial derivatives of the surface equations wrt u,v
      (may be could be computed automatically from S(u,v) ... see that later)
      """
      xu = -self.a*np.sin(u)*np.cos(v)
      yu =  self.a*np.cos(u)*np.cos(v)
      zu = 0
      xv = -self.a*np.cos(u)*np.sin(v)
      yv = -self.a*np.sin(u)*np.sin(v)
      zv = self.b*np.cos(v)

      return np.array([[xu,xv],[yu,yv],[zu,zv]])

    def secondsuu(self,u,v):
      """
      second derivatives of the position vector at the surface
      """
      xuu = -self.a*np.cos(u)*np.cos(v)
      yuu = -self.a*np.sin(u)*np.cos(v)
      zuu = 0
      return np.array([xuu,yuu,zuu])

    def secondsuv(self,u,v):
      xuv =  self.a*np.sin(u)*np.sin(v)
      yuv = -self.a*np.cos(u)*np.sin(v)
      zuv = 0
      return np.array([xuv,yuv,zuv])

    def secondsvv(self,u,v):
      xvv = -self.a*np.cos(u)*np.cos(v)
      yvv = -self.a*np.sin(u)*np.cos(v)
      zvv = -self.b*np.sin(v)

      return np.array([xvv,yvv,zvv])

    def metric_tensor(self,u,v):
      guu = ((self.a*np.cos(v)))**2
      guv = gvu = 0.
      gvv = (self.a*np.sin(v))**2 + (self.b*np.cos(v))**2
      return np.array([[guu,guv],[gvu,gvv]])

    def normal(self,u,v):
      '''
      Note: this normal is well defined everywhere despite the fact that the poles
      are degenerated. This comes that in the computation cos(v) appears as a global
      factor, that cancels out in the normed vector.
      '''

      #den = np.sqrt((self.a*np.sin(v))**2 + (self.b*np.cos(v))**2 )
      nx = self.b*np.cos(u)*np.cos(v)
      ny = self.b*np.sin(u)*np.cos(v)
      nz = self.a*np.sin(v)

      nvec = np.array([nx,ny,nz])
      lennvec = np.linalg.norm(nvec)
      return nvec/lennvec


    def rotate_surface_vector_degenerate_basis(self, u, v, vectpq, angle):
        '''
        rotation in case of a degenerated covariant basis (eg. at the poles of a sphere.
        This class must be implemented by the daughter classes depending on their specificities
        Same procedure as for the sphere.
        '''

        #print("!!! Degenerated covariant basis detected ...")

        # At the poles, (u,v) = (u,+/-pi/2), u being the value of the azimuth
        # of the great circle passing at the pole that conducted to the pole
        # therefore, at the poles, turning means changing the value of u by the
        # turning angle!
        # The value of the parameter u is the value when arriving at the pole from previous
        # points on the trajectory. It therefore represents the heading of the turtle
        # Then turning by an angle  means adding an angle to this heading direction
        #print ("BEFORE: u,v = ", u,v,vectpq)
        u += angle
        v = np.pi/2 if v >=0 else -np.pi/2
        #print ("AFTER : u,v = ", u,v,vectpq)

        # In adition, the velocity needs be reinitiated in the new direction of u
        # FIXME: at speed one by default, but should be able to use any figure
        vectpqy = [u, np.pi/2, 0., 1.]

        # Note that u has been changed by this rotation in the retured vector
        return vectpqy

    def geodesic_eq(self,uvpq,s):
      u,v,p,q = uvpq

      X = self.S(u, v)  # current 3D point on the trajectory
      P1 = self.S(0, np.pi / 2)  # pole 1
      P2 = self.S(0, -np.pi / 2)  # pole 2

      den = (self.a * np.sin(v)) ** 2 + (self.b * np.cos(v)) ** 2

      # the following test check in the 3D space (and not in the parameter space)
      # whether the current point is close to either poles.
      if np.allclose(X, P1) or np.allclose(X, P2):  # np.isclose(np.sin(v), 1.): #(v = pi/2)
          # If yes slightly approximates the equation to avoid divergence at the pole
          # by pretending that p keeps constant while the point is in the narrow
          # degenerated region.
          # print("PI/2 REACHED ...", np.allclose(X, P1), np.allclose(X, P1))
          # makes the approximation that p does not change close to this very narrow region
          # this avoids the divergence of p at the poles

          # p and q are the next values for u,v. One must check that q stays between [-pi/2,pi/2]
          # print("angles = ", u, v, p, q)

          return [p, q, p,-((self.a**2-self.b**2)*np.cos(v)*np.sin(v)/den)*q**2 - ((self.a**2)*np.cos(v)*np.sin(v)/den)*p**2]
      else:  # standard equation
          return [p,q,2*np.tan(v)*p*q,-((self.a**2-self.b**2)*np.cos(v)*np.sin(v)/den)*q**2 - ((self.a**2)*np.cos(v)*np.sin(v)/den)*p**2]


class Torus(ParametricSurface):

    def __init__(self, R = 1.0, r = 0.2):
      super(Torus, self).__init__(umin=0,umax=2*np.pi,vmin=0.,vmax=2*np.pi)
      self.R = R # radius of the torus (center to medial circle)
      self.r = r # radius of the torus cylinder

    # Surface position vector
    def S(self,u,v):
      """ Returns the coordinates (x,y,z) of a position vector restricted to the torus surface
      as a function of the two surface coordinates (u,v)
      u = azimuth (u in [0,2Pi], counted from the x-axis, where u = 0)
      v = elevation (v in [0,2Pi] )
      """
      x = (self.R+self.r*np.cos(v))*np.cos(u)
      y = (self.R+self.r*np.cos(v))*np.sin(u)
      z = self.r*np.sin(v)
      return np.array([x,y,z])

    def Shift(self,u,v):
      """ Shit tensor (3,2 matrix) to transform the coordinates u,v in x,y,z
      It is derived from the partial derivatives of the surface equations wrt u,v
      (may be could be computed automatically from S(u,v) ... see that later)
      """
      xu = -(self.R+self.r*np.cos(v))*np.sin(u)
      yu =  (self.R+self.r*np.cos(v))*np.cos(u)
      zu = 0
      xv = -self.r*np.cos(u)*np.sin(v)
      yv = -self.r*np.sin(u)*np.sin(v)
      zv = self.r*np.cos(v)
      return np.array([[xu,xv],[yu,yv],[zu,zv]])

    def secondsuu(self,u,v):
      """
      second derivatives of the position vector at the surface
      """
      xuu = -(self.R+self.r*np.cos(v))*np.cos(u)
      yuu = -(self.R+self.r*np.cos(v))*np.sin(u)
      zuu = 0
      return np.array([xuu,yuu,zuu])

    def secondsuv(self,u,v):
      xuv =  self.r*np.sin(u)*np.sin(v)
      yuv = -self.r*np.cos(u)*np.sin(v)
      zuv = 0
      return np.array([xuv,yuv,zuv])

    def secondsvv(self,u,v):
      # CORRECTION:
      #xvv = -self.r*np.cos(u)*np.sin(v)
      xvv = -self.r*np.cos(u)*np.cos(v)
      #yvv = -self.r*np.sin(u)*np.sin(v)
      yvv = -self.r*np.sin(u)*np.cos(v)
      zvv = -self.r*np.sin(v)

      return np.array([xvv,yvv,zvv])

    def normal(self,u,v):
      nx = np.cos(u)*np.cos(v)
      ny = np.sin(u)*np.cos(v)
      nz = np.sin(v)

      nvec = np.array([nx,ny,nz])
      lennvec = np.linalg.norm(nvec)
      return nvec/lennvec


    def metric_tensor(self,u,v):
      guu = (self.R + self.r*np.cos(v))**2
      guv = gvu = 0.
      gvv = self.r**2
      return np.array([[guu,guv],[gvu,gvv]])

    # COMMENTER POUR TESTER L'EQUATION GEODESIC GENERIQUE (de la classe parametric surface et utilisant les symboles de RiemannChristoffel)
    def geodesic_eq(self,uvpq,s):
      u,v,p,q = uvpq

      # checks the equality of Riemmann Christoffel coefs -------
      ''' Uncomment to check if the different ways to compute RC symbols are consistent
      if False:
          Gamma = self.ChristoffelSymbols(u,v)
          print ("u,v=",u,v)
          coef010 = coef001 = -(self.r*np.sin(v)/(self.R+self.r*np.cos(v)))
          coef100 = (self.R+self.r*np.cos(v))*np.sin(v)/self.r
          if not np.isclose(Gamma[0,0,0],0.,rtol=1e-03, atol=1e-05): print("  Error: Gamma[0,0,0] = ", Gamma[0,0,0], "instead of: ", 0.)
          if not np.isclose(Gamma[0,0,1],coef001,rtol=1e-03, atol=1e-05):
              Gamma = self.ChristoffelSymbols(u,v,True)
              S_u, S_v = self.covariant_basis(u,v)
              ig = self.inverse_metric_tensor(u,v)
              Su = ig[0][0] * S_u + ig[0][1] * S_v
              Sv = ig[1][0] * S_u + ig[1][1] * S_v
              S_uv = self.secondsuv(u,v)
              print("  Error: Gamma[0,0,1] = ", Gamma[0,0,1], "instead of: ", coef001)
              print("  Su = ", Su)
              print("  S_uv = ", S_uv)
              print("  Su.S_uv = ", np.dot(Su,S_uv))
          if not np.isclose(Gamma[0,1,0],coef001,rtol=1e-03, atol=1e-05): print("  Error: Gamma[0,1,0] = ", Gamma[0,1,0], "instead of: ", coef001)
          if not np.isclose(Gamma[0,1,1],0.,rtol=1e-03, atol=1e-05): print("  Error: Gamma[0,1,1] = ", Gamma[0,1,1], "instead of: ", 0.)
          if not np.isclose(Gamma[1,0,0],coef100,rtol=1e-03, atol=1e-05): print("  Error: Gamma[1,0,0] = ", Gamma[1,0,0], "instead of: ", coef100)
          if not np.isclose(Gamma[1,0,1],0.,rtol=1e-03, atol=1e-05): print("  Error: Gamma[1,0,1] = ", Gamma[1,0,1], "instead of: ", 0.)
          if not np.isclose(Gamma[1,1,0],0.,rtol=1e-03, atol=1e-05): print("  Error: Gamma[1,1,0] = ", Gamma[1,1,0], "instead of: ", 0.)
          if not np.isclose(Gamma[1,1,1],0.,rtol=1e-03, atol=1e-05): print("  Error: Gamma[1,1,1] = ", Gamma[1,1,1], "instead of: ", 0.)
      # end test equality of coefs -------     
      '''

      return [p,q,2*(self.r*np.sin(v)/(self.R+self.r*np.cos(v)))*p*q,-((self.R+self.r*np.cos(v))*np.sin(v)/self.r)*p**2]


class Paraboloid(ParametricSurface):
    """
    u = radius of the circle (r)
    v = position on the circle (theta)
    """

    def __init__(self, radiusmax = 0.5):
      super(Paraboloid, self).__init__(umin=0,umax=radiusmax,vmin=0.,vmax=2*np.pi)

    def S(self,u,v):
      """ Returns the coordinates (x,y,z) of a position vector restricted to the paraboloid surface
      as a function of the two surface coordinates (u,v)
      u = radius (is not restrricted in principle)
      v = elevation (v in [-Pi/2,+Pi/2] )
      """
      x = u*np.cos(v)
      y = u*np.sin(v)
      z = u**2
      return np.array([x,y,z])

    # the Shift tensor may be wieved as the coordinates of the surface
    # covariant basis expressed in the ambiant basis (3x2) = 2 column vectors in 3D.
    def Shift(self,u,v):
      """ Shit tensor (3,2 matrix) to transform the coordinates u,v in x,y,z
      It is derived from the partial derivatives of the surface equations wrt u,v
      (may be could be computed automatically from S(u,v) ... see that later)
      """
      xu = np.cos(v)
      yu = np.sin(v)
      zu = 2*u
      xv = -u*np.sin(v)
      yv = u*np.cos(v)
      zv = 0

      return np.array([[xu,xv],[yu,yv],[zu,zv]])

    def secondsuu(self,u,v):
      """
      second derivatives of the position vector at the surface
      """
      xuu = 0
      yuu = 0
      zuu = 2
      return np.array([xuu,yuu,zuu])

    def secondsuv(self,u,v):
      xuv = -np.sin(v)
      yuv =  np.cos(v)
      zuv = 0
      return np.array([xuv,yuv,zuv])

    def secondsvv(self,u,v):
      xvv = -u*np.cos(v)
      yvv = -u*np.sin(v)
      zvv = 0

      return np.array([xvv,yvv,zvv])

    def metric_tensor(self,u,v):
      guu = 1 + 4 * u**2
      guv = gvu = 0.
      gvv = u**2
      return np.array([[guu,guv],[gvu,gvv]])

    def normal(self,u,v):
      nx = -2*(u**2)*np.cos(v)
      ny = -2*(u**2)*np.sin(v)
      nz = u

      nvec = np.array([nx,ny,nz])
      lennvec = np.linalg.norm(nvec)
      return nvec/lennvec

    def geodesic_eq(self,uvpq,s):
      u,v,p,q = uvpq
      return [p,q, -4*u*p**2+ u*q**2, -2*u*p*q]


class MonkeySaddle(ParametricSurface):

    def __init__(self, a = 1., n=3, umax = 1.):
      super(MonkeySaddle, self).__init__(umin=0,umax=umax,vmin=0.,vmax=2*np.pi)

      self.a = a # dilation factor
      self.n = n # dilation factor
      self.umin = 0
      self.umax = umax
      self.vmin = 0.
      self.vmax = 2*np.pi

    def S(self,u,v):
      """ Returns the coordinates (x,y,z) of a position vector restricted to the sphere surface
      as a function of the two surface coordinates (u,v)
      u = azimuth (u in [0,2Pi], counted from the x-axis, where u = 0)
      v = elevation (v in [-Pi/2,+Pi/2] )
      """
      x = self.a *u*np.cos(v)
      y = self.a *u*np.sin(v)
      z = self.a *(u**self.n)*np.cos(self.n*v)
      return np.array([x,y,z])

    # the Shift tensor may be wieved as the coordinates of the surface
    # covariant basis expressed in the ambiant basis (3x2) = 2 column vectors in 3D.
    def Shift(self,u,v):
      """ Shit tensor (3,2 matrix) to transform the coordinates u,v in x,y,z
      It is derived from the partial derivatives of the surface equations wrt u,v
      (may be could be computed automatically from S(u,v) ... see that later)
      """
      xu = self.a *np.cos(v)
      yu = self.a *np.sin(v)
      zu = self.n*self.a *(u**(self.n-1))*np.cos(self.n*v)
      xv = -self.a *u*np.sin(v)
      yv = self.a *u*np.cos(v)
      zv = - self.n *self.a *(u**self.n)*np.sin(self.n*v)

      return np.array([[xu,xv],[yu,yv],[zu,zv]])

    def secondsuu(self,u,v):
      """
      second derivatives of the position vector at the surface
      """
      xuu = 0.
      yuu = 0.
      zuu = self.n*(self.n-1)*self.a*(u**(self.n-2))*np.cos(self.n*v)
      return np.array([xuu,yuu,zuu])

    def secondsuv(self,u,v):
      xuv =  -self.a *np.sin(v)
      yuv =  self.a *np.cos(v)
      zuv =  -(self.n**2)*self.a*(u**(self.n-1))*np.sin(self.n*v)
      return np.array([xuv,yuv,zuv])

    def secondsvv(self,u,v):
      xvv = -self.a*u*np.cos(v)
      yvv = -self.a*u*np.sin(v)
      zvv = -(self.n**2)*self.a*(u**self.n)*np.cos(v)

      return np.array([xvv,yvv,zvv])

    '''
    TODO: not implemented (metric tensor is Ok, but inverse metric tensor is more complicated analytically
    def metric_tensor(self,u,v):
      guu = 
      guv = gvu = 
      gvv = 
      return np.array([[guu,guv],[gvu,gvv]])
    '''

    def normal(self,u,v):
      nx = -self.n*(self.a**2)*(u**self.n)*np.cos((self.n-1)*v)
      ny =  self.n*(self.a**2)*(u**self.n)*np.sin((self.n-1)*v)
      nz = (self.a**2)*u

      nvec = np.array([nx,ny,nz])
      lennvec = np.linalg.norm(nvec)
      return nvec/lennvec

"""
def nb_getSecondDerivativeUUAt(self, u,v):
    return self.getDerivativeAt(u,v,2,0)

def nb_getSecondDerivativeUVAt(self, u,v):
    return self.getDerivativeAt(u,v,1,1)

def nb_getSecondDerivativeVVAt(self, u,v):
    return self.getDerivativeAt(u,v,0,2)

from openalea.plantgl.all import NurbsPatch
NurbsPatch.getSecondDerivativeUUAt = nb_getSecondDerivativeUUAt
NurbsPatch.getSecondDerivativeUVAt = nb_getSecondDerivativeUVAt
NurbsPatch.getSecondDerivativeVVAt = nb_getSecondDerivativeVVAt
"""

class Patch(ParametricSurface):

    def __init__(self, patch, utoric = False, vtoric = False, STOP_AT_BOUNDARY_U = False, STOP_AT_BOUNDARY_V = False):
      self.patch = patch

      umin = min(self.patch.uknotList)
      umax = max(self.patch.uknotList)
      vmin = min(self.patch.vknotList)
      vmax = max(self.patch.vknotList)

      self.utoric = utoric
      self.vtoric = vtoric

      super(Patch, self).__init__(umin=umin, umax=umax, vmin=vmin, vmax=vmax, STOP_AT_BOUNDARY_U = STOP_AT_BOUNDARY_U, STOP_AT_BOUNDARY_V = STOP_AT_BOUNDARY_U)

    def normalizeuv(self, u, v):
      if self.utoric:
          u = self.umin + (u-self.umin) % (self.umax-self.umin) 
      else:
          u = min(self.umax,max(self.umin,u))
      if self.vtoric:
          v = self.vmin + (v-self.vmin) % (self.vmax-self.vmin) 
      else:
          v = min(self.vmax,max(self.vmin,v))
      return u,v

    def getPointAt(self, u,v):
      return self.patch.getPointAt(u,v)

    def getUTangentAt(self, u,v):
      return self.patch.getUTangentAt(u,v)

    def getVTangentAt(self, u,v):
      return self.patch.getVTangentAt(u,v)

    def getSecondDerivativeUUAt(self, u,v):
      return self.patch.getDerivativeAt(u,v,2,0)

    def getSecondDerivativeUVAt(self, u,v):
      return self.patch.getDerivativeAt(u,v,1,1)

    def getSecondDerivativeVVAt(self, u,v):
      return self.patch.getDerivativeAt(u,v,0,2)

    # Surface position vector
    # uvpq is an 1x4 array of reals where:
    # - u,v are the coordinates on the surface of the moving point
    # - p,q are the coordinates of the initial vector corresponding to
    # the initial direction in the local covariant basis (at point [u,v]
    def S(self,u,v):
      """ Returns the coordinates (x,y,z) of a position vector restricted to the sphere surface
      as a function of the two surface coordinates (u,v)
      u = azimuth (u in [0,2Pi], counted from the x-axis, where u = 0)
      v = elevation (v in [-Pi/2,+Pi/2] )
      """
      u,v = self.normalizeuv(u,v)
      return np.array(self.getPointAt(u,v))

    # the Shift tensor may be wieved as the coordinates of the surface
    # covariant basis expressed in the ambiant basis (3x2) = 2 column vectors in 3D.
    def Shift(self,u,v):
      """ Shit tensor (3,2 matrix) to transform the coordinates u,v in x,y,z
      It is derived from the partial derivatives of the surface equations wrt u,v
      (may be could be computed automatically from S(u,v) ... see that later)
      """
      # getDerivativeAt(u,v,nu,nv) returns the dérivative at point (u,v)
      # where nu, and nv specifies the depth of the derivative (number of time the derivative is applied)
      # print("S_u: ", self.patch.getDerivativeAt(u,v,1,0), " S_v: ", self.patch.getDerivativeAt(u,v,0,1))
      u,v = self.normalizeuv(u,v)
      S_u = np.array(self.getUTangentAt(u,v)) #
      S_v = np.array(self.getVTangentAt(u,v))

      return np.array([[S_u[0],S_v[0]],[S_u[1],S_v[1]],[S_u[2],S_v[2]]])

    def secondsuu(self,u,v):
      """
      second derivatives of the position vector at the surface
      """
      u,v = self.normalizeuv(u,v)
      S_u = self.getSecondDerivativeUUAt(u,v)
      return np.array(S_u)

    def secondsuv(self,u,v):
      u,v = self.normalizeuv(u,v)
      S_u = self.getSecondDerivativeUVAt(u,v)
      return np.array(S_u)

    def secondsvv(self,u,v):
      u,v = self.normalizeuv(u,v)
      S_u = self.getSecondDerivativeVVAt(u,v)
      return np.array(S_u)


def to_nurbs_python(sh):
    from geomdl import NURBS, BSpline
    surf = NURBS.Surface()
    surf.degree_u = sh.udegree
    surf.degree_v = sh.vdegree

    npctrls = np.array(sh.ctrlPointMatrix)
    shape = npctrls.shape
    npctrls = np.reshape(npctrls,(shape[0]*shape[1],shape[2]))
    npctrls[:,0] *= npctrls[:,3]
    npctrls[:,1] *= npctrls[:,3]
    npctrls[:,2] *= npctrls[:,3]
    surf.set_ctrlpts(npctrls[:,:].tolist(), shape[0], shape[1])
    surf.knotvector_u = list(sh.uknotList)
    surf.knotvector_v = list(sh.vknotList)

    surf.evaluate()

    return surf


class GeomLibPatch(ParametricSurface):

    def __init__(self, patch):
      self.patch = patch
      self.nurbssurf = to_nurbs_python(patch)

      umin = min(self.nurbssurf.knotvector_u)
      umax = max(self.nurbssurf.knotvector_u)
      vmin = min(self.nurbssurf.knotvector_v)
      vmax = max(self.nurbssurf.knotvector_v)

      super(Patch, self).__init__(umin=umin, umax=umax, vmin=vmin, vmax=vmax)


    def S(self,u,v):
      """ Returns the coordinates (x,y,z) of a position vector restricted to the sphere surface
      as a function of the two surface coordinates (u,v)
      u = azimuth (u in [0,2Pi], counted from the x-axis, where u = 0)
      v = elevation (v in [-Pi/2,+Pi/2] )
      """
      u = min(self.umax,max(self.umin,u))
      v = min(self.vmax,max(self.vmin,v))
      p = self.nurbssurf.evaluate_single((u,v))
      return np.array(p)

    # the Shift tensor may be wieved as the coordinates of the surface
    # covariant basis expressed in the ambiant basis (3x2) = 2 column vectors in 3D.
    def Shift(self,u,v):
      """ Shit tensor (3,2 matrix) to transform the coordinates u,v in x,y,z
      It is derived from the partial derivatives of the surface equations wrt u,v
      (may be could be computed automatically from S(u,v) ... see that later)
      """
      # getDerivativeAt(u,v,nu,nv) returns the dérivative at point (u,v)
      # where nu, and nv specifies the depth of the derivative (number of time the derivative is applied)
      # print("S_u: ", self.patch.getDerivativeAt(u,v,1,0), " S_v: ", self.patch.getDerivativeAt(u,v,0,1))
      u = min(self.umax,max(self.umin,u))
      v = min(self.vmax,max(self.vmin,v))
      skl = derivatives(self.nurbssurf, u, v, 1)
      S_u = np.array(skl[1][0]) #
      S_v = np.array(skl[0][1])

      return np.array([[S_u[0],S_v[0]],[S_u[1],S_v[1]],[S_u[2],S_v[2]]])

    def secondsuu(self,u,v):
      """
      second derivatives of the position vector at the surface
      """
      u = min(self.umax,max(self.umin,u))
      v = min(self.vmax,max(self.vmin,v))
      skl = derivatives(self.nurbssurf, u, v, 2)
      return np.array(skl[2][0])

    def secondsuv(self,u,v):
      u = min(self.umax,max(self.umin,u))
      v = min(self.vmax,max(self.vmin,v))
      skl = derivatives(self.nurbssurf, u, v, 1)
      return np.array(skl[1][1])

    def secondsvv(self,u,v):
      u = min(self.umax,max(self.umin,u))
      v = min(self.vmax,max(self.vmin,v))
      skl = derivatives(self.nurbssurf, u, v, 2)
      return np.array(skl[0][2])

from openalea.plantgl.math import Matrix3, norm

def m3_tolist(self):
    return [self.getColumn(v) for v in range(3)]

Matrix3.tolist = m3_tolist

class ExtrusionSurface(Patch):

    def __init__(self, extrusion, STOP_AT_BOUNDARY_U = False, STOP_AT_BOUNDARY_V = False):
      extrusion.uknotList = [extrusion.axis.firstKnot,extrusion.axis.lastKnot]
      cs = extrusion.crossSection
      extrusion.vknotList = [cs.firstKnot,cs.lastKnot]
      self.vtoric = (norm(cs.getPointAt(cs.firstKnot)-cs.getPointAt(cs.lastKnot)) < 1e-5)
      self.framecache = {}
      self.ducache = (extrusion.axis.lastKnot-extrusion.axis.firstKnot) / extrusion.axis.stride

      super(ExtrusionSurface, self).__init__(extrusion, vtoric=self.vtoric, STOP_AT_BOUNDARY_U = STOP_AT_BOUNDARY_U, STOP_AT_BOUNDARY_V = STOP_AT_BOUNDARY_V)

      self.build_cache()

    def build_cache(self):
        frame = self.patch.getFrameAt(self.umin)
        self.framecache[0] = frame.tolist()
        prevu = self.umin
        for i,u in enumerate(np.arange(self.umin+self.ducache, self.umax+self.ducache/2, self.ducache), 1):
            frame = self.patch.getNextFrameAt(prevu, frame, u-prevu)
            self.framecache[i] = frame.tolist()
            prevu = u


    def getFrame(self,u):
        assert self.umin <= u <= self.umax
        #return self.patch.getFrameAt(u)

        div, mod = divmod(u-self.umin, self.ducache)
        approxu = (div*self.ducache) + self.umin

        approxframe = Matrix3(*self.framecache[int(div)])

        if mod < 1e-5:
            res = approxframe
        else:
            res = self.patch.getNextFrameAt(approxu, approxframe, u-approxu)
        #print(self.patch.getFrameAt(u))
        #print(res)
        return res

    def getPointAt(self, u,v):
      return self.patch.getPointAt(u,v, self.getFrame(u))

    def getUTangentAt(self, u,v):
      return self.patch.getUTangentAt(u,v, self.getFrame(u))

    def getVTangentAt(self, u,v):
      return self.patch.getVTangentAt(u,v, self.getFrame(u))

    def getSecondDerivativeUUAt(self, u,v):
      return self.patch.getSecondDerivativeUUAt(u,v, self.getFrame(u))

    def getSecondDerivativeUVAt(self, u,v):
      return self.patch.getSecondDerivativeUVAt(u,v, self.getFrame(u))

    def getSecondDerivativeVVAt(self, u,v):
      return self.patch.getSecondDerivativeVVAt(u,v, self.getFrame(u))



class Revolution(ParametricSurface):
    """
    u = theta - azimuthal position around the symmetry axis
    v = z - altitude on the symmetry axis

    r is a function of z (i.e. v) that defines in 3D the radius of the point at altitude z
    The first and second derivatives are computed automatically
    """

    def __init__(self, rfunc, args = [], zmin = -2*np.pi, zmax = 2*np.pi):
      super(Revolution, self).__init__(umin=0,umax=2*np.pi,vmin=zmin,vmax=zmax,STOP_AT_BOUNDARY_V=True) # only keep z in the domain

      #print('args = ', args)
      #print('rfunc(2.,args)', rfunc(2.,args))
      f = gen_func(rfunc, args)
      df = gen_prime_deriv(rfunc, args)
      ddf = gen_second_deriv(rfunc, args)
      #print(rfunc(2.,args), df(0.1), ddf(0.1))

      self.rfunc = f        # care this are functions
      self.args = args
      self.rprime = df      # first derivative of the radius with respect to z
      self.rsecond = ddf    # second derivative of the radius with respect to z

    def S(self,u,v):
      """ Returns the coordinates (x,y,z) of a position vector restricted to the paraboloid surface
      as a function of the two surface coordinates (u,v)
      u = radius (is not restricted in principle)
      v = elevation (v in [-Pi/2,+Pi/2] )
      """
      x = self.rfunc(v)*np.cos(u)
      y = self.rfunc(v)*np.sin(u)
      z = v
      return np.array([x,y,z])

    # the Shift tensor may be wieved as the coordinates of the surface
    # covariant basis expressed in the ambiant basis (3x2) = 2 column vectors in 3D.
    def Shift(self,u,v):
      """ Shit tensor (3,2 matrix) to transform the coordinates u,v in x,y,z
      It is derived from the partial derivatives of the surface equations wrt u,v
      (may be could be computed automatically from S(u,v) ... see that later)
      """
      xu = -self.rfunc(v)*np.sin(u)
      yu = self.rfunc(v)*np.cos(u)
      zu = 0
      xv = self.rprime(v)*np.cos(u)
      yv = self.rprime(v)*np.sin(u)
      zv = 1

      return np.array([[xu,xv],[yu,yv],[zu,zv]])

    def secondsuu(self,u,v):
      """
      second derivatives of the position vector at the surface
      """
      xuu = -self.rfunc(v)*np.cos(u)
      yuu = -self.rfunc(v)*np.sin(u)
      zuu = 0
      return np.array([xuu,yuu,zuu])

    def secondsuv(self,u,v):
      xuv = -self.rprime(v)*np.sin(u)
      yuv =  self.rprime(v)*np.cos(u)
      zuv = 0
      return np.array([xuv,yuv,zuv])

    def secondsvv(self,u,v):
      xvv = self.rsecond(v)*np.cos(u)
      yvv = self.rsecond(v)*np.sin(u)
      zvv = 0
      return np.array([xvv,yvv,zvv])

    def metric_tensor(self,u,v):
      guu = self.rfunc(v)**2
      guv = gvu = 0.
      gvv = 1+self.rprime(v)**2
      return np.array([[guu,guv],[gvu,gvv]])

    def normal(self,u,v):
      '''
      Note: the normal is always defined for all u and v.
      '''
      factor = 1/(np.sqrt(1+self.rprime(v)**2))
      nx = factor * np.cos(u)
      ny = factor * np.sin(u)
      nz = - factor * self.rprime(v)

      nvec = np.array([nx,ny,nz])
      lennvec = np.linalg.norm(nvec)
      return nvec/lennvec

    #def geodesic_eq(self,uvpq,s):
    #  u,v,p,q = uvpq
    #  return [p,q, -4*u*p**2+ u*q**2, -2*u*p*q]

    def ChristoffelSymbols(self, u, v, printflag = False):
        """
        defined as the scalar product of S^a . dS_b / dS^c, with a,b,c in {u,v}
        """
        # Christoffel symbols (RCS) that is an array of 2x2x2 = 8 numbers
        # Gamma^a_bc  = RCS[a,b,c]
        den = 1+self.rprime(v)**2
        RCS = np.zeros((2,2,2))
        RCS[0,0,0] = 0.
        RCS[1,0,0] = -self.rfunc(v)*self.rprime(v)/den
        RCS[0,1,0] = self.rprime(v)/self.rfunc(v)
        RCS[1,1,0] = 0.
        RCS[0,0,1] = RCS[0,1,0]
        RCS[1,0,1] = RCS[1,1,0]
        RCS[0,1,1] = 0.
        RCS[1,1,1] = self.rprime(v)*self.rsecond(v)/den

        return RCS

    def geodesic_eq(self, uvpq, s):
        u, v, p, q = uvpq
        r = self.rfunc(v)
        r1 = self.rprime(v)
        r2 = self.rsecond(v)
        den = 1 + r1 ** 2

        # Are we on a degenerated point (Gamma coeff diverging)
        if np.isclose(r, 0.):
            # If yes slightly approximates the equation to avoid divergence at the pole
            # by pretending that p keeps constant while the point is in the narrow
            # degenerated region.

            #print("Surface of revolution: Degenerated point detected ...")

            # makes the approximation that p does not change close to this very narrow region this avoids the
            # divergence of p at the poles

            return [p, q, p, -r1 * r2 * (q ** 2) / den + r * r1 * (p ** 2) / den]
        else:  # standard equation
            return [p, q, -2 * p * q * r1 / r, -r1 * r2 * (q ** 2) / den + r * r1 * (p ** 2) / den]


def tractrix(x, R=10):
    if math.isclose(x, 0.0):
        return math.inf
    else:
        try:
            b = math.sqrt(R ** 2 - x ** 2)
            res = R * math.log((R + b) / x) - b
        except ValueError:
            print("tractrix curve: bad domain for x=", x, " (should be 0 < x <=", R, ")")
        else:
            return res

class ChineseHat(Revolution):
    '''
    This surface is made with a tractrix revoluting around its x-axis (instead of y as in the PseudoSphere)
    '''
    def __init__(self, R, zmin = 0.1, zmax = 0.99):
        super(ChineseHat,self).__init__(tractrix, args = (R), zmin = zmin, zmax = zmax)


#################################
# Other surface tools
#################################

# For plotting any surface defined by an explicit equation S(u,v) with Quads
##############################################################################

def QuadifySurfEquation(surface,umin=0,umax=1,vmin=0,vmax=1,Du=0.01,Dv=0.01):
    '''
    Computes a quad representation of the surface defined as surface(u,v) --> scalar
    u is in [umax, umin], varies by steps of size Du
    v is in [vmax, vmin], varies by steps of size Dv
    '''

    # arange does not include the last bound if equal
    # --> append this value in both arrays

    #print("QUADIFY", umin,umax,vmin,vmax)
    ilist = np.arange(umin,umax,Du)
    #print("ilist before = ", ilist)
    ilist = np.append(ilist, umax) # add the last bound as it is not done by arange
    #print("ilist after  = ", ilist)

    jlist = np.arange(vmin,vmax,Dv)
    #print("jlist before = ", jlist)
    jlist = np.append(jlist, vmax) # case of a periodic list
    #print("jlist after  = ", jlist)

    M = len(ilist)
    N = len(jlist)

    uvList = [(i,j) for i in ilist for j in jlist]
    # 1D list containing the 3D points of the grid made by i,j values
    # (j varies quicker than i): kth point in the list corresponds
    # to point (i,j) such that k = i*N+j
    grid3Dpoints = [surface(u,v) for u,v in uvList]

    # Constructs the list of quad indexes pointing to the 3D points
    # in the previous list
    quadindexlist = [(N * i + j - N - 1, N * i + j - 1, N * i + j, N * i + j - N)
                     for i in range(1, M) for j in range(1, N)]

    '''
    if not REVERSE_NORMALS :
        quadindexlist = [ (N*i+j-N-1,N*i+j-1,N*i+j,N*i+j-N)
            for i in range(1,M) for j in range(1,N)]
    else:
        quadindexlist = [ (N*i+j-N-1,N*i+j-N,N*i+j,N*i+j-1)
            for i in range(1,M) for j in range(1,N)]
    '''

    return grid3Dpoints, quadindexlist, uvList

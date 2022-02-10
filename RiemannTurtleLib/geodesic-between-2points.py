"""
    Computing the Geodesics between two points in a Riemannian space.

    Author: C. Godin, Inria
    Date: Jan. 2021-2022
    Lab: RDP ENS de Lyon, Mosaic Inria Team

    Implementation of the algorithm described in on (Makeawa, 1996)

"""

import numpy as np

#from scipy.linalg import solve_banded # works for symmetric or hermitian matrices

from sparse.linalg import spsolve      # sparse matrix solver


# A curve on a parametric surface defined explicitly
#
# notations with respect to paper Maekawa 1996:
# - m --> K (nb points of the curve)
# - points are indexed from 0 to K-1 instead of 1 to K (python compliance)
class ExplicitCurveByEndPoints:

    def __init__(self, surf, K, A, B, X):
      self.surf = surf  # Parametric surface on which the curve runs
      self.K = K        # nb of points of the curve (including extremities)
      self.A = A        # Origin of the curve = (uA,vA)
      self.B = B        # End of the curve = (ub,vb)

      self.X = X        # array of dim K containing [uk,vk,pk,qk] for each curve point


def InitialCurve(surf,K, A, B):
    # compute a first series of K uvpg values from A to B

    X = np.zeros(K*4).reshape(K,4) # create a matrix with K lines of 4 values each
    for k in range(K):


    intial_curve = ExplicitCurveByEndPoints(surf, A, B, X)
    return

def h(s, k):
    ''' s = array of curvilinear abscissa
    returns the kth distance between the consecutive points
    - precondition: 0 < k <= m
    '''
    return s[k] - s[k-1]

def omega(k, j, Xk, RCSk):
    '''
    Xk is the kth state vector [u,v,p,q] on the current trajectory
    RCSk are the kth Christoffel symbols on the current trajectory
    '''
    pk = Xk[2]
    qk = Xk[3]
    a = RCSk[j,0,0]
    b = RCSk[j,0,1]
    return pk*a+qk*b

def theta(k, j, Xk, RCSk):
    '''
    Xk, Xkp are the kth and (k-1)th state vector [u,v,p,q] on the current trajectory
    RCSk are the kth Christoffel symbols on the current trajectory
    '''
    pk = Xk[2]
    qk = Xk[3]
    a = RCSk[j,0,1]
    b = RCSk[j,1,1]
    return pk*a+qk*b

def A(s, k, j, Xk, RCSk):
    '''

    '''
    A0 = np.array([-h(s, k)/2.,-1,0]))
    A1 = np.array([0,-h(s, k)/2.,0,-1])
    A2 = np.array([h(s, k)*omega(k-1,0)-1, h(s, k)*theta(k-1,0), 0, 0])
    A3 = np.array([h(s, k)*omega(k-1,1)  , h(s, k)*theta(k-1,1)-1, 0, 0])

    return np.array([A0,A1,A2,A3])


def B(s, k, j, Xk, RCSk):
    '''

    '''
    B0 = np.array([-h(s, k) / 2., 1, 0])
    B1 = np.array([0, -h(s, k) / 2., 0, 1])
    B2 = np.array([h(s, k) * omega(k, 0) + 1, h(s, k) * theta(k, 0), 0, 0])
    B3 = np.array([h(s, k) * omega(k, 1),     h(s, k) * theta(k, 1) + 1, 0, 0])

    return np.array([B0,B1,B2,B3])

def Bfirst(s, k, j, Xk, RCSk):
    '''
    corresponds to B0
    '''
    B0 = np.array([1,0,0,0])
    B1 = np.array([0,1,0,0])

    return np.array([B0,B1])

def Aswapped(s, k, j, Xk, RCSk)):

def Bswapped(s, k, j, Xk, RCSk)):


def Alast(s, k, j, Xk, RCSk):
    '''
    corresponds to Am
    '''
    A0 = np.array([0,0,1,0])
    A1 = np.array([0,0,0,1])

    return np.array([A0,A1])


def build_jacobian_csc(m):
    '''
    Builds the jacobian as a sparse matrix in csc format (list of (row,col) indexes of non-null entries)
    '''

    row = [0,1]  # list of row indexes of non-null elements
    col = [0,1]  # corresponding list of row indexes of non-null elements.
    dat = [1,1]  # data array contains non-null elements of the jacobian matrix (corresponds to blocks Ak and Bk)
                 # The lists are initialized by non-null elements of matrix B0

    # fills the jacobian matrix with non-null elements from Ak and Bk k = 1 .. m-1
    for k in range(1,m):
        # first compute the Ak and Bk+1 matrices
        Ak = Aswapped(s, k, Xk, RCSk)
        Bkp1 = Bswapped(s, k+1, Xk, RCSk)
        for j in range(4):
            # indexes present in the blocks Ak, Bk:
            h  = 4*(k-1) + j + 2 # line index in the jacobian
            c  = 4*(k-1) + j     # col index is non-null
            c2 = 4*(k) + j       # second column on the same line that is also non-null for this j

            # Now fills in the correct data
            d  = Ak[j]
            d2 = Bkp1[j]

            # add two entries on line h corresponding to non-null j entries in both Ak and Bk
            row.append(h)
            col.append(c)
            dat.append(d)

            row.append(h)
            col.append(c2)
            dat.append(d2)

    # in the end add the 2 non-null elements of array Am
    row.append(4*m-4)
    col.append(4*m-2)
    dat.append(1)

    row.append(4*m-3)
    col.append(4*m-1)
    dat.append(1)

    return csc_matrix((dat, (row, col)), shape=(4*m, 4*m))


def compute_residual_vec(res_vec):
    '''
    Computes the path increment on the current path to lead to the geodesic path
    res_mat = matrix of residuals at step k
    res_vec = column vector of residuals
    '''

    # 1. Compute the set of matrices A_swapped_k, B_swapped_k

    A_array = [Aswapped(...) for k in range(1,m)]
    B_array = [Bswapped(...) for k in range(1,m)]

    # Computes the residuals jacobian matrix from ... in a csc format
    j_mat = build_jacobian_csc()

    # solves:    delta_X . jmat = res_vec    (delta_X is the unknown var)
    delta_X = spsolve(j_mat, res_vec)

    return delta_X

def standardized_norm(v, MU = 1., MV = 1., MP = 10., MQ = 10.):
    '''
    v is assumed to be a swapped vector over the indexes k =  1 .. m-1 (note that 0 is missing)
    MU, MV, MP and MQ are factors corresponding to the magnitude of variables delta_u,delta_v,delta_p,delta_q
    '''
    dim = v.shape[0] # dimension of v should be one column of 4*m coordinates
    assert(len(v.shape == 1))

    m, r = divmod(dim,4)
    assert(r == 0)   # the vector dim should be a multiple of 4

    # the sum is initialized with the first 4 terms in the order u,v,p,q (not swapped)
    sum = abs(v[0]/MU) + abs(v[1]/MV) + abs(v[2]/MP) + abs(v[3]/MQ)

    for k in range(1,m): # now taking into account the fact that terms are swapped in the order p,q,u,v
        sum += abs(v[4*k]) / MP
        sum += abs(v[4*k+1]) / MQ
        sum += abs(v[4*k+2]) / MU
        sum += abs(v[4*k+3]) / MV

    return sum









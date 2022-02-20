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



def omega(pk,qk,CSj00,CSj01):

    return pk*CSj00+qk*CSj01


def theta(pk,qk,CSj01,CSj11):

    return pk*CSj01+qk*CSj11

'''
def omega(k, j, Xk, CSk):
    
    Xk is the kth state vector [u,v,p,q] on the current trajectory
    RCSk are the kth Christoffel symbols on the current trajectory
    
    pk = Xk[2]
    qk = Xk[3]
    a = CSk[j,0,0]
    b = CSk[j,0,1]
    return pk*a+qk*b

def theta(k, j, Xk, CSk):
    
    Xk, Xkp are the kth and (k-1)th state vector [u,v,p,q] on the current trajectory
    RCSk are the kth Christoffel symbols on the current trajectory
    
    pk = Xk[2]
    qk = Xk[3]
    a = CSk[j,0,1]
    b = CSk[j,1,1]
    return pk*a+qk*b

def A_swapped(s, k, j, Xk, CSk):

    A0 = np.array([-h(s, k)/2.,-1,0]))
    A1 = np.array([0,-h(s, k)/2.,0,-1])
    A2 = np.array([h(s, k)*omega(k-1,0)-1, h(s, k)*theta(k-1,0), 0, 0])
    A3 = np.array([h(s, k)*omega(k-1,1)  , h(s, k)*theta(k-1,1)-1, 0, 0])

    return np.array([A0,A1,A2,A3])

def B_swapped(s, k, j, Xk, CSk):

    B0 = np.array([-h(s, k) / 2., 1, 0])
    B1 = np.array([0, -h(s, k) / 2., 0, 1])
    B2 = np.array([h(s, k) * omega(k, 0) + 1, h(s, k) * theta(k, 0), 0, 0])
    B3 = np.array([h(s, k) * omega(k, 1),     h(s, k) * theta(k, 1) + 1, 0, 0])

    return np.array([B0,B1,B2,B3])
'''

def A_swapped(h,omega0,omega1,theta0,theta1):

    A0 = np.array([-h/2.,-1,0]))
    A1 = np.array([0,-h/2.,0,-1])
    A2 = np.array([h*omega0-1, h*theta0, 0, 0])
    A3 = np.array([h*omega1  , h*theta1-1, 0, 0])

    return np.array([A2,A3,A0,A1])


def B_swapped(h,omega0,omega1,theta0,theta1):

    B0 = np.array([-h/2.,1,0]))
    B1 = np.array([0,-h/2.,0,1])
    B2 = np.array([h*omega0+1, h*theta0, 0, 0])
    B3 = np.array([h*omega1  , h*theta1+1, 0, 0])

    return np.array([B2,B3,B0,B1])


def Bfirst_swapped():
    '''
    corresponds to B0 (not swapped, or with a swap = identity)
    '''
    B0 = np.array([1,0,0,0])
    B1 = np.array([0,1,0,0])

    return np.array([B0,B1])

# VERIFIER CE QUE FAIT LE SWAP LA DESSUS:
def Alast_swapped():
    '''
    corresponds to Am swapped
    '''
    A0 = np.array([0,0,1,0])
    A1 = np.array([0,0,0,1])

    return np.array([A0,A1])


def compute_residual_vec(space, X_swapped, delta_s):
    '''
    Compute the residual of the newton equation (function that should become 0)

    - X is the state of the current path: a 4*m vector of the form
    [u0,v0,p0,q0,u1,v1,p1,q1, ..., u_4m-1,v_4m-1,p_4m-1,q_4m-1]
    - delta_s is an array of dim (m-1) of approximated distances ds_k between
    consecutive points P_k, P_k+1 contained in X
    [ds_0,ds_1,..., ds_m-2], where ds_k = |S(uk,vk)-S(uk-1,vk-1)|
    '''
    m = len(X)/4

    G = np.zeros(4*m)
    R = np.zeros(4*m)

    # Initialization of the G array for k = 0.
    # The first 2 components of X_swapped are not swapped in the swapped vector X_swapped,
    # (hence treated separately here).
    uvpq = (X_swapped[0], X_swapped[1], X_swapped[2], X_swapped[3])
    new_uvpq = space.geodesic_eq(uvpq, 0)
    G[0],G[1],G[2],G[3] = new_uvpq

    # Initialization of the R array for k = 0
    R[0] = R[1] = 0.

    # then compute the rest of the vectors G (derivatives, used internally) and R (residuals)
    for k in range(1,m): # k = 1,.., m-1
        i = 4*k
        pquv = (X_swapped[i],X_swapped[i+1],X_swapped[i+2],X_swapped[i+3])
        # unswap before calling geodesic
        uvpq = [pquv[2],pquv[3],pquv[0],pquv[1]]
        # Compute the discretization of G using the geodesic equations
        # Note: the second argument is not used by the function geodesic_eq
        # (0 here after is not used by the function but required in the signature).
        new_uvpq = space.geodesic_eq(uvpq, 0)
        # swap back the returned uvpq value to use them in the swapped vectors G and R
        new_pquv = [new_uvpq[2],new_uvpq[3],new_uvpq[0],new_uvpq[1]]
        for h in range(4):
            j = i+h
            # ex: k = 1, h = 0, => j = 4, j-2 = 2;
            # ex: k = 2, h = 0, => j = 8, j-2 = 6
            G[j] = new_pquv[h]
            R[j-2] = (X[j] - X[j - 4]) / delta_s[k-1] - 0.5*(G[j] + G[j - 4])

    # R has dimension 4*m, according to the loop above, two values are missing
    # They corrrespond to the boundary conditions at the target point
    R[4*m-2] = R[4*m-1] = 0.

    return R


def build_jacobian_swapped_csc(space, X_swapped, delta_s):
    '''
    Builds the jacobian as a sparse matrix in csc format (list of (row,col) indexes of non-null entries)
    The jacobian matrix is computed with swapped columns to optimize pivot algotithm
    as follows:
    for k = 1..m-1,
      - column (4k-3) is interchanged with column (4k-1)
      - column (4k-2) is interchanged with column (4k)


    '''

    # To state that elements (0,0) and (1,1) of Jacobian matrix are non null:
    row = [0,1]  # list of row indexes of non-null elements
    col = [0,1]  # corresponding list of row indexes of non-null elements.
    # Then the values of the two preceding non-null elements are 1, 1
    dat = [1,1]  # data array contains non-null elements of the jacobian matrix (corresponds to blocks Ak and Bk)
                 # The lists are initialized by non-null elements of matrix B0

    # fills the jacobian matrix with non-null elements from Ak and Bk k = 1 .. m-1
    for k in range(1,m):
        i = 4*k

        # Compute the Christoffel Symbols at point k and k-1
        uk,vk = X_swapped[i+2], X_swapped[i+3]
        # Compute uk,vk at point k and k-1
        if k == 1:
            uk1,vk1 = X_swapped[i-4], X_swapped[i-3] # at previous point (k = 0), components are not swapped
        else:
            uk1,vk1 = X_swapped[i-2], X_swapped[i-1]

        # Compute the pk,qk at k and k-1
        pk, qk = X_swapped[i], X_swapped[i+1]
        if k == 1:
            pk1, qk1 = X_swapped[i-2], X_swapped[i-1]
        else:
            pk1, qk1 = X_swapped[i - 4], X_swapped[i - 3]

        # Compute the Christoffel Symbols at point k and k-1
        CSk = space.ChristoffelSymbols(uk, vk)
        CSk1 = space.ChristoffelSymbols(uk1, vk1)

        h = delta_s[k-1] # delta_s stores |s_k+1 - s_k| at index k => here we have h = |s_k - s_k-1|

        # A l'appel pour A
        omega0 = omega(pk1,qk1,CSk1[0,0,0],CSk1[0,0,1])
        omega1 = omega(pk1,qk1,CSk1[1,0,0],CSk1[1,0,1])
        theta0 = theta(pk1,qk1,CSk1[0,1,0],CSk1[0,1,1])
        theta1 = theta(pk1,qk1,CSk1[1,1,0],CSk1[1,1,1])

        # first compute the Ak and Bk+1 matrices
        Ak_swapped = A_swapped(h,omega0,omega1,theta0,theta1)

        # A l'appel pour B
        omega0 = omega(pk,qk,CSk[0,0,0],CSk[0,0,1])
        omega1 = omega(pk,qk,CSk[1,0,0],CSk[1,0,1])
        theta0 = theta(pk,qk,CSk[0,1,0],CSk[0,1,1])
        theta1 = theta(pk,qk,CSk[1,1,0],CSk[1,1,1])

        Bk_swapped = B_swapped(h,omega0,omega1,theta0,theta1)

        for j in range(4):
            # indexes present in the blocks Ak, Bk:
            h  = 4*(k-1) + j + 2 # line index in the jacobian
            c  = 4*(k-1) + j     # col index is non-null
            c2 = 4*(k) + j       # second column on the same line that is also non-null for this j

            # Now fills in the correct data (swapped columns)
            d  = Ak_swapped[j]
            d2 = Bk_swapped[j]

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

def swap_columns(X):

    Xswapped = np.zeros(len(X))

    # First quadruplet is not swapped
    Xswapped[0] = X[0]
    Xswapped[1] = X[1]
    Xswapped[2] = X[2]
    Xswapped[3] = X[3]

    # then the rest of the vector
    for k in range(1, m):
        Xswapped[4 * k]     = X[4 * k + 2]
        Xswapped[4 * k + 1] = X[4 * k + 3]
        Xswapped[4 * k + 2] = X[4 * k]
        Xswapped[4 * k + 3] = X[4 * k + 1]

    return Xswapped

def compute_deltaX(space, X_swapped, R_swapped, delta_s):
    '''
    Computes the path increment on the current path to lead to the geodesic path
    X_swapped = matrix of residuals at step k
    res_vec_swapped = column vector of residuals
    '''

    # Computes the residuals' jacobian matrix from Christoffel symbols and X (swapped) in a csc format
    # matrix J is a (4*m,4*m) matrix
    # note that the Jacobian has its columns already swapped to optimize matrix inversion

    J_mat_swapped = build_jacobian_swapped_csc(space, X_swapped, delta_s)

    # solves:    delta_X_swapped . J_mat_swapped = R_swapped (delta_X_swapped is the unknown var)
    # Note that this inversion is carried out directly on swapped quantities
    delta_X_swapped = spsolve(J_mat_swapped, R_swapped)

    return delta_X_swapped

def standardized_L1norm(v, MU = 1., MV = 1., MP = 10., MQ = 10.):
    '''
    v is assumed to be a swapped vector over the indexes k =  1 .. m-1 (note that 0 is missing)
    MU, MV, MP and MQ are factors corresponding to the magnitude of variables delta_u,delta_v,delta_p,delta_q
    '''
    dim = v.shape[0] # dimension of v should be one column of 4*m coordinates
    assert(len(v.shape == 1))

    m, r = divmod(dim,4)
    assert(r == 0)   # the vector dim should be a multiple of 4

    sum = 0.
    for k in range(0,m): # now taking into account the fact that terms are swapped in the order p,q,u,v
        sum += abs(v[4*k]) / MU
        sum += abs(v[4*k+1]) / MV
        sum += abs(v[4*k+2]) / MP
        sum += abs(v[4*k+3]) / MQ

    return sum

def standardized_L1norm_swapped(v, MU = 1., MV = 1., MP = 10., MQ = 10.):
    '''
    Swapped form of the norm.
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









"""
    Computing the Geodesics between two points in a Riemannian space.

    Author: C. Godin, Inria
    Date: Jan. 2021-2022
    Lab: RDP ENS de Lyon, Mosaic Inria Team

    Implementation of the algorithm described in on (Makeawa, 1996)

"""

import numpy as np

#from scipy.linalg import solve_banded # works for symmetric or hermitian matrices

from scipy.sparse import csc_matrix      # sparse matrix solver
from scipy.sparse.linalg import spsolve      # sparse matrix solver



def omega(pk,qk,CSj00,CSj01):

    return pk*CSj00+qk*CSj01

def theta(pk,qk,CSj01,CSj11):

    return pk*CSj01+qk*CSj11


def A_swapped(h,omega0,omega1,theta0,theta1):

    A0 = np.array([-h/2.,0,-1,0])
    A1 = np.array([0,-h/2.,0,-1])
    A2 = np.array([h*omega0-1, h*theta0, 0, 0])
    A3 = np.array([h*omega1  , h*theta1-1, 0, 0])

    return np.array([A0,A1,A2,A3])

def A(h,omega0,omega1,theta0,theta1):

    A0 = np.array([-1,0,-h/2.,0])
    A1 = np.array([0,-1,0,-h/2.])
    A2 = np.array([0, 0,h*omega0-1, h*theta0])
    A3 = np.array([0, 0,h*omega1  , h*theta1-1])

    return np.array([A0,A1,A2,A3])

def B_swapped(h,omega0,omega1,theta0,theta1):

    B0 = np.array([-h/2.,0,1,0])
    B1 = np.array([0,-h/2.,0,1])
    B2 = np.array([h*omega0+1, h*theta0, 0, 0])
    B3 = np.array([h*omega1  , h*theta1+1, 0, 0])

    return np.array([B0,B1,B2,B3])

def B(h,omega0,omega1,theta0,theta1):

    B0 = np.array([1,0,-h/2.,0])
    B1 = np.array([0,1,0,-h/2.])
    B2 = np.array([ 0, 0,h*omega0+1, h*theta0])
    B3 = np.array([ 0, 0,h*omega1  , h*theta1+1])

    return np.array([B0,B1,B2,B3])

def compute_residual_vec_swapped(space, X_swapped, delta_s):
    '''
    Compute the residual of the newton equation (function that should become 0)

    - X_swapped is the state of the current path: a 4*m vector of the form
    [u0,v0,p0,q0,p1,q1,u1,v1, ...,p_m-1,q_m-1,u_m-1,v_m-1]
    - delta_s is an array of dimension (m-1) of approximated distances ds_k between
    consecutive points P_k, P_k+1 contained in X
    [ds_0,ds_1,..., ds_m-2], where ds_k = |S(u_k+1,v_k+1)-S(u_k,v_k)|
    '''
    m, r = divmod(len(X_swapped),4)
    assert(r == 0)   # the vector dim should be a multiple of 4

    G = np.zeros(4*m)  # array of derivatives (corresponds to differential equations)
    R = np.zeros(4*m)  # array of residuals between consecutive points in the curve

    # Initialization of the G array for k = 0.
    # The first 2 components of X_swapped are not swapped in the swapped vector X_swapped,
    # (hence treated separately here).
    uvpq = (X_swapped[0], X_swapped[1], X_swapped[2], X_swapped[3])
    # Compute G from the geodesic equations (second argument is not used --> set to 0)
    new_uvpq = space.geodesic_eq(uvpq, 0)
    G[0],G[1],G[2],G[3] = new_uvpq

    # Initialization of the R array for k = 0 at u,v (and not p,q)
    # residuals at u,v corresponds to boundary conditions, i.e should be 0
    R[0] = R[1] = 0.

    # then compute the rest of the vectors G (derivatives, used internally) and R (residuals)
    for k in range(1,m): # k = 1,.., m-1
        i = 4*k    # index corresponding to kth uvpq set in vector X

        pquv = (X_swapped[i], X_swapped[i + 1], X_swapped[i + 2], X_swapped[i + 3])

        # Compute the discretization of G using the geodesic equations
        # Note: the second argument is not used by the function geodesic_eq
        # (0 here after is not used by the function but required in the signature).

        uvpq = [pquv[2],pquv[3],pquv[0],pquv[1]] # unswap pquv before calling geodesic equation
        new_uvpq = space.geodesic_eq(uvpq, 0)

        # Then swap back the returned uvpq value to use them in the swapped vectors G and R
        new_uvpq_swapped = [new_uvpq[2],new_uvpq[3],new_uvpq[0],new_uvpq[1]]

        for h in range(4):
            j = i+h
            # ex: k = 1, h = 0, => j = 4, j-2 = 2;
            # ex: k = 2, h = 0, => j = 8, j-2 = 6

            G[j] = new_uvpq_swapped[h]
            if k==1: # to account for the fact that values at k == 0 are not swapped
                if h < 2:# h = 0,1 (i.e. j = 4,5)
                    R[j - 2] = (X_swapped[j] - X_swapped[j - 2]) / delta_s[k - 1] - 0.5 * (G[j] + G[j - 2])
                else:    # h = 2,3 (i.e. j = 6,7)
                    R[j - 2] = (X_swapped[j] - X_swapped[j - 6]) / delta_s[k - 1] - 0.5 * (G[j] + G[j - 6])
            else:
                R[j - 2] = (X_swapped[j] - X_swapped[j - 4]) / delta_s[k - 1] - 0.5 * (G[j] + G[j - 4])

    # R has dimension 4*m, according to the loop above, two values are missing
    # They correspond to the boundary conditions at the target point
    # [p_4m-1,q_4m-1,u_4m-1,v_4m-1] where the curve should pass through point (u_m-1,v_m-1)
    # corresponding to indexes 4*m-2 (u) and 4*m-1 (v) in swapped vectors of dim 4*m.
    R[4*m-2] = R[4*m-1] = 0.

    # This means that:
    #   R[0] = 0 # imposed as a boundary condition
    #   R[1] = 0 # imposed as a boundary condition
    # k = 1, h=0, (j=4) (corresponding to residual at p1) is
    #   R[2] = (X[4]-X[2])/delta_s[0]-0.5*(G[4]+G[2])
    # k = 1, h=1, (j=5) (corresponding to residual at q1) is
    #   R[3] = (X[5]-X[3])/delta_s[0]-0.5*(G[5]+G[3])
    # k = 1, h=2, 3, (j=6, 7) (corresponding to residual at u1 then v1) is
    #   R[4] = (X[6]-X[0])/delta_s[0]-0.5*(G[6]+G[0])
    #   R[5] = (X[7]-X[1])/delta_s[0]-0.5*(G[7]+G[1])
    # k = 2, h=0, 1  (j=8, 9) (corresponding to residual at p2 then q2) is
    #   R[6] = (X[8]-X[4])/delta_s[1]-0.5*(G[8]+G[4])
    #   R[7] = (X[9]-X[5])/delta_s[1]-0.5*(G[9]+G[5])
    # ...
    # k = m-1, h=3, (j=4(m-1)+3=4m-1) (corresponding to residual at v_m-1) is
    #   R[4*m-3] = (X[4*m-1]-X[4*m-5])/delta_s[m-2]-0.5*(G[4*m-1]+G[4*m-5])
    #
    # Note that p0,q0,pm-1 and qm-1 have no residuals in R
    # R[0] --> u0 residuals (forced to 0)
    # R[1] --> v0 residuals (forced to 0)
    # R[2] --> p1 residuals
    # R[3] --> q1 residuals
    # R[4] --> u1 residuals
    # R[5] --> v1 residuals
    # R[6] --> p2 residuals
    # R[7] --> q2 residuals
    # R[8] --> u2 residuals
    # R[9] --> v2 residuals
    # R[10] --> q3 residuals
    # ...
    # R[4m-7] --> v_m-2 residuals
    # R[4m-6] --> p_m-1 residuals
    # R[4m-5] --> q_m-1 residuals
    # R[4m-4] --> u_m-1 residuals
    # R[4m-3] --> v_m-1 residuals
    # R[4m-2] --> u_m residuals (forced to 0 as the m+1th point is considered as the given target values )
    # R[4m-1] --> v_m residuals (forced to 0 as the m+1th point is considered as the given target values )

    return R


def compute_residual_vec(space, X, delta_s):
    '''
    Compute the residual of the newton equation (function that should become 0)

    - X is the state of the current path: a 4*m vector of the form
    [u0,v0,p0,q0,u1,v1,q1,u1,v1, ...]
    - delta_s is an array of dimension (m-1) of approximated distances ds_k between
    consecutive points P_k, P_k+1 contained in X
    [ds_0,ds_1,..., ds_m-2], where ds_k = |S(u_k+1,v_k+1)-S(u_k,v_k)|
    '''
    m, r = divmod(len(X),4)
    assert(r == 0)   # the vector dim should be a multiple of 4

    G = np.zeros(4*m)  # array of derivatives (corresponds to differential equations)
    R = np.zeros(4*m)  # array of residuals between consecutive points in the curve

    # Initialization of the G array for k = 0.
    # The first 2 components of X_swapped are not swapped in the swapped vector X_swapped,
    # (hence treated separately here).
    uvpq = (X[0], X[1], X[2], X[3])
    # Compute G from the geodesic equations (second argument is not used --> set to 0)
    new_uvpq = space.geodesic_eq(uvpq, 0)
    G[0],G[1],G[2],G[3] = new_uvpq

    # Initialization of the R array for k = 0 at u,v (and not p,q)
    # residuals at u,v corresponds to boundary conditions, i.e should be 0
    R[0] = R[1] = 0.

    # then compute the rest of the vectors G (derivatives, used internally) and R (residuals)
    for k in range(1,m): # k = 1,.., m-1
        i = 4*k    # index corresponding to kth uvpq set in vector X

        uvpq = (X[i], X[i + 1], X[i + 2], X[i + 3])

        # Compute the discretization of G using the geodesic equations
        # Note: the second argument is not used by the function geodesic_eq
        # (0 here after is not used by the function but required in the signature).

        new_uvpq = space.geodesic_eq(uvpq, 0)

        for h in range(4):
            j = i+h
            # ex: k = 1, h = 0, => j = 4, j-2 = 2;
            # ex: k = 2, h = 0, => j = 8, j-2 = 6

            G[j] = new_uvpq[h]
            R[j - 2] = (X[j] - X[j - 4]) / delta_s[k - 1] - 0.5 * (G[j] + G[j - 4])

    # R has dimension 4*m, according to the loop above, two values are missing
    # They correspond to the boundary conditions at the target point
    # [p_4m-1,q_4m-1,u_4m-1,v_4m-1] where the curve should pass through point (u_m-1,v_m-1)
    # corresponding to indexes 4*m-2 (u) and 4*m-1 (v) in swapped vectors of dim 4*m.
    R[4*m-2] = R[4*m-1] = 0.

    # This means that:
    #   R[0] = 0 # imposed as a boundary condition
    #   R[1] = 0 # imposed as a boundary condition
    # k = 1, h=0, (j=4) (corresponding to residual at p1) is
    #   R[2] = (X[4]-X[2])/delta_s[0]-0.5*(G[4]+G[2])
    # k = 1, h=1, (j=5) (corresponding to residual at q1) is
    #   R[3] = (X[5]-X[3])/delta_s[0]-0.5*(G[5]+G[3])
    # k = 1, h=2, 3, (j=6, 7) (corresponding to residual at u1 then v1) is
    #   R[4] = (X[6]-X[0])/delta_s[0]-0.5*(G[6]+G[0])
    #   R[5] = (X[7]-X[1])/delta_s[0]-0.5*(G[7]+G[1])
    # k = 2, h=0, 1  (j=8, 9) (corresponding to residual at p2 then q2) is
    #   R[6] = (X[8]-X[4])/delta_s[1]-0.5*(G[8]+G[4])
    #   R[7] = (X[9]-X[5])/delta_s[1]-0.5*(G[9]+G[5])
    # ...
    # k = m-1, h=3, (j=4(m-1)+3=4m-1) (corresponding to residual at v_m-1) is
    #   R[4*m-3] = (X[4*m-1]-X[4*m-5])/delta_s[m-2]-0.5*(G[4*m-1]+G[4*m-5])
    #
    # Note that p0,q0,pm-1 and qm-1 have no residuals in R
    # R[0] --> u0 residuals (forced to 0)
    # R[1] --> v0 residuals (forced to 0)
    # R[2] --> p1 residuals
    # R[3] --> q1 residuals
    # R[4] --> u1 residuals
    # R[5] --> v1 residuals
    # R[6] --> p2 residuals
    # R[7] --> q2 residuals
    # R[8] --> u2 residuals
    # R[9] --> v2 residuals
    # R[10] --> q3 residuals
    # ...
    # R[4m-7] --> v_m-2 residuals
    # R[4m-6] --> p_m-1 residuals
    # R[4m-5] --> q_m-1 residuals
    # R[4m-4] --> u_m-1 residuals
    # R[4m-3] --> v_m-1 residuals
    # R[4m-2] --> u_m residuals (forced to 0 as the m+1th point is considered as the given target values )
    # R[4m-1] --> v_m residuals (forced to 0 as the m+1th point is considered as the given target values )

    return R


def build_jacobian_swapped_csc(space, X_swapped, delta_s):
    '''
    Builds the jacobian as a sparse matrix in csc format (list of (row,col) indexes of non-null entries)
    The jacobian matrix is computed with swapped columns to optimize pivot algorithm
    as follows:
    for k = 1..m-1,
      - column (4k-3) is interchanged with column (4k-1)
      - column (4k-2) is interchanged with column (4k)


    '''

    m, r = divmod(len(X_swapped),4)
    assert(r == 0)   # the vector dim should be a multiple of 4

    # To state that elements (0,0) and (1,1) of Jacobian matrix are non null:
    row = [0,1]  # list of row indexes of non-null elements
    col = [0,1]  # corresponding list of row indexes of non-null elements.
    # Then the values of the two preceding non-null elements are 1, 1
    dat = [1,1]  # data array contains non-null elements of the jacobian matrix (corresponds to blocks Ak and Bk)
                 # The lists are initialized by non-null elements of matrix B0

    # fills the jacobian matrix with non-null elements from Ak and Bk k = 1 .. m-1
    for k in range(1,m):
        i = 4*k

        # Compute uk,vk at point k
        uk,vk = X_swapped[i+2], X_swapped[i+3]
        # Compute uk,vk at point k-1
        if k == 1:
            uk1,vk1 = X_swapped[i-4], X_swapped[i-3] # at previous point (k = 0), components are not swapped
        else:
            uk1,vk1 = X_swapped[i-2], X_swapped[i-1]

        # Compute the pk,qk at k
        pk, qk = X_swapped[i], X_swapped[i+1]
        # Compute the pk,qk at k-1
        if k == 1:
            pk1, qk1 = X_swapped[i-2], X_swapped[i-1]
        else:
            pk1, qk1 = X_swapped[i - 4], X_swapped[i - 3]

        # Compute the Christoffel Symbols at point k and k-1
        CSk = space.ChristoffelSymbols(uk, vk)
        CSk1 = space.ChristoffelSymbols(uk1, vk1)

        # delta_s[k] stores |s_k+1 - s_k| at index k => here we have h = |s_k - s_k-1|=delta_s[k-1]
        h = delta_s[k-1]

        # Building of matrix A (at point k-1)
        omega0 = omega(pk1,qk1,CSk1[0,0,0],CSk1[0,0,1])
        omega1 = omega(pk1,qk1,CSk1[1,0,0],CSk1[1,0,1])
        theta0 = theta(pk1,qk1,CSk1[0,1,0],CSk1[0,1,1])
        theta1 = theta(pk1,qk1,CSk1[1,1,0],CSk1[1,1,1])

        # first compute the Ak and Bk+1 matrices
        # WARNING: A1 should not be swapped !!!!!!!!!!!!!!!!!
        if k == 1: # A should not be swapped
            Ak_swapped = A(h, omega0, omega1, theta0, theta1)
        else:
            Ak_swapped = A_swapped(h,omega0,omega1,theta0,theta1)

        # Building of matrix B (at k)
        omega0 = omega(pk,qk,CSk[0,0,0],CSk[0,0,1])
        omega1 = omega(pk,qk,CSk[1,0,0],CSk[1,0,1])
        theta0 = theta(pk,qk,CSk[0,1,0],CSk[0,1,1])
        theta1 = theta(pk,qk,CSk[1,1,0],CSk[1,1,1])

        Bk_swapped = B_swapped(h,omega0,omega1,theta0,theta1)

        for n in range(4): # line in Ak
            for j in range(4): # column j
                # indexes present in the blocks Ak, Bk:
                h  = 4*(k-1) + n + 2  # line index in the jacobian
                c  = 4*(k-1) + j      # col index is non-null
                c2 = 4*(k) + j        # second column on the same line that is also non-null for this j

                # Now fills in the correct data (swapped columns)
                d  = Ak_swapped[n][j]
                d2 = Bk_swapped[n][j]

                # add two entries on line h corresponding to non-null j entries in both Ak and Bk
                row.append(h)
                col.append(c)
                dat.append(d)

                row.append(h)
                col.append(c2)
                dat.append(d2)

    # in the end add the 2 non-null elements of array Am
    row.append(4*m-2)
    col.append(4*m-2)
    dat.append(1)

    row.append(4*m-1)
    col.append(4*m-1)
    dat.append(1)

    #print('BEFORE Jacobian row = ', row, len(row))
    #print('BEFORE Jacobian col = ', col, len(col))
    #print('BEFORE Jacobian dat = ', dat, len(dat))
    jacobian = csc_matrix((dat, (row, col)), shape=(4*m, 4*m))
    #print(' AFTER Jacobian jac = ')
    #print(jacobian.toarray())

    return jacobian

def build_jacobian_csc(space, X, delta_s):
    '''
    Builds the jacobian as a sparse matrix in csc format (list of (row,col) indexes of non-null entries)
    '''

    m, r = divmod(len(X),4)
    assert(r == 0)   # the vector dim should be a multiple of 4

    # To state that elements (0,0) and (1,1) of Jacobian matrix are non null:
    row = [0,1]  # list of row indexes of non-null elements
    col = [0,1]  # corresponding list of row indexes of non-null elements.
    # Then the values of the two preceding non-null elements are 1, 1
    dat = [1,1]  # data array contains non-null elements of the jacobian matrix (corresponds to blocks Ak and Bk)
                 # The lists are initialized by non-null elements of matrix B0

    # fills the jacobian matrix with non-null elements from Ak and Bk k = 1 .. m-1
    for k in range(1,m):
        i = 4*k  # current point index
        i1 = i-4  # previous point index

        # Compute uk,vk at point k
        uk,vk = X[i], X[i+1]
        # Compute uk,vk at point k-1
        uk1, vk1 = X[i1], X[i1+1]

        # Compute the pk,qk at k
        pk, qk = X[i+2], X[i+3]
        # Compute the pk,qk at k-1
        pk1, qk1 = X[i1+2], X[i1+3]

        # Compute the Christoffel Symbols at point k and k-1
        CSk = space.ChristoffelSymbols(uk, vk)
        CSk1 = space.ChristoffelSymbols(uk1, vk1)

        # delta_s[k] stores |s_k+1 - s_k| at index k => here we have h = |s_k - s_k-1|=delta_s[k-1]
        h = delta_s[k-1]

        # Building of matrix A (at point k-1)
        omega0 = omega(pk1,qk1,CSk1[0,0,0],CSk1[0,0,1])
        omega1 = omega(pk1,qk1,CSk1[1,0,0],CSk1[1,0,1])
        theta0 = theta(pk1,qk1,CSk1[0,1,0],CSk1[0,1,1])
        theta1 = theta(pk1,qk1,CSk1[1,1,0],CSk1[1,1,1])

        Ak = A(h,omega0,omega1,theta0,theta1)

        # Building of matrix B (at k)
        omega0 = omega(pk,qk,CSk[0,0,0],CSk[0,0,1])
        omega1 = omega(pk,qk,CSk[1,0,0],CSk[1,0,1])
        theta0 = theta(pk,qk,CSk[0,1,0],CSk[0,1,1])
        theta1 = theta(pk,qk,CSk[1,1,0],CSk[1,1,1])

        Bk = B(h,omega0,omega1,theta0,theta1)

        for n in range(4): # line in Ak
            for j in range(4): # column j
                # indexes present in the blocks Ak, Bk:
                h  = 4*(k-1) + n + 2  # line index in the jacobian
                c  = 4*(k-1) + j      # col index is non-null
                c2 = 4*(k) + j        # second column on the same line that is also non-null for this j

                # Now fills in the correct data (swapped columns)
                d  = Ak[n][j]
                d2 = Bk[n][j]

                # add two entries on line h corresponding to non-null j entries in both Ak and Bk
                row.append(h)
                col.append(c)
                dat.append(d)

                row.append(h)
                col.append(c2)
                dat.append(d2)

    # in the end add the 2 non-null elements of array Am
    row.append(4*m-2)
    col.append(4*m-4)
    dat.append(1)

    row.append(4*m-1)
    col.append(4*m-3)
    dat.append(1)

    #print('BEFORE Jacobian row = ', row, len(row))
    #print('BEFORE Jacobian col = ', col, len(col))
    #print('BEFORE Jacobian dat = ', dat, len(dat))
    jacobian = csc_matrix((dat, (row, col)), shape=(4*m, 4*m))
    #print(' AFTER Jacobian jac = ')
    #print(jacobian.toarray())

    return jacobian

def swap_columns(X):
    '''
    swaps the indexes of vector X of dimension 4m as follows:
    - The first 4 values are not swapped
    - For the rest, values at swapped according to index rule:
    4k+2 <--> 4k
    4k+3 <--> 4k+1

    precond: X is a vector of dim 4*m (m integer >= 1)
    '''

    dim = len(X)
    m, r = divmod(dim,4)
    assert(r == 0)   # the vector dim should be a multiple of 4

    Xswapped = np.zeros(dim)

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

def compute_deltaX_swapped(space, X_swapped, R_swapped, delta_s):
    '''
    Computes the path increment on the current path to lead to the geodesic path
    X_swapped = matrix of residuals at step k
    res_vec_swapped = column vector of residuals
    '''

    # Computes the residuals' jacobian matrix from Christoffel symbols and X (swapped) in a csc format
    # matrix J is a (4*m,4*m) matrix
    # note that the Jacobian has its columns already swapped to optimize matrix inversion

    J_mat_swapped = build_jacobian_swapped_csc(space, X_swapped, delta_s)

    # solves:    delta_X_swapped . J_mat_swapped = - R_swapped
    # (delta_X_swapped is the unknown var)
    # Note that this inversion is carried out directly on swapped quantities
    delta_X_swapped = spsolve(J_mat_swapped, -R_swapped)

    # checks the solution delta_X_swapped found by verifying if
    # delta_X_swapped . J_mat_swapped = - R_swapped
    check_sol = np.allclose(J_mat_swapped.dot(delta_X_swapped), (-R_swapped))
    print('delta_X_swapped computed: ', check_sol)
    #print(delta_X_swapped)

    return delta_X_swapped

def compute_deltaX(space, X, R, delta_s):
    '''
    Computes the path increment on the current path to lead to the geodesic path
    X_swapped = matrix of residuals at step k
    res_vec_swapped = column vector of residuals
    '''

    # Computes the residuals' jacobian matrix from Christoffel symbols and X in a csc format
    # matrix J is a (4*m,4*m) matrix
    # note that the Jacobian has its columns already swapped to optimize matrix inversion

    J_mat = build_jacobian_csc(space, X, delta_s)

    # solves:    delta_X . J_mat = - R
    # (delta_X is the unknown var)

    delta_X = spsolve(J_mat, -R)

    # checks the solution delta_X found by verifying if
    # delta_X . J_mat = - R
    check_sol = np.allclose(J_mat.dot(delta_X), (-R))
    print('delta_X computed: ', check_sol)
    #print(delta_X)

    return delta_X

def standardized_L1norm(v, MU = 1., MV = 1., MP = 1., MQ = 1.):
    '''
    v is assumed to be a swapped vector over the indexes k =  1 .. m-1 (note that 0 is missing)
    MU, MV, MP and MQ are factors corresponding to the magnitude of variables delta_u,delta_v,delta_p,delta_q
    '''
    dim = v.shape[0] # dimension of v should be one column of 4*m coordinates
    assert(len(v.shape) == 1)

    m, r = divmod(dim,4)
    assert(r == 0)   # the vector dim should be a multiple of 4

    sum = 0.
    for k in range(0,m): # now taking into account the fact that terms are swapped in the order p,q,u,v
        sum += abs(v[4*k]) / MU
        sum += abs(v[4*k+1]) / MV
        sum += abs(v[4*k+2]) / MP
        sum += abs(v[4*k+3]) / MQ

    return sum/dim

def standardized_L1norm_swapped(v, MU = 1., MV = 1., MP = 1., MQ = 1.):
    '''
    Swapped form of the norm.
    v is assumed to be a swapped vector over the indexes k =  1 .. m-1 (note that 0 is missing)
    MU, MV, MP and MQ are factors corresponding to the magnitude of variables delta_u,delta_v,delta_p,delta_q
    '''
    dim = v.shape[0] # dimension of v should be one column of 4*m coordinates
    assert(len(v.shape) == 1)

    m, r = divmod(dim,4)
    assert(r == 0)   # the vector dim should be a multiple of 4

    # the sum is initialized with the first 4 terms in the order u,v,p,q (not swapped)
    sum = abs(v[0])/MU + abs(v[1])/MV + abs(v[2])/MP + abs(v[3])/MQ

    for k in range(1,m): # now taking into account the fact that terms are swapped in the order p,q,u,v
        sum += abs(v[4*k]) / MP
        sum += abs(v[4*k+1]) / MQ
        sum += abs(v[4*k+2]) / MU
        sum += abs(v[4*k+3]) / MV

    return sum/dim









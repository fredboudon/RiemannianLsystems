"""
    Computing the Geodesics between two points in a Riemannian space.

    Author: C. Godin, Inria
    Date: Jan. 2021-2022
    Lab: RDP ENS de Lyon, Mosaic Inria Team

    Implementation of the algorithm described in on (Makeawa, 1996)

"""

import numpy as np
import numpy.linalg as linalg

#from scipy.linalg import solve_banded # works for symmetric or hermitian matrices

from scipy.sparse import csc_matrix      # sparse matrix solver
from scipy.sparse.linalg import spsolve      # sparse matrix solver
from scipy.integrate import odeint
from scipy.optimize import least_squares

def omega(pk,qk,CSj00,CSj01):

    return pk*CSj00+qk*CSj01

def theta(pk,qk,CSj01,CSj11):

    return pk*CSj01+qk*CSj11


def A(h,omega0,omega1,theta0,theta1):

    A0 = np.array([-1,0,-h/2.,0])
    A1 = np.array([0,-1,0,-h/2.])
    A2 = np.array([0, 0,h*omega0-1, h*theta0])
    A3 = np.array([0, 0,h*omega1  , h*theta1-1])

    return np.array([A0,A1,A2,A3])

def B(h,omega0,omega1,theta0,theta1):

    B0 = np.array([1,0,-h/2.,0])
    B1 = np.array([0,1,0,-h/2.])
    B2 = np.array([ 0, 0,h*omega0+1, h*theta0])
    B3 = np.array([ 0, 0,h*omega1  , h*theta1+1])

    return np.array([B0,B1,B2,B3])

def compute_delta_s(surf,X,m):
    # Here, X is assumed to have a shape (4*m,)

    # X being defined (a set of (uvpq) values along the path of size m
    # we approximate the distance between consecutive points
    # in the Riemannian space by the euclidean chord between these points:
    # There are m points and m-1 segments indexed 0 .. m-2 in delta_s
    # with delta_s[k] being the distance P_k+1 - P_k on the curve.

    delta_s = np.zeros(m - 1)
    for k in range(m - 1):
        # extracts u,v coords of consecutive points on the curve
        u1, v1 = X[4 * k], X[4 * k + 1]
        u2, v2 = X[4 * (k + 1)], X[4 * (k + 1) + 1]
        # computes the corresponding points (np arrays) in the physical space
        P1 = surf.S(u1, v1)
        P2 = surf.S(u2, v2)
        # the norm is euclidean as a proxy, relying on the fact
        # that if the points are close enough, we can consider them
        # almost in a locally euclidean space.
        # Note that delta_s[k] contains P_k+1 - P_k on the curve
        delta_s[k] = linalg.norm(P2 - P1)

    #print("compute_delta_s:", delta_s)
    return delta_s


def compute_residual_vec_old(space, X, delta_s):
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
    # [p_4m-1,q_4m-1,u_4m-1,v_4m-1] where the curve should pass through point (u_m-1,v_m-1).
    R[4*m-2] = R[4*m-1] = 0. #BUG !!! Should be R[4*m-4] = R[4*m-3] = 0 see new procedure below

    return R

def compute_residual_vec(space, X, uv, utvt, delta_s):
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

    uvpq = (X[0], X[1], X[2], X[3])
    # Compute G from the geodesic equations (second argument is not used --> set to 0)
    new_uvpq = space.geodesic_eq(uvpq, 0)
    G[0],G[1],G[2],G[3] = new_uvpq

    # Initialization of the R array for k = 0 at u,v (and not p,q)
    # residuals at u,v corresponds to boundary conditions, i.e should be 0
    R[0] = X[0]-uv[0]
    R[1] = X[1]-uv[1]

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
    # [u_4m-1,v_4m-1,p_4m-1,q_4m-1] where the curve should pass through point (u_m-1,v_m-1).
    R[4*m-4] = X[4*m-4]-utvt[0]
    R[4*m-3] = X[4*m-3]-utvt[1]

    return R

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

def compute_deltaX(space, X, R, delta_s):
    '''
    Computes the path increment on the current path to lead to the geodesic path
    X = matrix of residuals at step k
    R = column vector of residuals
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
    #print('delta_X computed: ', check_sol)
    #print(delta_X)

    return delta_X

def standardized_L1norm(v, MU = 1., MV = 1., MP = 10., MQ = 10.):
    '''
    v is assumed to be a vector over the indexes k =  1 .. m-1 (note that 0 is missing)
    MU, MV, MP and MQ are factors corresponding to the magnitude of variables delta_u,delta_v,delta_p,delta_q
    MP and MQ are taken 10 times higher than MU and MV, as for similar order of magnitude we want that p and q have less importance in the error.
    '''
    dim = v.shape[0] # dimension of v should be one column of 4*m coordinates
    assert(len(v.shape) == 1)

    m, r = divmod(dim,4)
    assert(r == 0)   # the vector dim should be a multiple of 4

    sum = 0.
    sum_u = 0.
    sum_v = 0.
    sum_p = 0.
    sum_q = 0.

    for k in range(0,m):
        sum += abs(v[4*k]) / MU
        sum += abs(v[4*k+1]) / MV
        sum += abs(v[4*k+2]) / MP
        sum += abs(v[4*k+3]) / MQ
        # Below: just used for printing debug (printed below)
        sum_u += abs(v[4 * k]) / MU
        sum_v += abs(v[4 * k + 1]) / MV
        sum_p += abs(v[4 * k + 2]) / MP
        sum_q += abs(v[4 * k + 3]) / MQ

    #print(f"average sums u,v,p,q: {sum_u/dim : .3f}, {sum_v/dim : .3f}, {sum_p/dim : .3f}, {sum_q/dim : .3f}")

    return sum/dim





#########################################################
# Variant on Maekawa's BVP algorithm using least squares
#########################################################

def bvp_residuals(X, uv, utvt, surf, delta_s):

    # Note that the endpoint constraints on uv and utvt
    # also taken into account in the definitions of bounds.
    # see find_solution_bvp() function below.

    R = compute_residual_vec(surf, X, uv, utvt, delta_s)

    return R   # residual vector: its L2 norm should be minimized.

def find_solution_bvp(surf, uvpq_s, utvt):
    '''
    uvpq_s is the initial path (= sequence of uvpq starting at u0,v0 in direction p0,q0)
    Finds the sequence of uvpq coordinates that leads from point (u0,v0)
    to target point (ut,vt) on a given surface surf on a geodesic,
    starting with a first guess in direction (p0,q0)
    '''
    m = len(uvpq_s)
    uv = uvpq_s[0][:2]

    X = uvpq_s.reshape(4*m,)

    # array of min bounds on X values: fix the values of u0v0 and utvt
    # so that they do not vary. All other values are unbounded.
    bounds_min = np.empty(4*m)
    bounds_min.fill(-np.inf)
    bounds_min[0] = uvpq_s[0][0]-10-6
    bounds_min[1] = uvpq_s[0][1]-10-6
    bounds_min[m-4] = utvt[0]-10-6
    bounds_min[m-3] = utvt[1]-10-6

    bounds_max = np.empty(4*m)
    bounds_max.fill(np.inf)
    bounds_max[0] = uvpq_s[0][0]+10-6
    bounds_max[1] = uvpq_s[0][1]+10-6
    bounds_max[m-4] = utvt[0]+10-6
    bounds_max[m-3] = utvt[1]+10-6

    delta_s = compute_delta_s(surf,X,m)

    J_mat = build_jacobian_csc(surf, X, delta_s)

    #print(uv,utvt)
    #print(X,delta_s)
    #print(J_mat)
    #X_sol = least_squares(bvp_residuals, X, J_mat_dense, bounds = (bounds_min,bounds_max), args = (uv,utvt,surf,delta_s))
    X_sol = least_squares(bvp_residuals, X,
                          jac_sparsity = J_mat,
                          bounds=(bounds_min, bounds_max),
                          args=(uv, utvt, surf, delta_s))
    #X_sol = least_squares(bvp_residuals, X, args=(uv, utvt, surf, delta_s))

    #print("*** Solution least squares= ", X_sol.x)
    print("*** Reason for stopping --> ", X_sol.status)

    return X_sol.x.reshape((m,4))  ## returns the found path (array of uvpq)





#######################################################
# Alternative method: Shooting to find the target point
#######################################################

# Shoots straight in the direction pq = [p,q] over a distance l from point uv = [u,v] (considered constant)

def shooting(pql, uv, surf, SUBDIV):
    u,v = uv
    p,q,l = pql
    n = surf.norm(u, v, [p, q])
    pp = p / n
    qq = q / n

    # computes the time tics at which the geodesic equation must be integrated
    # over l: sudivides [0,l] into SUBDIV points (including extremities)
    # defining SUBDIV-1 segments with equal space.
    s = np.linspace(0, l, SUBDIV)

    uvpq_s = odeint(surf.geodesic_eq, np.array([u, v, pp, qq]), s)
    #print("shootin()--> uv = ", u,v )
    #print(uvpq_s)
    return uvpq_s

def shooting_residuals(pql, uv, utvt, surf, SUBDIV):

    uvpq_s = shooting(pql, uv, surf, SUBDIV)

    ufvf = uvpq_s[-1][:2]  # extracts the last uvpq and only the two first elements (u,v)
    residues_vec = ufvf - utvt
    return residues_vec

def find_shooting_solution(surf, uvpq_s, utvt, SUBDIV):
    '''
    uvpq_s is the initial path (= sequence of uvpq starting at u0,v0 in direction p0,q0)
    Finds the sequence of uvpq coordinates that leads from point (u0,v0)
    to target point (ut,vt) on a given surface surf on a geodesic,
    starting with a first guess in direction (p0,q0)
    '''

    uv = uvpq_s[0][:2]
    l = surf.path_length(uvpq_s)
    pql = np.array([uvpq_s[0][2], uvpq_s[0][3], l])
    #print("*** pql=", pql,"uv=",uv, "utvt=", utvt)
    pql_sol = least_squares(shooting_residuals, pql,  args = (uv,utvt,surf,SUBDIV))

    #print("*** Solution least squares= ", pql_sol)
    #print("*** find_shooting_solution: Reason for stopping --> ", pql_sol.status)

    # recomputes the found solution ... (with found p,q,l)
    #print("uv", uv, "pql_sol.x", pql_sol.x)
    uvpq_s = shooting(pql_sol.x, uv, surf, SUBDIV)
    #print(uvpq_s)
    return uvpq_s  ## returns the found path (array of uvpq)


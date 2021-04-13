"""
    Classes and functions for manipulating a Riemannian Turtle in LP-y

    Author: C. Godin, Inria
    Date: Jan. 2021
    Lab: RDP ENS de Lyon, Mosaic Inria Team

    scipy.linalg.solveh_banded: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solveh_banded.html

"""
import numpy as np

from scipy.linalg import solve_banded

from surfaces import *

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

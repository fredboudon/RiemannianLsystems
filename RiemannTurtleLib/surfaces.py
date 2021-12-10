"""
    Classes and functions for manipulating a Riemannian Turtle in LP-y

    Author: C. Godin, Inria
    Date: 2019-2021
    Lab: RDP ENS de Lyon, Mosaic Inria Team

"""
import numpy as np
import numpy.linalg

# Base class for the definition of parametric surfaces
class ParametricSurface:
# Surface position vector
    def S(self,u,v):
       print("S: base ParametricSurface class - ABSTRACT: NO IMPLEMENTATION")

    def uv_domain(self):
        return self.umin,self.umax,self.vmin,self.vmax

    def Shift(self,u,v):
      """ Shit tensor (3,2 matrix) to transform the coordinates u,v in x,y,z
      It is derived from the partial derivatives of the surface equations wrt u,v
      DS(u,v)/du, DS(u,v)/dv
      (may be could be computed automatically from S(u,v) ... see that later)
      """
      print("Shift tensor: base ParametricSurface class - ABSTRACT: NO IMPLEMENTATION")

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
      S_u, S_v = self.covariant_basis(u,v)
      N1 = - np.cross(S_u,S_v)

      return N1/np.linalg.norm(N1)

    def covariant_basis(self,u,v):
      A = self.Shift(u,v) # Compute shitf tensor at the new u,v,
      S1 = A[:,0] # first colum of A = first surface covariant vector in ambiant space
      S2 = A[:,1] # second column of A = second surface covariant vector in ambiant space
      return S1,S2

    def metric_tensor(self,u,v):
      """ NYI: To be defined for the base class
      = scalar product of covariant basis
      """
      guu = 1
      guv = gvu = 0.
      gvv = 1
      return np.array([[guu,guv],[gvu,gvv]])

    def inverse_metric_tensor(self,u,v):
      """
      tensor = inverse of the metric tensor
      """
      return np.linalg.inv(self.metric_tensor(u,v))

    def fundFormCoef(self,u,v):
      """
      Returns the coefficients of the fondamental forms I and II at point (u,v)
      - E,F,G: First fundamental form = metric (I = ds^2 = E du^2 + 2F dudv + G dv2)
      - L,M,N: Second funcdamental form = curvature (II = L du^2 + 2M dudv + N dv2)
      The second fondamental form represents for given du,dv how the surface
      gets quadratically away from the tangent plane.
      if d(du,dv) is the distance of a a point P(du,dv) to the tangent compliance
      II = 2 * d
      """
      g = self.metric_tensor(u,v)

      E = g[0,0]
      F = g[1,0]
      G = g[1,1]

      Suu = self.secondsuu(u,v)
      Suv = self.secondsuv(u,v)
      Svv = self.secondsvv(u,v)

      Normal = self.normal(u,v)

      L = np.dot(Suu,Normal)
      M = np.dot(Suv,Normal)
      N = np.dot(Svv,Normal)

      return (E,F,G,L,M,N)

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

    def principalCurvatures(self,u,v):
        """
        Defines kappa_min, kappa_max, and their direction
        """
        E,F,G,L,M,N = self.fundFormCoef(u,v)

        K = (L*N-M**2)/(E*G-F**2)
        H = (E*N+G*L-2*F*M)/2*(E*G-F**2)

        kappa_min = H + np.sqrt(H**2 - K)
        kappa_max = H - np.sqrt(H**2 - K)

        return kappa_min, kappa_max

    def J2orthonormal(self,u,v):
      """
      defines the Jacobian to pass from the covariant basis to an orthomormal
      basis constructed using Gram-Schmidt method starting with S1
      (useful to make rotation in the tangent plane defined by u,v
      """
      S1,S2 = self.covariant_basis(u,v)
      a = 1/np.linalg.norm(S1)
      g12 = self.metric_tensor(u,v)[0,1]
      S22 = S2 - g12 * S1
      b = 1/np.linalg.norm(S22)
      # print("J2orthonormal")
      # print("covariant S1 = ", S1)
      # print("covariant S2 = ", S2)
      # print("         S22 = ",S22)
      # print("g12 = ", g12, " a = ", a, " b = ", b)
      return np.array([[a,-g12*b],[0,b]])

    def Jfromorthonormal(self,u,v):
      S1,S2 = self.covariant_basis(u,v)
      a = np.linalg.norm(S1)
      g12 = self.metric_tensor(u,v)[0,1]
      S22 = S2 - g12 * S1
      b = np.linalg.norm(S22)
      #print("Jfromorthonormal")
      #print("covariant S1 = ", S1)
      #print("covariant S2 = ", S2)
      #print("         S22 = ",S22)
      #print("g12 = ", g12, " a = ", a, " b = ", b)
      return np.array([[a,g12*a],[0,b]])

    def RiemannChristoffelSymbols(self,u,v):
        """
        defined as the scalar product of S^a . dS_b / dS^c, with a,b,c in {u,v}
        """
        u = min(self.umax,max(self.umin,u))
        v = min(self.vmax,max(self.vmin,v))
        
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
        # Riemann-Christoffel symbols (RCS) that is an array of 2x2x2 = 8 numbers
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

        return RCS

    def geodesic_eq(self,uvpq,s):
        """
        definition of a geodesic at the surface starting at point u,v and heading in direction p,q
        expressing the tangent components in the surface covariant basis.
        """
        u,v,p,q = uvpq
        Gamma = self.RiemannChristoffelSymbols(u,v)
        lhs1 = - Gamma[0,0,0]*p**2 - 2*Gamma[0,0,1]*p*q - Gamma[0,1,1]*q**2
        lhs2 = - Gamma[1,0,0]*p**2 - 2*Gamma[1,0,1]*p*q - Gamma[1,1,1]*q**2
        return [p,q,lhs1,lhs2]

class Sphere(ParametricSurface):

    def __init__(self, R = 1.0):
      self.R = R # radius of the sphere
      self.CIRCUM = 2*np.pi*self.R  # Circumference
      self.umin = 0
      self.umax = 2*np.pi
      self.vmin = -np.pi/2.
      self.vmax = np.pi/2.

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

    # the Shift tensor may be wieved as the coordinates of the surface
    # covariant basis expressed in the ambiant basis (3x2) = 2 column vectors in 3D.
    def Shift(self,u,v):
      """ Shit tensor (3,2 matrix) to transform the coordinates u,v in x,y,z
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
      nx = np.cos(u)*np.cos(v)
      ny = np.sin(u)*np.cos(v)
      nz = np.sin(v)
      return np.array([nx,ny,nz])

    def geodesic_eq(self,uvpq,s):
      u,v,p,q = uvpq
      return [p,q,2*np.tan(v)*p*q,-np.cos(v)*np.sin(v)*p**2]



# For the moment the implementation is not done (copied from Sphere)
class EllipsoidOfRevolution(ParametricSurface):

    def __init__(self, a = 1.0, b = 0.5):
      self.a = a # radius of the circle at equator
      self.b = b # other radius

      self.umin = 0
      self.umax = 2*np.pi
      self.vmin = -np.pi/2.
      self.vmax = np.pi/2.

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
      den = np.sqrt((self.a*np.sin(v))**2 + (self.b*np.cos(v))**2 )
      nx = self.b*np.cos(u)*np.cos(v)
      ny = self.b*np.sin(u)*np.cos(v)
      nz = self.a*np.sin(v)
      return np.array([nx,ny,nz])

    def geodesic_eq(self,uvpq,s):
      u,v,p,q = uvpq
      den = (self.a*np.sin(v))**2 + (self.b*np.cos(v))**2
      return [p,q,2*np.tan(v)*p*q,-((self.a**2-self.b**2)*np.cos(v)*np.sin(v)/den)*q**2 - ((self.a**2)*np.cos(v)*np.sin(v)/den)*p**2]


class Torus(ParametricSurface):

    def __init__(self, R = 1.0, r = 0.2):
      self.R = R # radius of the torus (center to medial circle)
      self.r = r # radius of the torus cylinder
      self.umin = 0
      self.umax = 2*np.pi
      self.vmin = 0
      self.vmax = 2*np.pi

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
      return np.array([nx,ny,nz])

    def metric_tensor(self,u,v):
      guu = ((self.R + self.r*np.cos(v)))**2
      guv = gvu = 0.
      gvv = self.r**2
      return np.array([[guu,guv],[gvu,gvv]])

    def geodesic_eq(self,uvpq,s):
      u,v,p,q = uvpq
      return [p,q,2*(self.r*np.sin(v)/(self.R+self.r*np.cos(v)))*p*q,-((self.R+self.r*np.cos(v))*np.sin(v)/self.r)*p**2]


class Paraboloid(ParametricSurface):

    def __init__(self, radiusmax = 0.5):
      """
      u = radius of the circle (r)
      v = position on the circle (theta)
      """
      self.umin = 0
      self.umax = radiusmax
      self.vmin = 0.
      self.vmax = 2*np.pi

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
      yv = -u*np.cos(v)
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
      yuv =  np.cos(u)
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
      nx = -2*np.cos(v)/(4*u**2 + 1)
      ny = -2*np.sin(v)/(4*u**2 + 1)
      nz = 1/(u * (4*u**2 + 1) )
      return np.array([nx,ny,nz])

    def geode_sic_eq(self,uvpq,s):
      u,v,p,q = uvpq
      return [p,q, -4*u*p**2+ u*q**2, -2*u*p*q]

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


def derivatives(surf, u, v, order):
    return surf.derivatives(u,v,order)


import numpy as np
def derivatives(surf, u, v, order):
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


class Patch(ParametricSurface):

    def __init__(self, patch):
      self.patch = patch
      self.nurbssurf = to_nurbs_python(patch)
      self.umin = min(self.nurbssurf.knotvector_u)
      self.umax = max(self.nurbssurf.knotvector_u)
      self.vmin = min(self.nurbssurf.knotvector_v)
      self.vmax = max(self.nurbssurf.knotvector_v)

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

    def RiemannChristoffelSymbols(self,u,v):
        """
        defined as the scalar product of S^a . dS_b / dS^c, with a,b,c in {u,v}
        """
        u = min(self.umax,max(self.umin,u))
        v = min(self.vmax,max(self.vmin,v))
        # covariant basis
        S_u, S_v = self.covariant_basis(u,v)

        # inverse metric tensor ig
        ig = self.inverse_metric_tensor(u,v)

        # contravariant basis
        Su = ig[0][0] * S_u + ig[0][1] * S_v
        Sv = ig[1][0] * S_u + ig[1][1] * S_v

        # derivatives of the covariant basis
        skl = derivatives(self.nurbssurf, u, v, 2)

        S_uu = np.array(skl[2][0])
        #S_uu = S_uu[0:3]
        S_uv = S_vu = np.array(skl[1][1])
        #S_uv = S_uv[0:3]
        #S_vu = S_vu[0:3]
        S_vv = np.array(skl[0][2])
        #S_vv = S_vv[0:3]

        # compute the dot product ...
        # Riemann-Christoffel symbols (RCS) that is an array of 2x2x2 = 8 numbers
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

        return RCS


class Revolution(ParametricSurface):

    def __init__(self, rfunc, rprime, rsecond, zmin = -2*np.pi, zmax = 2*np.pi):
      """
      u = theta - azimuthal position aroundd the symmetry axis
      v = z - altitude on the symmetry axis

      r is a function of z (i.e. v) that defines in 3D the radius of the point at altitude z
      rprime is its derivative
      rsecond is its second derivative
      """
      self.umin = 0
      self.umax = 2*np.pi
      self.vmin = zmin
      self.vmax = zmax
      self.rfunc = rfunc        # care this are functions
      self.rprime = rprime      # first derivative of the radius with respect to z
      self.rsecond = rsecond    # second derivative of the radius with respect to z

    def S(self,u,v):
      """ Returns the coordinates (x,y,z) of a position vector restricted to the paraboloid surface
      as a function of the two surface coordinates (u,v)
      u = radius (is not restrricted in principle)
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
      factor = 1/(np.sqrt(1+self.rprime(v)**2))
      nx = factor * np.cos(u)
      ny = factor * np.sin(u)
      nz = - factor * self.rprime(v)
      return np.array([nx,ny,nz])

    #def geodesic_eq(self,uvpq,s):
    #  u,v,p,q = uvpq
    #  return [p,q, -4*u*p**2+ u*q**2, -2*u*p*q]

    def RiemannChristoffelSymbols(self,u,v):
        """
        defined as the scalar product of S^a . dS_b / dS^c, with a,b,c in {u,v}
        """
        # Riemann-Christoffel symbols (RCS) that is an array of 2x2x2 = 8 numbers
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

    def geodesic_eq(self,uvpq,s):
      u,v,p,q = uvpq
      r = self.rfunc(v)
      r1 = self.rprime(v)
      r2 = self.rsecond(v)
      den = 1+r1**2
      return [p,q, -2*p*q*r1/r, -r1*r2*(q**2)/den + r*r1*(p**2)/den]

#################################
# Other surface tools
#################################

# computation of generic derivatives using scipy
################################################
from scipy.misc import derivative

def prime_derivative(func, x):
    """
    Given a function, use a central difference formula with spacing dx to compute the nth derivative at x0, and using an odd number of points around the x value (order = 3)
    """
    return derivative(func, x, n=1, dx=1e-6, order = 3)

def second_derivative(func, x):
    return derivative(func, x, n=2, dx=1e-6, order = 3)


# For plotting any surface defined by an explicit equation S(u,v) with Quads
##############################################################################

def QuadifySurfEquation(surface,umin=0,umax=1,vmin=0,vmax=1,Du=0.01,Dv=0.01):
    '''
    Computes a quad representation of the surface defined as surface(u,v) --> scalar
    u is in [umax, umin], varies by steps of size Du
    v is in [vmax, vmin], varies by steps of size Dv
    '''
    # arange does not include the last bound if equal
    # --> shift the last value by Du (Dv)
    # to include the umax (vmax) bound in the array
    ilist = np.arange(umin,umax+Du,Du)
    jlist = np.arange(vmin,vmax+Dv,Dv)

    M = len(ilist)
    N = len(jlist)

    # 1D list containing the 3D points of the grid made by i,j values
    # (j varies quicker than i): kth point in the list corresponds
    # to point (i,j) such that k = i*N+j
    grid3Dpoints = [surface(i,j) for i in ilist for j in jlist]

    # Constructs the list of quad indexes pointing to the 3D points
    # in the previous list
    quadindexlist = [ (N*i+j-N-1,N*i+j-1,N*i+j,N*i+j-N)
            for i in range(1,M) for j in range(1,N)]

    return grid3Dpoints, quadindexlist

import openalea.plantgl.all as pgl
from math import *

class NurbsSwung:
    def __init__(self, profileList, angleList, degree = 3, ccw = True, slices = 30, stride = 30):
        assert degree in [1,3]
        for c in profileList:
          assert len(c.ctrlPointList) ==  len(profileList[0].ctrlPointList)
        self.profileList = profileList
        self.angleList = angleList
        self.degree = degree
        self.ccw = ccw
        self.slices = slices
        self.stride = stride

        self.build_interpolator()

    def build_interpolator(self):
        from openalea.plantgl.scenegraph.cspline import CSpline, cspline
        if self.degree > 1:
            #cpoints = [discretize(NurbsCurve([Vector4(p.x,p.y,a,p.z) for p in s.ctrlPointList])).pointList for s,a in zip(self.profileList, self.angleList)]
            cpoints = [[pgl.Vector3(p.x,p.y,p.z) for p in s.ctrlPointList] for s,a in zip(self.profileList, self.angleList)]
            cnurbs = [cspline([cpoints[i][j] for i in range(len(cpoints))]) for j in range(len(cpoints[0]))]
            cpoints = [[pgl.Vector4(p.x,p.y,0,p.z) for p in n.ctrlPointList] for n in cnurbs]
        else:
            cpoints = [[pgl.Vector4(p.x,p.y,0,p.z) for p in s.ctrlPointList] for s,a in zip(self.profileList, self.angleList)]
        knots =  [self.angleList[0] for i in range(self.degree+1)]+[a for a in self.angleList[1:-1] for i in range(3)]+[self.angleList[-1] for i in range(self.degree+1)]
        self.profileInterpolator = pgl.NurbsPatch(cpoints, ccw=self.ccw, vstride=self.stride, ustride=self.slices, vknotList=knots if self.degree > 1 else None, udegree=self.degree)
    
    def get_uknotList(self):
        return self.profileInterpolator.uknotList
    
    def set_uknotList(self, values):
        self.profileInterpolator.uknotList = values
    
    uknotList = property(get_uknotList)
    
    def get_vknotList(self):
        return self.profileInterpolator.vknotList
    
    def set_vknotList(self, values):
        self.profileInterpolator.vknotList = values
    
    vknotList = property(get_vknotList,)
    
    def getPointAt(self, u, v):
        result = self.profileInterpolator.getPointAt(u,v)
        theta = v
        return pgl.Vector3(result.x*cos(theta), result.x*sin(theta), result.y)
    
    def discretize(self):
        mesh = discretize(self.profileInterpolator)
        mesh.pointList = [pgl.Vector3(p.x*cos(p.z),p.x*sin(p.z), p.y) for p in mesh.pointList]
        return mesh
    
    def getUTangentAt(self,u,v):
        theta = v
        tangent = self.profileInterpolator.getUTangentAt(u,v)
        return (tangent.x*cos(theta),tangent.x*sin(theta),tangent.y)
    
    def getVTangentAt(self,u,v):
        p = self.profileInterpolator.getPointAt(u,v)
        theta = v
        tangent = self.profileInterpolator.getVTangentAt(u,v)
        return (tangent.x*cos(theta)-p.x*sin(theta),tangent.x*sin(theta)+p.x*cos(theta),tangent.y)
    
    def getSecondDerivativeUUAt(self, u,v):
      theta = v
      der = self.profileInterpolator.getDerivativeAt(u,v,2,0)
      return (der.x*cos(theta),der.x*sin(theta),der.y)
    
    def getSecondDerivativeUVAt(self, u,v):
      theta = v
      tangent = self.profileInterpolator.getUTangentAt(u,v)
      der = self.profileInterpolator.getDerivativeAt(u,v,1,1)
      return (der.x*cos(theta)-tangent.x*sin(theta),
              der.x*sin(theta)+tangent.x*cos(theta),
              der.y)
    
    def getSecondDerivativeVVAt(self, u,v):
      p = self.profileInterpolator.getPointAt(u,v)
      theta = v 
      tangent = self.profileInterpolator.getVTangentAt(u,v)
      der = self.profileInterpolator.getDerivativeAt(u,v,0,2)
      return (der.x*cos(theta) - 2*tangent.x*sin(theta) - p.x*cos(theta),
              der.x*sin(theta) + 2*tangent.x*cos(theta) - p.x*sin(theta),
              der.y)
    
    def getDerivativeAt(self, u, v, du, dv):
        if du == 2 : return self.getSecondDerivativeUUAt(u,v)
        elif dv == 2: return self.getSecondDerivativeVVAt(u,v)
        else: return self.getSecondDerivativeUVAt(u,v)

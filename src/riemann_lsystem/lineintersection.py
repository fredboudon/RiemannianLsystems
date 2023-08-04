#from openalea.plantgl.all import Point2Grid, segmentIntersect
import numpy as np
import bisect

def segmentIntersect(p1a,p1b,p2a,p2b):
    d0 = p2a-p1a
    d1 = p1b-p1a
    d2 = p2b-p2a
    a = np.cross(d1,d2)
    if abs(a) < 1e-10 : return False
    b = np.cross(d0,d1)
    c = np.cross(d0,d2)
    #print(a,b,c)
    beta = b/a
    alpha = c/a
    #print(alpha,beta)
    if beta <= 0 or 1-beta <= 0 : return False
    if alpha <= 0 or 1-alpha <= 0 : return False
    return True #, p1a+beta*d1, p2a+alpha*d2

def bbox(points):
    return (np.min(points,axis=0),np.max(points,axis=0))

def bbox_intersect(bbx1, bbx2):
    for i in range(2):
        if bbx1[0][i] >  bbx2[1][i] : return False
        if bbx1[1][i] <  bbx2[0][i] : return False
    return True

def bbox_pt_intersect(pt, bbx):
    for i in range(2):
        if pt[i] >  bbx[1][i] : return False
        if pt[i] <  bbx[0][i] : return False
    return True

def line_intersection(points1, points2):
        for i, (p1a, p1b) in enumerate(zip(points1,points1[1:])):
            for p2a, p2b in zip(points2,points2[1:]):
                if segmentIntersect(p1a,p1b,
                                    p2a,p2b):
                    return i
        return None

class LineIntersection:
    def __init__(self, numericratio = 1):
        self.points = {}
        self.bvh = {}
        self.numericratio = numericratio

    def __contains__(self, lineid):
        return lineid in self.points

    def ids(self):
        return self.points.keys()

    def add_line(self, points, id = None):
        points = np.array(points)*self.numericratio
        if not id in self.points:
            self.points[id] = points
        else:
            self.points[id] = np.concatenate((self.points[id],points))
        self.bvh[id] = bbox(self.points[id])
        return id

    def line_points(self, lineid):
        return self.points[lineid]

    def bboxes(self, pos):
        result = []
        for bid, bbxL in self.bvh.item():
            if bbox_pt_intersect(np.array(pos)*self.numericratio, bbxL):
               result.append(bid)
        return result

    def nb_bboxes(self, pos):
        return len(self.bboxes(pos))

    def test_inter_intersection(self, lid1, lid2):
        #return bbox_intersect(self.bvh[lid1], self.bvh[lid2])
        return self.test_intersection(self.points[lid1]/self.numericratio, exclude=[lid for lid in self.ids() if lid != lid2])

    def test_intersection_to_other(self, lid1):
        #return bbox_intersect(self.bvh[lid1], self.bvh[lid2])
        return self.test_intersection(self.points[lid1]/self.numericratio, exclude=[lid1])

    def test_intersection(self, points, bbxtestonly = False, verbose = False, exclude = []):
        """
        Return False in case of non intersection
        Return point id before intersection, line id with which it intersect
        """
        points = np.array(points)*self.numericratio
        bbxC = bbox(points)

        exclude = set(exclude)
        linetotest = set([])
        for bid, bbxL in self.bvh.items():
            if (not bid in exclude) and bbox_intersect(bbxC, bbxL):
               if bbxtestonly: return None, bid
               linetotest.add(bid)
        if bbxtestonly: return False

        intersections = []
        for l in linetotest:
            intersect = line_intersection(points, self.line_points(l))
            if not intersect is None:
                intersections.append((intersect,l))

        if verbose:
            print(list(sorted(set([(self.lineids[l],l) for l in linetotest]))),' --> ', [(self.lineids[l],l,i) for l,i in intersections])

        if len(intersections) == 0 : 
            return False

        return min(intersections)


def test():
    nr = 1000
    a = np.array([ 0.31316293, -0.05532964])*nr
    b = np.array([ 0.31556908, -0.05443881])*nr
    c = np.array([ 0.3107225 , -0.05447474])*nr
    d = np.array([ 0.3083074 , -0.05360834])*nr
    print(a,b,c,d)
    res = segmentIntersect(a,b,c,d)
    print(res)
    from openalea.plantgl.all import Viewer,Scene, Polyline2D,Translated
    Viewer.display(Scene([Translated(-a[0],-a[1],0,Polyline2D([a,b])),Translated(-a[0],-a[1],0,Polyline2D([c,d]))]))

if __name__ == '__main__':
    print('test')
    test()


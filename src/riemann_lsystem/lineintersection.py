#from openalea.plantgl.all import Point2Grid, segmentIntersect
import numpy as np
import bisect

def segmentIntersection(p1a,p1b,p2a,p2b):
    """ Predicate """
    d0 = p2a-p1a
    d1 = p1b-p1a
    d2 = p2b-p2a
    a = np.cross(d1,d2)
    if abs(a) < 1e-10 : return None
    b = np.cross(d0,d1)
    c = np.cross(d0,d2)
    beta = b/a
    alpha = c/a
    if beta <= 0 or 1-beta < 0 : return None
    if alpha <= 0 or 1-alpha < 0 : return None
    return p1a+alpha*d1 #p2a+alpha*d2

def bbox(pointset):
    """
    Compute the bounding box of a set of points

    :param pointset: n points (n being whatever, each point of shape = 2x1
    :return: 2 points corresponding to min and max coords of the bbox
    """
    return (np.min(pointset,axis=0),np.max(pointset,axis=0))

def bbox_union(bbx1, bbx2):
    return ([min(bbx1[0][i],bbx2[0][i]) for i in range(len(bbx1[0]))],[max(bbx1[1][i],bbx2[1][i]) for i in range(len(bbx1[0]))])

def bbox_bbox_intersection(bbx1, bbx2):
    """ 
    Test whether 2 bbox intersect
    :param bbox: 2 points of min and max coords of bbox
    :return: bool
    """
    for i in range(len(bbx1[0])):
        if bbx1[0][i] >  bbx2[1][i] : return False
        if bbx1[1][i] <  bbx2[0][i] : return False
    return True

def bbox_point_intersection(bbx, pt):
    """ Test whether pt is in bbox including the border:
    - bbox: 2 points of min and max coords of bbox
    """
    for i in range(len(pt)):
        if pt[i] >  bbx[1][i] : return False
        if pt[i] <  bbx[0][i] : return False
    return True

def line_intersection(pointset1, pointset2):
    """ Test whether two sequences of segments intersect each other at least once
    - returns the index of the id of the first segment that intersects a segment in
    the second list and the id of the segment of the second list
    """
    #runs over the segments of the first list, and then over the segments of the second list
    print(pointset1, pointset2)
    for i, (p1a, p1b) in enumerate(zip(pointset1,pointset1[1:])):
            for j, (p2a, p2b) in enumerate(zip(pointset2,pointset2[1:])):
                print('Test',p1a,p1b, p2a,p2b)              
                intsct = segmentIntersection(p1a,p1b, p2a,p2b)
                if not intsct is None:
                    return i, j, intsct
    return None

def border_crossing( p0, p1, periodicrange, coord = 0):
    if periodicrange is None : 
        return [p0,p1]
    def normalize_coord( u):
        if periodicrange and u < periodicrange[0] or u > periodicrange[1]:
            u = periodicrange[0] + (u-periodicrange[0]) % urange
        return u
    def normalize_point(p):
        return [normalize_coord(v) if i == coord else v for i,v in enumerate(p)]
    urange = periodicrange[1] - periodicrange[0]
    start = int(((p0[coord]-periodicrange[0])//urange))
    end = int(((p1[coord]-periodicrange[0])//urange))
    if start == end:
        res = [normalize_point(p0),normalize_point(p1)]
        return res
    elif start < end:
        mrange = range(start+1,end+1)
        borders = (periodicrange[1],periodicrange[0])
    else:
        mrange = range(start,end,-1)
        borders = periodicrange
    uvalues = [periodicrange[0]+urange*i for i in mrange]
    res = [normalize_point(p0)]+[[b if i == coord else p0[i]+(p1[i]-p0[i])*((u-p0[coord])/(p1[coord]-p0[coord])) for i in range(len(p0))] for u in uvalues for b in borders ]+[normalize_point(p1)]
    return res

class LineSet:
    '''
    The idea is to test the intersection of a new line with a set of preexisting lines representing polylines
    (a polyline is called called 'line' here.

    - linepoint is an np.array of shape nx2
    - lines = dict {line_id: list of line}
    - bvh = bounding volume hierarchy (the hierachical aspect is not used here) = dict of bounding boxes = {line_id: bbox}
    bbox here is a tuple of two points with min and max coordinates
    '''
    def __init__(self, numericalratio = 1, uperiodicrange = None, vperiodicrange = None):
        """
        This object registers set of lines and is able to determine if an intersection exists with a new line.
        """
        # A dictionnary of (id, set of points composing a line)
        self.lines = {}
        # A simple bounding volume hierarchy represented as a dict associating line ids with min and max coordinates
        self.bvh = {}
        # A ratio with which all coordinates are multiplied to avoid numerical issues
        self.numericalratio = numericalratio
        # The periodicity information
        self.uperiodicrange = uperiodicrange
        self.vperiodicrange = vperiodicrange
    
    def setSpace(self, space):
        if space.UPERIODIC:
            self.uperiodicrange = (space.umin, space.umax)
        else:
            self.uperiodicrange = None
        if space.VPERIODIC:
            self.vperiodicrange = (space.vmin, space.vmax)
        else:
            self.vperiodicrange = None

    def __contains__(self, lineid):
        return lineid in self.lines

    def ids(self):
        """ Return all the ids of the lines """
        return self.lines.keys()

    def normalize_coord(self, uv):
        u,v = uv
        if self.uperiodicrange and u < self.uperiodicrange[0] or u > self.uperiodicrange[1]:
            urange = self.uperiodicrange[1] - self.uperiodicrange[0]
            u = self.uperiodicrange[0] + (u-self.uperiodicrange[0]) % urange
        if self.vperiodicrange and v < self.vperiodicrange[0] or v > self.vperiodicrange[1]:
            vrange = self.vperiodicrange[1] - self.vperiodicrange[0]
            v = self.vperiodicrange[0] + (v-self.vperiodicrange[0]) % vrange
        return (u,v)

    def normalize_line(self, linepoints):
        if self.uperiodicrange is None and self.vperiodicrange is None :  
            return [np.array(linepoints)], bbox(linepoints)
        _linepoints = []
        for pi,pj in zip(linepoints,linepoints[1:]):
            _sublinepoints = border_crossing(pi,pj, self.uperiodicrange, 0)
            for i in range(len(_sublinepoints)//2):
                pii,pjj = _sublinepoints[2*i],_sublinepoints[2*i+1]
                _subsublinepoints = border_crossing(pii,pjj, self.vperiodicrange, 1)
                _linepoints.append([pii])                        
                for j in range(len(_subsublinepoints)//2):
                    pii,pjj = _subsublinepoints[2*j],_subsublinepoints[2*j+1]
                    if j != 0:
                        _linepoints.append([pii])
                    _linepoints[-1].append(pjj)
        linepoints = [np.array(_ilinepoints)*self.numericalratio for _ilinepoints in _linepoints]
        from functools import reduce
        bbx = reduce(bbox_union,[bbox(line) for line in linepoints])
        return linepoints, bbx

    def add_line(self, linepoints, id = None) -> int:
        """ Add a new line. Return its id """
        assert len(linepoints) > 1
        linepoints, bbx = self.normalize_line(linepoints)
        if id is None:
            if len(self.lines) == 0:
                id = 0
            else:
                id = max(self.lines.keys())+1
        #if not id in self.lines:
        self.lines[id] = linepoints
        #else:
            #raise ValueError(id)
            # self.lines[id] = np.concatenate((self.lines[id],linepoints))
        self.bvh[id] = bbx
        return id

    def remove_line(self, id) -> bool:
        """ Removes a line with id from the list of dictionary entries
        as a side effect, this also removes the corresponding entry in the bbox dictionary
        Returns True if the action was successful (id was found), False otherwise"""

        if id in self.lines:
            del self.lines[id]
            del self.bvh[id]
            return True
        else:
            return False

    def add_line_from_point(self, initpoint, linepoints, id = None) -> int:
        fp = np.array([initpoint[i] for i in range(len(linepoints[0]))])
        if np.linalg.norm(fp-linepoints[0]) > 1e-5:
            linepoints = [fp]+linepoints
        return self.add_line(linepoints, id)

    def remove_line(self, id) :
        del self.lines[id]
        del self.bvh[id]

    def line_points(self, lineid) -> np.array:
        """ Return the points of a line """
        return self.lines[lineid]

    def bboxes(self, pos) -> list:
        """
        For a given 2D point pos,
        returns the list bounding box ids containing the point
        """
        result = []
        for bid, bbxL in self.bvh.items():
            if bbox_point_intersection(bbxL, np.array(pos)*self.numericalratio):
               result.append(bid)
        return result

    def nb_bboxes(self, pos) -> int:
        """ Return the number of bounding boxes containing the point pos """
        return len(self.bboxes(pos))

    def test_intersection(self, linepoints, bbxtestonly = False, verbose = False, exclude = []):
        """
        Test the intersection of the line defined by linepoints with lines defined in self.

        :param linepoints : set of points that represent the line to test
        :param bbxtestonly : make only the bounding box test
        :param verbose : print information during test
        :param exclude : list of ids of lines of self to not test
        :return: False in case of non intersection,
          otherwise returns
          (index of the point id before intersection in linepoints,
           index of the sub line and the point id before intersection of the intersected line,
           line id with which it intersects
           found intersection point)
        """
        linepoints, bbxC = self.normalize_line(linepoints)
        #linepoints = np.array(linepoints)*self.numericalratio
        #bbxC = bbox(linepoints)

        exclude = set(exclude)
        linetotest = set([])
        for bid, bbxL in self.bvh.items():
            if (not bid in exclude) and bbox_bbox_intersection(bbxC, bbxL):
               if bbxtestonly: return None, bid
               linetotest.add(bid)
        if bbxtestonly: return False

        intersections = []
        for l in linetotest:
            for i,sublinei in enumerate(linepoints):
                for j,sublinej in enumerate( self.line_points(l)):
                    intersect = line_intersection(sublinei,sublinej)
                    if not intersect is None:
                        
                        intersections.append((intersect[0]-i, (j, intersect[1]), l, intersect[2]))

        if verbose:
            print(list(sorted(set([(self.lineids[l],l) for l in linetotest]))),' --> ', [(self.lineids[l],l,i) for l,i in intersections])

        if len(intersections) == 0 : 
            return False

        return min(intersections)

    def test_inter_intersection(self, lid1, lid2):
        """
        Test intersection between 2 lines of self.
        """
        return self.test_intersection(self.lines[lid1]/self.numericalratio, exclude=[lid for lid in self.ids() if lid != lid2])

    def test_intersection_to_other(self, lid1):
        """
        Test intersection of one line of self with the others.
        """
        return self.test_intersection(self.lines[lid1]/self.numericalratio, exclude=[lid1])

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

def test2():
    from math import pi
    a = [(-9.312379632169453e-17, 0.09935873376730332), (-9.500930985241659e-17, -0.0001545677755249164)]
    b = [[6.25958653, 0.        ], [6.28318531, 0.        ]]
    print(2*pi)
    trajectories = LineSet(uperiodicrange=(0,2*pi))
    print(a)
    an, _ = trajectories.normalize_line(a)
    print(an)
    print(b)
    bn, _ = trajectories.normalize_line(b)
    print(bn)
    intersect = line_intersection(an[0],bn[0])
    print(intersect)
    

if __name__ == '__main__':
    print('test')
    test2()


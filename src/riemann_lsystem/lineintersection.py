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
    if beta <= 0 or 1-beta <= 0 : return None
    if alpha <= 0 or 1-alpha <= 0 : return None
    return p1a+alpha*d1 #p2a+alpha*d2

def bbox(pointset):
    """
    Compute the bounding bow of a set of points

    :param pointset: n points (n being whatever, each point of shape = 2x1
    :return: 2 points corresponding to min and max coords of the bbox
    """
    return (np.min(pointset,axis=0),np.max(pointset,axis=0))

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
    - returns the index of the id of the first segment that intersects a segment in the second list and the id of the segment of the second list
    """
    #runs over the segments of the first list, and then over the segments of the second list
    for i, (p1a, p1b) in enumerate(zip(pointset1,pointset1[1:])):
            for j, (p2a, p2b) in enumerate(zip(pointset2,pointset2[1:])):
                intsct = segmentIntersection(p1a,p1b, p2a,p2b)
                if not intsct is None:
                    return i, j, intsct
    return None

class LineSet:
    '''
    The idea is to test the intersection of a new line with a set of preexisting lines representing polylines
    (a polyline is called called 'line' here.

    - linepoint is an np.array of shape nx2
    - lines = dict {line_id: list of line}
    - bvh = bounding volume hierarchy (the hierachical aspect is not used here) = dict of bounding boxes = {line_id: bbox}
    bbox here is a tuple of two points with min and max coordinates
    '''
    def __init__(self, numericalratio = 1):
        """
        This object registers set of lines and is able to determine if an intersection exists with a new line.
        """
        # A dictionnary of (id, set of points composing a line)
        self.lines = {}
        # A simple bounding volume hierarchy represented as a dict associating line ids with min and max coordinates
        self.bvh = {}
        # A ratio with which all coordinates are multiplied to avoid numerical issues
        self.numericalratio = numericalratio

    def __contains__(self, lineid):
        return lineid in self.lines

    def ids(self):
        """ Return all the ids of the lines """
        return self.lines.keys()

    def add_line(self, linepoints, id = None) -> int:
        """ Add a new line. Return its id """
        linepoints = np.array(linepoints)*self.numericalratio
        if id is None:
            if len(self.lines) == 0:
                id = 0
            else:
                id = max(self.lines.keys())+1
        if not id in self.lines:
            self.lines[id] = linepoints
        else:
            self.lines[id] = np.concatenate((self.lines[id],linepoints))
        self.bvh[id] = bbox(self.lines[id])
        return id

    def remove_line(self, id) -> int:
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
        lpoints = [[initpoint[i] for i in range(len(linepoints[0]))]]+linepoints
        self.add_line(lpoints, id)

    def line_points(self, lineid) -> np.array:
        """ Return the points of a line """
        return self.lines[lineid]

    def bboxes(self, pos) -> list:
        """
        For a given 2D point pos,
        returns the list bounding box ids containing the point
        """
        result = []
        for bid, bbxL in self.bvh.item():
            if bbox_point_intersection(np.array(pos)*self.numericalratio, bbxL):
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
          (point id before intersection,
          point id before intersection of intersected line,
          line id with which it intersect
          found intersection point)
        """
        linepoints = np.array(linepoints)*self.numericalratio
        bbxC = bbox(linepoints)

        exclude = set(exclude)
        linetotest = set([])
        for bid, bbxL in self.bvh.items():
            if (not bid in exclude) and bbox_bbox_intersection(bbxC, bbxL):
               if bbxtestonly: return None, bid
               linetotest.add(bid)
        if bbxtestonly: return False

        intersections = []
        for l in linetotest:
            intersect = line_intersection(linepoints, self.line_points(l))
            if not intersect is None:
                intersections.append((intersect[0], intersect[1],l,intersect[2]))

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

if __name__ == '__main__':
    print('test')
    test()


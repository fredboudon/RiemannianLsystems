# L-Py primitives in Riemannian (curved) spaces
Authors: C. Godin & F. Boudon

## Importing Riemannian header files

These lines should be placed as a header of a L-py file to use Riemannian L-Py code

    %pastemodule riemann_lsystem.riemannianrules
    

## Creating a space, plotting the space
Note: Can be set in either production, decomposition or interpretation rules

    nproduce SetSpace(Sphere(radius)
    nproduce ;(color) PlotSpace     # Plots the 'current' space if 
                                    # the global flag PLOTSPACE 
                                    # is True (default)

### Other available spaces:
    nproduce SetSpace(EllipsoidOfRevolution(ra,rb))`
    nproduce SetSpace(Torus(R,r))
    nproduce SetSpace(Paraboloid(R))
    nproduce SetSpace(ChineseHat(radius,zmin = 0.1,zmax=1.))
    nproduce SetSpace(PseudoSphere(R, zmin=-3, zmax=3))
    nproduce SetSpace(MonkeySaddle(a=0.1, n=3, umax=1.))
    nproduce SetSpace(Patch(patch1)) # patch1 can be a PlantGL nurbs or swung patch
    nproduce SetSpace(Revolution(profile, zmin=0, zmax=10)) # profile=PlantGL curve

### Ending the use of a space

To end the use of the active Space in the turtle stack:

    nproduce EndSpace()

### PlotSpace parameters
A dictionary can be passed on to Plot space to PlotSpace. The dictionary may be given the following entries (with user defined values):
* CCW = True (Default): change the orientation of the surface (invert the normals): 
* colorvalfunc: Function mapping [u,v] to scalar values which can be used to color the space
* cmap : 'jet' : color map ro
* 'Du' : 0.1, u resolution to draw the space
* 'Dv' : 0.1, v resolution to draw the space
* 
Example:

    def scalarField(u,v):
        return u+v 
        
    dict= {colorvalfunc: scalarField, cmap : 'jet', CCW:False)
    PlotSpace(dict)
    

## Initializing the turtle

First set the initial position and orientation of the turtle as a real set of 4 values for `[u0,v0,p0,q0]` in the space parameter domain, and then:

    nproduce InitTurtle([u0,v0,p0,q0])

Note that one can first orient the turtle in a reference orientation (e.g. [p0,q0]=[0,1]), and then rotate from this reference position and orientation:

    nproduce InitTurtle([u0,v0,p0,q0])
    nproduce +(30)
    
## Turtle movements

### Moving forward (geodesic segment) in a direction

If a Space has been created, the F primitive will draw a geodesic segment at the surface

Starting from the last state of the turtle:
    
    dl = 12         # length of the geodesic segment to draw
    nproduce F(dl)  # moves forward and draws
                    # or
    nproduce f(dl)  # moves forward without drawing
    
A variant makes it possible to optimize the drawing of curves as derivations proceed.

    StaticF(dl)

Using `StaticF` (instead of F) makes it possible to cach geodesic already computed (if one knows that they won't be changed at the next derivation step. In this way computations get faster).

**Discretization of the plotted curved segment**. The curved segments plotted in a curved space are approximated by polylines made of straigth small subsegments to fit the curvature of the space with sufficient precision.

Assume a primitive`F(steplen)` is used to draw a segment of length `steplen` in a curved space. The discretization of a step length is controled by the global constant MAXSTEPLEN that specifies the minimal resolution that the user requires for drawing curved segments on the surface. 

To plot a curved segment, a discretization of this segment in n points is made (including the segment end-points). This creates n-1 segments of equal size dl. n and dl are determined as follows:
- First, a maximum size of a segment is defined by the constant MAXSTEPLEN. The default constant can be overridden by the user.
By default: MAXSTEPLEN = 0.01 (in arbitrary units)
- Then, we compute n0:
     n0 = int(length // MAXSTEPLEN)+1    (integer division by a float returns a float)
- Then the curve is divided into n = n0-1 segments of equal length (and not n0 !) so that the the length dl of the segments is less or equal to MAXSTEPLEN (dl = length / (n0-1) ) and thus ensuring that dl <= MAXSTEPLEN. 

To summarize, any curved segment is plotted as n subsegments of equal length (n being determined automatically using the above procedure), such that:
- it ensures that the length of the subsegments remains under the constant MAXSTEPLEN.
- n is the minimal integer that achieves this constraint.

**Pre-computing a path sequence without moving**. In some situations, before making effectively a move, one first needs to estimate the trajectory that will be made if the move were to be made (i.e. before any actual move). This can be done using:

    uvpqs = forward(turtle, length, dl = maxstepsize)
    
This command will return the sequence of`uvpqs` computed for a geodesic starting at current position and orientation of the turtle over a given `length` and with a parameter `dl = maxstepsize`overridding the global constant `MAXSTEPLEN` (see previous paragraph).

Then, to apply effectively this move, the module StaticF() can be used:

    nproduce StaticF(length,uvpqs)

Notes: 
- the path `uvpqs` is cached by the module`StaticF`
- if `StaticF` has no second argument, it will first call the forward function (as described above), and cache its path (StaticF will check if the last argument is a dictionary,f not will call a forward first and cache the result as an exra argument)

It is also possible to draw a line following a forward using the new primitive P (for path):

    uvpqs = forward(turtle, length, dl = maxstepsize)
    ... extra code ...
    nproduce P(uvpqs)

produces a segment corresponding to the sequence of uvpq surface coordinates (with a result similar to a F(length) ).

### Moving forward (geodesic segment) to a target point

This is the equivalent to the LineTo procedure for the Euclidean turtle,
but now computes the geodesic between the current position of the turtle and the target point target_pt = [ut,vt]. 

A first strategy solves the problem as a BVP (Maekawa 1886) and is implemented in the RiemannLineTo() primitive:

    nproduce RiemannLineTo(target_pt,20)
    
The second argument defines a number of intermediate points that should be computed on the geodesic (= number of segments + 1).

A second implementation of LineTo in curved space is based on a shooting strategy using an Initial Value Problem (IVP). 

    nproduce RiemannShootingLineTo(target_pt,20)
    
For the moment, this shooting strategy is slower than the first BVP one.

Note that after a RiemannLineTo or RiemannShootingLineTo, the Head vector at the target point is systemically aligned with the tangent of the constructed geodesic.

### Moving to a target point without drawing a line

produce RiemannMoveTo(uv)

### Orienting the turtle in a direction

produce RiemannPinpoint(pq)

### Computing a geodesic path without moving

Same as F(slen, MAXSTEPLEN), except that the list of uv coordinates on the computed geodesic segment is returned and the turtle does not change its current state. 

    uvpq_s = forward(turtle, slen, MAXSTEPLEN)

This makes it possible to test possible movements on the surface without actually moving at once.

The bounding box in the parameter domain can be computed:

    bbx = bbox(uvpq_s)

### Indirect interpretation

The Forward instructions can be intperpreted either on the surface (standard behaviour) or can be executed in the parameter space and then reported on the surface. 

The second case correspond to an indirect interpretation of the turtle instructions and is activated by the statement:

    nproduce StartIndirectInterpretation
    ...  # other turtle movements now interpreted 
    ...  # as a movement in space uv, and then interpreted on the surface
    nproduce StopIndirectInterpretation
    
The last statement deactivate the indirect interpretation and the turtle resumes the interpretation of the forward instructions directly on the surface.

## Accessing the turtle state and turtle commands

`turtle` is a global variable accessible **only within the interpretation rules**

    space = turtle.space
    u,v,p,q = turtle.uvpq 

### Direction of the turtle at position (u,v)

It corresponds to the pushforward of [p,q] at [u,v] in R3:

    u,v,p,q = turtle.uvpq
    direction = turtle.space.shift_vector(u,v,p,q)
    
`direction` is a 3D vector.

This command can be used to push forward any vector `v=[v1,v2]` from the parameter space to R3:

    direction = turtle.space.shift_vector(u,v,v1,v2)
    
### Normal at position (u,v)

The normal is a 3D vector and is normalized:

    normal = turtle.space.normal(u,v)
    
### Local curvatures at position (u,v)
    
    u,v,p,q = turtle.uvpq
    K,H,kmin,kmax = turtle.space.localCurvatures(u,v)
    
    # K = Gaussian curvature
    # H = Mean curvature = (kmin + kmax)/2
    # kmin, kmax = principal curvatures

### Principal directions at position (u,v)

Code to retrieve the principal directions on the surface in 3D at the current position of the turtle:

    curvmin, curvmax, pdirmin_uv, pdirmax_uv = 
                = turtle.space.principalDirections(u,v)

    pushforward = turtle.space.Shift(u,v)
    
    pdirmax = pushforward.dot(pdirmax_uv)
    ndir = turtle.space.normal(u,v)
    pdirmin = np.cross(pdirmax, ndir)

### Surface mapping (according to the surface equations)

The 3D point P corresponding to a pair of parameter [u,v] on the surface is given by:

    P = turtle.space.S(u,v)
    
### pushforward/shift operator at (u,v)
  
    u,v,p,q = turtle.uvpq
    pushforward = turtle.space.Shift(u,v) # 3x2 operator
    mapped_vector = pushforward.dot(np.array([p,q])) # 2D direction vector mapped to R3
    
Alternatively one can carry out the pushforward of a vector at position `u,v` in one instruction (without accessing to the pushforward operator directly):
    
    vec = [v1,v2] # R2 vector in the parameter space
    mapped_vec = turtle.space.pushforward(u,v,vec) # vector vec pushedforward in R3

### Coefficients of the fundamental forms

	# Computed numerical coef o the fundamental forms I and II
	E,F,G,L,M,N = turtle.space.fundFormCoef(u,v)
    
### Cumulated rotation since the beginning of a 


This makes it possible to quantify parallel transport of vectors along turtle paths:

    cumrotation = turtle.cumrotation
    
### Testing whether domain boundaries are reached by the turtle

Test the boolean value (if true the boundaries of the domain have been reached):

    turtle.boundary_reached

### Querying the turtle's state from productions 

The turtle state can only be accessed within interpretation rules. However, one sometimes needs to take a growth decision in the production rule related to the turtle state associated with the last interpretation (i.e. interpretation of the last derived L-string). 

This can be carried out using the following query modules:

    nproduce ?T(parameters)
    nproduce ?UVPQ(uvpq)
    nproduce ?UV(uv)
    nproduce ?PQ(pq)

Example:

    Axiom: 
        nproduce ... ?T(ParameterSet())A(N)   # ParameterSet() = empty parameter set
    productions:
    ?T(t)A(n):
    
        # ?T(t) contains the last state of the turtle 
        # (corresponding to the last interpretation,
        # just before the current derivation, in particular e.g. uvpq
        
        u0,v0,p0,q0 = t.uvpq
        ...                                     # computation using e.g. u0,v0,p0,q0
        nproduce ... ?T(ParameterSet())A(n-1)
    

### Computing the length of a path on the surface

It is assumed that the path is given by an array uvpqs of uvpq coordinates (as returned for instance by space.forward() operator - see Forward movement above).

Then the length of this trajectory can be computed by:

    length = turtle.space.forward(uvpqs)


## Displaying fields

### Points

To draw a point as a small sphere on the surface use:

    radius = 0.3
    uv = [1,2.2]
    nproduce _(0.03)DrawPoint(uv,radius)

### Vectors
There are two ways to draw vectors. They can be either specified with coordinates in the surface local covariant basis, or directly in R3:

Vector vpq = [p,q] specified in the surface local covariant basis:

    nproduce _(0.03)DrawVectpq(vpq)
    # or
    scaling = 0.3
    nproduce _(0.03)DrawVectpq(v,scaling)
    
For a 3D vector v = [x,y,z] (can be a Vector3 or a Numpy array)

    nproduce _(0.03)DrawVect(v)
    # or
    scaling = 0.3
    nproduce _(0.03)DrawVect(v,scaling)
    
    
### Covariant basis

    scaling = 0.3
    nproduce CovariantBasis(scaling)
    
### Draw a vector parallel transported along a turtle path 

Let `rel_angle` be the initial angle between the transported vector and the Head of the turtle. Then at any moment on the turtle path, one can draw the transported vector:

    nproduce ParallelTransportedArrow(rel_angle,0.5)
    
### Drawing a polygon defined in the uv space

Define a polygon in the uv-space and fill the corresponding polygon on the surface in 3D:

    nproduce ClosedPolygon(polyline_uv, resolution, ccw)

the last two parameters are optional float and boolean.

## Draw a (2,2) tensor using two perpendicular lines

* `pdirs` = [pdirmax,pdirmin] are the two principal directions (eigen vectors)
* `eigenvals` = [evalmax,evalmin]are the two eigenvals
* `scalefactor` makes it to adapt the drawing scale to the tensor amplitudes
* colors `color_plus,color_minus` reflect the positive values of the eigenvals
*

    nproduce DrawTensor2Lines(pdirs,eigenvals,scalefactor,color_plus,color_minus)
                              
    
## Geodesic distances between two points

To get the geodesic distance on a surface between a point [u,v] and a target point [ut,vt], use:

    distance, errarray, errorval = geodesic_distance_to_point(turtle.space,
                                   (u,v),(ut,vt), 
                                   nb_points = NB_POINTS, 
                                   max_iter= MAXITER)

or directly from the curved space:

    dist, _ = turtle.space.geodesic_distance((u,v),(ut,vt))
    
## Turtle in abstract Riemannian spaces

A 2-D abstract Riemannian space is defined using a metric, boundary values for the parameter domain, and parameters that are used by the 
metric model.

The metric tensor is passed to the RiemannianSpace2D constructor using a dictionary of user-defined functions as its first argument:

    params = (source, ry, rz)     # parameters used by the model defining the metric
    nproduce SetSpace(RiemannianSpace2D(**metric_funcs, 
             umin = -1, umax = 1., vmin = 0, vmax = 2,
             metric_tensor_params = params))

The dictionary of function is defined as follows (but see the note below):

    metric_funcs = {'g11' : g11, 'g12' : g12, 'g22' : g22}

where the metric functions must have the following arguments: the position u,v as well as optional named parameters and return a positive scalar value, e.g.

    def g11(u,v,*args):
        r, alpha = metric_model(u, v, *args)
        return alpha * r
    def g12(u,v,*args):
        ...

For a hyperbolic metric model (half Poincaré-Beltrami plane) for example, the metric will be defined as follows:
 
    def g11_hyperbol(u,v,*args):
        return 1./v**2
    def g12_hyperbol(u,v,*args):        # g12 == g21
        return 0.
    def g22_hyperbol(u,v,*args):
        return 1./v**2

    metric_funcs = {'g11' : g11_hyperbol, 'g12' : g12_hyperbol, 'g22' : g22_hyperbol}
    
**Important Note**: 
For the moment (summer 2023), all the metric functions must be 'doubled' to define a swapping of the arguments necessary to handle the metric differentiation. This will be removed in the future releases of Riemannian L-Py. For now, the actual metric dictonary must thus be:

    metric_funcs = {'g11' : g11, 'g12' : g12, 'g22' : g22, 
                   'g11s' : g11s, 'g12s' : g12s, 'g22s' : g22s}

where the g11s ('s' for 'swapped arguments') must simply be defined from the g11 with swapped arguments:

    def g11s(v,u,*args):
        return g11(u,v,*args)
    ...

### Plot options of an abstract Riemannian space

The space in this case is defined by the domain of the u,v parameters.
As for surfaces it can be displayed using the module:

    nproduce PlotSpace

Optionally, one can display the local metric using small spheres/ellipses by using the global Flags at the begining of the file :

    nproduce PlotDS2

The size of the spheres/ellipses can be controlled using a scaling factor, e.g. `DS2_SCALEFACTOR = 4` will display sphere with a 4-fold diameter:

    nproduce PlotDS2(dict(DS2_SCALEFACTOR = 4))
    







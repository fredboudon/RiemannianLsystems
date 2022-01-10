"""
    Classes and functions for manipulating a Riemannian Turtle in LP-y

    Author: C. Godin, Inria
    Date: 2019-2021
    Lab: RDP ENS de Lyon, Mosaic Inria Team

"""
from surfaces import *

from scipy.integrate import odeint

def riemannian_turtle_init(uvpq,space):
  u,v,p,q = uvpq

  # Compute shitf tensor at u,v, then use it to transform p,q vector
  # representing the coordinates of the covariant basis on the curve
  # expressed in the surface covariant basis
  h = space.shift_vector(u,v,p,q)
  head = h / np.linalg.norm(h)
  up = np.array(space.normal(u,v))
  #print('head,up=',(head,up))
  return head,up

def riemannian_turtle_move_forward(p_uvpq,surf,ds,SUBDIV = 10):

  uu,vv,pp,qq = p_uvpq
  # print("Norm pq", np.linalg.norm(np.array([pp,qq])))

  npq = np.array([pp,qq])

  #if endflag and deviation_angle != 0 :
  #  print ("uvpq just after turning = ", [uu,vv,npq[0:2]] )
  #  print ("pq norm before rot = ", surf.norm(uu,vv,np.array([pp,qq])))
  #  print ("pq norm after rot  = ", surf.norm(uu,vv,npq[0:2]) )

  # NEXT POS: compute turtle's next position and orientation
  # and update the turtle's current state

  # to express the distance to move forward from coords expressed in the curve covariant basis

  n_ds = ds / surf.norm(uu,vv,npq[0:2])
  #print("*** ds = ",ds, " n_ds = ", n_ds)

  s = np.linspace(0,n_ds,SUBDIV)

  # computes the new state by integration of the geodesic equation
  # uvpq_s = odeint(pturtle_state.surf.geodesic_eq,pturtle_state.uvpq,s)
  #print("BEFORE: ", np.array([uu,vv,npq[0],npq[1]]))
  #print("s (len = ", len(s),"):", s)

  uvpq_s = odeint(surf.geodesic_eq,np.array([uu,vv,npq[0],npq[1]]),s)

  '''
  #print("AFTER(len = ", len(uvpq_s),"): ", uvpq_s[SUBDIV-1])
  #stores in u,v,p,q
  uvpq = uvpq_s[SUBDIV-1]
  u,v,p,q = uvpq

  # COVARIANT BASIS
  # Compute shitf tensor at u,v,
  A = surf.Shift(u,v) # Compute shitf tensor at the new u,v
  S1 = A[:,0] # first colum of A = first surface covariant vector in ambiant space
  S2 = A[:,1] # second column of A = second surface covariant vector in ambiant space

  # TURTLE FRAME:
  # uses the shift tensor to transform the p,q vector
  # representing the coordinates of the covariant basis "on the curve" (one vector)
  # expressed in the "surface" covariant basis
  h = A.dot(np.array([p,q]))
  head = h / np.linalg.norm(h)
  up = np.array(surf.normal(u,v))
  #print ("head = ",head, " head norm = ", np.linalg.norm(head))
  #print ("up = ",up, " up norm = ", np.linalg.norm(up))
  '''

  return uvpq_s


def riemannian_turtle_turn(p_uvpq,surf,deviation_angle):

  u,v,p,q = p_uvpq
  # print("Norm pq", np.linalg.norm(np.array([pp,qq])))

  # ROTATION : Turn turtle with deviation angle
  angle_rad = np.radians(deviation_angle)

  # a surface vector is defined by (pp,qq) at point (uu,vv)
  # the primitive returns
  uu,vv,pp,qq = surf.rotate_surface_vector(u,v,p,q,angle_rad)

  '''
  if not np.isclose(deviation_angle, 0.):
    p_npq = np.array([pp,qq])
    # FIXME: test and handle the case where the covariant basis is not degenerated (det = 0). This can occur if one of the basis vector for instance get to 0 (case of the sphere at the poles)
    # 1. computes an orthogonal frame from the local covariant basis using Gram-Schmidt orthog. principle)
    # 2. computes the components of the direction vector (given in the local covariant basis as [pp,qq]),
    # in the orthonormal frame
    # 3. then perform the rotation of the vector by an angle deviation_angle
    # 4. finally, transforms back the resulting vector in the covariant basis


    # Note: due to the contravariance of vector components,
    # the matrix transformation R = P R' P^-1
    # where R' is the known rotation matrix in the orthonormal basis
    # and P is the passage matrix from the covariant basis to the orthonormal basis
    S1, S2 = self.covariant_basis(u, v)
    len1 = np.linalg.norm(S1)
    if np.isclose(len1, 0.):
      # determine new p,q using geodesic equ.
      print("length basis vector is 0 !!!")
      npq1 = surf.passage_matrix_cb2ortho_inverse(uu, vv).dot(p_npq)
      npq2 = rotation_mat(angle_rad).dot(npq1)
      npq = surf.passage_matrix_cb2ortho(uu, vv).dot(npq2)
    else:
      npq1 = surf.passage_matrix_cb2ortho_inverse(uu,vv).dot(p_npq)
      npq2 = rotation_mat(angle_rad).dot(npq1)
      npq  = surf.passage_matrix_cb2ortho(uu,vv).dot(npq2)

    #J1 = surf.passage_matrix_cb2ortho(uu,vv)
    #J2 = surf.passage_matrix_cb2ortho_inverse(uu,vv)
    #Jprod = np.dot(J1,J2)
    #print ("J1 :", J1)
    #print ("J2 :", J2)
    #print ("J1*J2 = ", Jprod)
  else:
    npq = np.array([pp,qq])
  '''

  uvpq = np.array([uu,vv,pp,qq])

  return uvpq


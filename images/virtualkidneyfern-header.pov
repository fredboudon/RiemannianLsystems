/*
 * A povray file generated with GEOM.
 * Example of use : povray -Ifile.pov -Ofile.png +FN +H600 +W800 +A.
 * File Generated with PlantGL 3D Viewer.
 */


#ifndef (__camera_definition__)
#declare __camera_definition__ = true;

/*
camera {
   perspective
    location <35,-0.7,2.86518>
    direction <-1,-0,-0>
    up <0,0,1>
    right <0,4/3,0>
    angle 30
    rotate <0,-5,30>
}

camera {
   perspective
    location <40,-0.7,2.86518>
    direction <-1,-0,-0>
    up <0,0,1>
    right <0,4/3,0>
    angle 30
    // rotate <0,-5,30>
    rotate <0,-60,0>
}


camera {
   perspective
    location <35,-0.7,2.86518>
    direction <-1,-0,-0>
    up <0,0,1>
    right <0,4/3,0>
    angle 30
    rotate <0,-5,100>
}

*/

camera {
   perspective
    location <35,-0.7,2.86518>
    direction <-1,-0,-0>
    up <0,0,1>
    right <0,4/3,0>
    angle 30
    // rotate <0,-5,30>
    rotate <0,-95,0>
}
light_source {
     <-20,0,50>
    color rgb <0.8,1,0.8>
}



/*
light_source {
     <35,1.61505,5.19734>
    color rgb 1
}
*/


background { color rgb <1,1,1> }


#end // __camera_definition__


#declare Color_2 = texture {
  pigment {
    color rgbt <0.15,0.5,0.0,0.1>
  }
    finish {
    ambient 0.6
    diffuse 1.
    specular 0.
  }
}

#declare Color_4 = texture {
  pigment {
    color rgbt <0.6,0.85,0.24,0.>
  }
    finish {
    ambient 1
    diffuse 0.33
    specular 0
  }
}


#declare Color_tige = texture {
  pigment {
    color rgbt <0.117647,0.235294,0.0392157,0>
  }
    finish {
    ambient 1
    diffuse 3
    specular 0
  }
}

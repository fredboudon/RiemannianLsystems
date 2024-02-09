/*
 * A povray file generated with GEOM.
 * Example of use : povray -Ifile.pov -Ofile.png +FN +H600 +W800 +A.
 * File Generated with PlantGL 3D Viewer.
 */


#ifndef (__camera_definition__)
#declare __camera_definition__ = true;

#if (frame_number > 124)
#declare camera_rotation = (frame_number - 124)*5;
#else
#declare camera_rotation = 0;
#end

camera {
   perspective
    location <35,-0.7,2.86518>
    direction <-1,-0,-0>
    up <0,0,1>
    right <0,4/3,0>
    angle 30
    rotate <0,-5,30+camera_rotation>
}

/*
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

*/
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

/*
#declare Color_2 = texture {
  pigment {
    color rgbt <0.15,0.5,0.0,0.1>
    // 63,212,0,0.15
    // diffuse 1.66 
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
    // 153,217,61
    // diffuse 0.33
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
    // 33, 60, 10
  }
    finish {
    ambient 1
    diffuse 3
    specular 0
  }
}
*/

#switch (frame_number)
  #range (0,9)
    #declare frame_number_str = concat("00",str(frame_number,1,0));
  #break
  #range (10,99)
    #declare frame_number_str = concat("0",str(frame_number,2,0));
  #break
  #range (10,99)
    #declare frame_number_str = str(frame_number,3,0);
  #break
  #else
    #declare frame_number_str = "124";
  #break
#end

#include concat("/Users/fboudon/Develop/oagit/riemannien-l-systems/images/vkidneyfern/kidneyfern_",frame_number_str,".pov")


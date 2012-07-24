/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

OUTPIXELTYPE Functor(const INPIXELTYPE in)
{
  // Cast it and return
  OUTPIXELTYPE out = (OUTPIXELTYPE)in;
  return out;
}

#ifdef DIM_1
__kernel void CastImageFilter(__global const INPIXELTYPE* in,
                              __global OUTPIXELTYPE* out,
                              int width)
{
  int gix = get_global_id(0);
  if(gix < width)
  {
    out[gix] = Functor(in[gix]);
  }
}
#endif

#ifdef DIM_2
__kernel void CastImageFilter(__global const INPIXELTYPE* in,
                              __global OUTPIXELTYPE* out,
                              int width, int height)
{
  int gix = get_global_id(0);
  int giy = get_global_id(1);
  if(gix < width && giy < height)
  {
    unsigned int gidx = width*giy + gix;
    out[gidx] = Functor(in[gidx]);
  }
}
#endif

#ifdef DIM_3
__kernel void CastImageFilter(__global const INPIXELTYPE* in,
                              __global OUTPIXELTYPE* out,
                              int width, int height, int depth)
{
  int gix = get_global_id(0);
  int giy = get_global_id(1);
  int giz = get_global_id(2);

  /* NOTE: More than three-level nested conditional statements (e.g.,
  if A && B && C..) invalidates command queue during kernel
  execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
  GT). Therefore, we flattened conditional statements. */
  bool isValid = true;
  if(gix < 0 || gix >= width) isValid = false;
  if(giy < 0 || giy >= height) isValid = false;
  if(giz < 0 || giz >= depth) isValid = false;

  if( isValid )
  {
    unsigned int gidx = width*(giz*height + giy) + gix;
    out[gidx] = Functor(in[gidx]);
  }
}
#endif

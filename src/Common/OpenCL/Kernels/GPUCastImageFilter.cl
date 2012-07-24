/*=========================================================================
*
*  Copyright Insight Software Consortium
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*         http://www.apache.org/licenses/LICENSE-2.0.txt
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
*=========================================================================*/

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

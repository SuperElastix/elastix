/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

//------------------------------------------------------------------------------
// Apple OpenCL 1.0 support function
bool is_valid_3d(const uint3 index, const uint3 size)
{
  /* NOTE: More than three-level nested conditional statements (e.g.,
  if A && B && C..) invalidates command queue during kernel
  execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
  GT). Therefore, we flattened conditional statements. */
  if(index.x >= size.x){ return false; }
  if(index.y >= size.y){ return false; }
  if(index.z >= size.z){ return false; }
  return true;
}

//------------------------------------------------------------------------------
OUTPIXELTYPE Functor(const INPIXELTYPE in)
{
  // Cast it and return
  OUTPIXELTYPE out = (OUTPIXELTYPE)in;

  return out;
}

//------------------------------------------------------------------------------
#ifdef DIM_1
__kernel void CastImageFilter(__global const INPIXELTYPE *in,
                              __global OUTPIXELTYPE *out,
                              uint width)
{
  uint index = get_global_id(0);

  if(index < width)
  {
    out[index] = Functor(in[index]);
  }
}

#endif

//------------------------------------------------------------------------------
#ifdef DIM_2
__kernel void CastImageFilter(__global const INPIXELTYPE *in,
                              __global OUTPIXELTYPE *out,
                              uint width, uint height)
{
  uint2 index = (uint2)( get_global_id(0), get_global_id(1) );
  uint2 size = (uint2)(width, height);

  if(index.x < width && index.y < height)
  {
    uint gidx = mad24(size.x, index.y, index.x);
    out[gidx] = Functor(in[gidx]);
  }
}

#endif

//------------------------------------------------------------------------------
#ifdef DIM_3
__kernel void CastImageFilter(__global const INPIXELTYPE *in,
                              __global OUTPIXELTYPE *out,
                              uint width, uint height, uint depth)
{
  uint3 index = (uint3)( get_global_id(0), get_global_id(1), get_global_id(2) );
  uint3 size = (uint3)(width, height, depth);

  if( is_valid_3d(index, size) )
  {
    uint gidx = mad24(size.x, mad24(index.z, size.y, index.y), index.x);
    out[gidx] = Functor(in[gidx]);
  }
}

#endif

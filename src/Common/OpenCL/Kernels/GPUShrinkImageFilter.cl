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
  OUTPIXELTYPE out = (OUTPIXELTYPE)in;
  return out;
}

#ifdef DIM_1
__kernel void ShrinkImageFilter(__global const INPIXELTYPE* in,
                                __global OUTPIXELTYPE* out,
                                uint in_image_size, uint out_image_size,
                                uint offset, uint shrinkfactors)
{
  uint index = get_global_id(0);
  if(index < out_image_size)
  {
    uint input_index = index * shrinkfactors + offset;
    uint out_gidx = index;
    uint in_gidx = input_index;
    out[out_gidx] = Functor(in[in_gidx]);
  }
}
#endif

#ifdef DIM_2
__kernel void ShrinkImageFilter(__global const INPIXELTYPE* in, 
                                __global OUTPIXELTYPE* out, 
                                uint2 in_image_size, uint2 out_image_size,
                                uint2 offset, uint2 shrinkfactors)
{
  uint2 index = (uint2)(get_global_id(0), get_global_id(1));
  if(index.x < out_image_size.x && index.y < out_image_size.y)
  {
    uint2 input_index = index * shrinkfactors + offset;
    uint out_gidx = out_image_size.x * index.y + index.x;
    uint in_gidx = in_image_size.x * input_index.y + input_index.x;
    out[out_gidx] = Functor(in[in_gidx]);
  }
}
#endif

#ifdef DIM_3
__kernel void ShrinkImageFilter(__global const INPIXELTYPE* in,
                                __global OUTPIXELTYPE* out,
                                uint3 in_image_size, uint3 out_image_size,
                                uint3 offset, uint3 shrinkfactors)
{
  uint3 index = (uint3)(get_global_id(0), get_global_id(1), get_global_id(2));

  /* NOTE: More than three-level nested conditional statements (e.g.,
  if A && B && C..) invalidates command queue during kernel
  execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
  GT). Therefore, we flattened conditional statements. */
  bool isValid = true;
  if(index.x >= out_image_size.x) isValid = false;
  if(index.y >= out_image_size.y) isValid = false;
  if(index.z >= out_image_size.z) isValid = false;
  if(isValid)
  {
    uint3 input_index = index * shrinkfactors + offset;
    uint out_gidx = out_image_size.x * (index.z * out_image_size.y + index.y) + index.x;
    uint in_gidx = in_image_size.x * (input_index.z * in_image_size.y + input_index.y) + input_index.x;
    out[out_gidx] = Functor(in[in_gidx]);
  }
}
#endif

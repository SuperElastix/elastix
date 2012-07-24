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

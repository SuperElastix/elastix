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

// OpenCL implementation of GPU ImageBase Test kernels

//------------------------------------------------------------------------------
__kernel void GPUImageBaseTest2D(/* input ImageBase information */
                                 __global const INPIXELTYPE* in,
                                 const float2 spacing,
                                 const float2 origin,
                                 const uint2  size,
                                 const float3 direction,
                                 const float3 indexToPhysicalPoint,
                                 const float3 physicalPointToIndex,
                                 /* output ImageBase information */
                                 __global OUTPIXELTYPE* out,
                                 unsigned int nElem)
{
  uint2 index = (uint2)(get_global_id(0), get_global_id(1));
  unsigned int width = get_global_size(0);
  unsigned int gidx = index.y * width + index.x;

  // setup image base
  GPUImageBase2D image;
  set_image_base_info_2d(&image, spacing, origin, size,
    direction, indexToPhysicalPoint, physicalPointToIndex);

  // bound check
  if (gidx < nElem)
  {
    // spacing
    if(index.y == 0)
      out[gidx] = (OUTPIXELTYPE)(image.Spacing.x);
    if(index.y == 1)
      out[gidx] = (OUTPIXELTYPE)(image.Spacing.y);

    // origin
    if(index.y == 2)
      out[gidx] = (OUTPIXELTYPE)(image.Origin.x);
    if(index.y == 3)
      out[gidx] = (OUTPIXELTYPE)(image.Origin.y);

    // size
    if(index.y == 4)
      out[gidx] = (OUTPIXELTYPE)(image.Size.x);
    if(index.y == 5)
      out[gidx] = (OUTPIXELTYPE)(image.Size.y);

    // direction
    if(index.y == 6)
      out[gidx] = (OUTPIXELTYPE)(image.Direction.s0);
    if(index.y == 7)
      out[gidx] = (OUTPIXELTYPE)(image.Direction.s1);
    if(index.y == 8)
      out[gidx] = (OUTPIXELTYPE)(image.Direction.s2);
    if(index.y == 9)
      out[gidx] = (OUTPIXELTYPE)(image.Direction.s3);

    // indexToPhysicalPoint
    if(index.y == 10)
      out[gidx] = (OUTPIXELTYPE)(image.IndexToPhysicalPoint.s0);
    if(index.y == 11)
      out[gidx] = (OUTPIXELTYPE)(image.IndexToPhysicalPoint.s1);
    if(index.y == 12)
      out[gidx] = (OUTPIXELTYPE)(image.IndexToPhysicalPoint.s2);
    if(index.y == 13)
      out[gidx] = (OUTPIXELTYPE)(image.IndexToPhysicalPoint.s3);

    // physicalPointToIndex
    if(index.y == 14)
      out[gidx] = (OUTPIXELTYPE)(image.PhysicalPointToIndex.s0);
    if(index.y == 15)
      out[gidx] = (OUTPIXELTYPE)(image.PhysicalPointToIndex.s1);
    if(index.y == 16)
      out[gidx] = (OUTPIXELTYPE)(image.PhysicalPointToIndex.s2);
    if(index.y == 17)
      out[gidx] = (OUTPIXELTYPE)(image.PhysicalPointToIndex.s3);

    // check OpenCL implementation of transform_index_to_physical_point_2d()
    uint2 index_test1 = (uint2)(10, 25);
    float2 physical_point1 = transform_index_to_physical_point_2d(index_test1, &image);
    if(index.y == 18)
      out[gidx] = (OUTPIXELTYPE)(physical_point1.s0);
    if(index.y == 19)
      out[gidx] = (OUTPIXELTYPE)(physical_point1.s1);

    // check OpenCL implementation of transform_physical_point_to_continuous_index_2d()
    float2 point_test1 = (float2)(62.0, 11.0);
    float2 continuous_index1;
    int continuous_index1_valid = (int)transform_physical_point_to_continuous_index_2d(point_test1, &continuous_index1, &image);

    if(index.y == 20)
      out[gidx] = (OUTPIXELTYPE)(continuous_index1.s0);
    if(index.y == 21)
      out[gidx] = (OUTPIXELTYPE)(continuous_index1.s1);
    if(index.y == 22)
      out[gidx] = (OUTPIXELTYPE)(continuous_index1_valid);

    float2 point_test2 = (float2)(-67.4, 13.2);
    float2 continuous_index2;
    int continuous_index2_valid = (int)transform_physical_point_to_continuous_index_2d(point_test2, &continuous_index2, &image);

    if(index.y == 23)
      out[gidx] = (OUTPIXELTYPE)(continuous_index2.s0);
    if(index.y == 24)
      out[gidx] = (OUTPIXELTYPE)(continuous_index2.s1);
    if(index.y == 25)
      out[gidx] = (OUTPIXELTYPE)(continuous_index2_valid);
  }
}

/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
//
// \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
// Department of Radiology, Leiden, The Netherlands
//
// \note This work was funded by the Netherlands Organisation for
// Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
//
// OpenCL implementation of itk::NearestNeighborInterpolateImageFunction

//------------------------------------------------------------------------------
#ifdef DIM_1
float evaluate_at_continuous_index_1d(
  const float index,
  __global const INPIXELTYPE *in,
  const uint size,
  const int start_index,
  const int end_index )
{
  uint  nindex = convert_continuous_index_to_nearest_index_1d( index );

  uint  gidx = nindex;
  float image_value = (float)( in[gidx] );
  return image_value;
}
#endif // DIM_1

//------------------------------------------------------------------------------
#ifdef DIM_2
float evaluate_at_continuous_index_2d(
  const float2 index,
  __global const INPIXELTYPE *in,
  const uint2 size,
  const int2 start_index,
  const int2 end_index )
{
  uint2 nindex = convert_continuous_index_to_nearest_index_2d( index );

  uint  gidx = mad24( size.x, nindex.y, nindex.x );
  float image_value = (float)( in[gidx] );
  return image_value;
}
#endif // DIM_2

//------------------------------------------------------------------------------
#ifdef DIM_3
float evaluate_at_continuous_index_3d(
  const float3 index,
  __global const INPIXELTYPE *in,
  const uint3 size,
  const int3 start_index,
  const int3 end_index )
{
  uint3 nindex = convert_continuous_index_to_nearest_index_3d( index );

  uint  gidx = mad24( size.x, mad24( nindex.z, size.y, nindex.y ), nindex.x );
  float image_value = (float)( in[gidx] );
  return image_value;
}
#endif // DIM_3

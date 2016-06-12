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
// OpenCL implementation of itk::ImageFunction

//------------------------------------------------------------------------------
// Definition of GPUFunction 1D/2D/3D
typedef struct {
  int start_index;
  int end_index;
  float start_continuous_index;
  float end_continuous_index;
} GPUImageFunction1D;

typedef struct {
  int2 start_index;
  int2 end_index;
  float2 start_continuous_index;
  float2 end_continuous_index;
} GPUImageFunction2D;

typedef struct {
  int3 start_index;
  int3 end_index;
  float3 start_continuous_index;
  float3 end_continuous_index;
} GPUImageFunction3D;


//------------------------------------------------------------------------------
#ifdef DIM_1
bool interpolator_is_inside_buffer_1d(
  const float cindex,
  const float start_continuous_index,
  const float end_continuous_index )
{
  if( !( cindex >= start_continuous_index
      && cindex <  end_continuous_index ) )
  {
    return false;
  }

  return true;
}
#endif // DIM_1

//------------------------------------------------------------------------------
#ifdef DIM_2
bool interpolator_is_inside_buffer_2d(
  const float2 cindex,
  const float2 start_continuous_index,
  const float2 end_continuous_index )
{
  if( !( cindex.x >= start_continuous_index.x
      && cindex.x <  end_continuous_index.x ) )
  {
    return false;
  }
  if( !( cindex.y >= start_continuous_index.y
      && cindex.y <  end_continuous_index.y ) )
  {
    return false;
  }

  return true;
}
#endif // DIM_2

//------------------------------------------------------------------------------
#ifdef DIM_3
bool interpolator_is_inside_buffer_3d(
  const float3 cindex,
  const float3 start_continuous_index,
  const float3 end_continuous_index )
{
  if( !( cindex.x >= start_continuous_index.x
      && cindex.x <  end_continuous_index.x ) )
  {
    return false;
  }
  if( !( cindex.y >= start_continuous_index.y
      && cindex.y <  end_continuous_index.y ) )
  {
    return false;
  }
  if( !( cindex.z >= start_continuous_index.z
      && cindex.z <  end_continuous_index.z ) )
  {
    return false;
  }

  return true;
}
#endif // DIM_3

//------------------------------------------------------------------------------
// OpenCL 1D implementation of
// itkImageFunction::ConvertContinuousIndexToNearestIndex()
#ifdef DIM_1
uint convert_continuous_index_to_nearest_index_1d( const float cindex )
{
  return round( cindex );
}
#endif // DIM_1

//------------------------------------------------------------------------------
// OpenCL 2D implementation of
// itkImageFunction::ConvertContinuousIndexToNearestIndex()
#ifdef DIM_2
uint2 convert_continuous_index_to_nearest_index_2d( const float2 cindex )
{
  uint2 index;
  index.x = round( cindex.x );
  index.y = round( cindex.y );

  return index;
}
#endif // DIM_2

//------------------------------------------------------------------------------
// OpenCL 3D implementation of
// itkImageFunction::ConvertContinuousIndexToNearestIndex()
#ifdef DIM_3
uint3 convert_continuous_index_to_nearest_index_3d( const float3 cindex )
{
  uint3 index;
  index.x = round( cindex.x );
  index.y = round( cindex.y );
  index.z = round( cindex.z );

  return index;
}
#endif // DIM_3

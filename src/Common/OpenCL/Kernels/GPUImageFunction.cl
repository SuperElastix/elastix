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

// OpenCL implementation of itk::ImageFunction

//------------------------------------------------------------------------------
// Definition of GPUFunction 1D/2D/3D
typedef struct{
  uint   StartIndex;
  uint   EndIndex;
  float  StartContinuousIndex;
  float  EndContinuousIndex;
} GPUImageFunction1D;

typedef struct{
  uint2  StartIndex;
  uint2  EndIndex;
  float2 StartContinuousIndex;
  float2 EndContinuousIndex;
} GPUImageFunction2D;

typedef struct{
  uint3  StartIndex;           // OpenCL does not have uint3
  uint3  EndIndex;             // OpenCL does not have uint3
  float3 StartContinuousIndex; // OpenCL does not have float3
  float3 EndContinuousIndex;   // OpenCL does not have float3
} GPUImageFunction3D;

#ifdef DIM_1
//------------------------------------------------------------------------------
bool interpolator_is_inside_buffer_1d(const float index,
                                      __constant GPUImageFunction1D* image_function)
{
  if( ! (index >= image_function->StartContinuousIndex
    && index < image_function->EndContinuousIndex) )
    return false;

  return true;
}
#endif // DIM_1

#ifdef DIM_2
//------------------------------------------------------------------------------
bool interpolator_is_inside_buffer_2d(const float2 index,
                                      __constant GPUImageFunction2D* image_function)
{
  if( ! (index.x >= image_function->StartContinuousIndex.x
    && index.x < image_function->EndContinuousIndex.x) )
    return false;
  if( ! (index.y >= image_function->StartContinuousIndex.y
    && index.y < image_function->EndContinuousIndex.y) )
    return false;

  return true;
}
#endif // DIM_2

#ifdef DIM_3
//------------------------------------------------------------------------------
bool interpolator_is_inside_buffer_3d(const float3 index,
                                      __constant GPUImageFunction3D* image_function)
{
  if( ! (index.x >= image_function->StartContinuousIndex.x
    && index.x < image_function->EndContinuousIndex.x) )
    return false;
  if( ! (index.y >= image_function->StartContinuousIndex.y
    && index.y < image_function->EndContinuousIndex.y) )
    return false;
  if( ! (index.z >= image_function->StartContinuousIndex.z
    && index.z < image_function->EndContinuousIndex.z) )
    return false;

  return true;
}
#endif // DIM_3

//------------------------------------------------------------------------------
// OpenCL 1D implementation of
// itkImageFunction::ConvertContinuousIndexToNearestIndex()
uint convert_continuous_index_to_nearest_index_1d(const float cindex)
{
  return round(cindex);
}

//------------------------------------------------------------------------------
// OpenCL 2D implementation of
// itkImageFunction::ConvertContinuousIndexToNearestIndex()
uint2 convert_continuous_index_to_nearest_index_2d(const float2 cindex)
{
  uint2 index;
  index.x = round(cindex.x);
  index.y = round(cindex.y);
  return index;
}

//------------------------------------------------------------------------------
// OpenCL 3D implementation of
// itkImageFunction::ConvertContinuousIndexToNearestIndex()
uint3 convert_continuous_index_to_nearest_index_3d(const float3 cindex)
{
  uint3 index;
  index.x = round(cindex.x);
  index.y = round(cindex.y);
  index.z = round(cindex.z);
  return index;
}

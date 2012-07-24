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

// OpenCL implementation of itk::NearestNeighborInterpolateImageFunction

//------------------------------------------------------------------------------
float bspline_evaluate_at_continuous_index_1d(const float index,
                                              __global const INPIXELTYPE* in,
                                              __constant GPUImageBase1D *in_image,
                                              __constant GPUImageFunction1D* image_function,
                                              __global const INTERPOLATOR_PRECISION_TYPE* coefficients,
                                              __constant GPUImageBase1D *coefficients_image)
{
  return 0.0;
}

//------------------------------------------------------------------------------
float bspline_evaluate_at_continuous_index_2d(const float2 index,
                                              __global const INPIXELTYPE* in,
                                              __constant GPUImageBase2D *in_image,
                                              __constant GPUImageFunction2D* image_function,
                                              __global const INTERPOLATOR_PRECISION_TYPE* coefficients,
                                              __constant GPUImageBase2D *coefficients_image)
{
  return 0.0;
}

//------------------------------------------------------------------------------
float bspline_evaluate_at_continuous_index_3d(const float3 index,
                                              __global const INPIXELTYPE* in,
                                              __constant GPUImageBase3D *in_image,
                                              __constant GPUImageFunction3D* image_function,
                                              __global const INTERPOLATOR_PRECISION_TYPE* coefficients,
                                              __constant GPUImageBase3D *coefficients_image)
{
  return 0.0;
}

#ifdef DIM_1
//------------------------------------------------------------------------------
float evaluate_at_continuous_index_1d(const float index,
                                      __global const INPIXELTYPE* in,
                                      __constant GPUImageBase1D *image,
                                      __constant GPUImageFunction1D* image_function)
{
  uint nindex = convert_continuous_index_to_nearest_index_1d(index);
  unsigned int gidx = nindex;
  float image_value = (float)(in[gidx]);
  return image_value;
}
#endif // DIM_1

#ifdef DIM_2
//------------------------------------------------------------------------------
float evaluate_at_continuous_index_2d(const float2 index,
                                      __global const INPIXELTYPE* in,
                                      __constant GPUImageBase2D *image,
                                      __constant GPUImageFunction2D* image_function)
{
  uint2 nindex = convert_continuous_index_to_nearest_index_2d(index);
  unsigned int gidx = image->Size.x * nindex.y + nindex.x;
  float image_value = (float)(in[gidx]);
  return image_value;
}
#endif // DIM_2

#ifdef DIM_3
//------------------------------------------------------------------------------
float evaluate_at_continuous_index_3d(const float3 index,
                                      __global const INPIXELTYPE* in,
                                      __constant GPUImageBase3D *image,
                                      __constant GPUImageFunction3D* image_function)
{
  uint3 nindex = convert_continuous_index_to_nearest_index_3d(index);
  unsigned int gidx = image->Size.x *(nindex.z * image->Size.y + nindex.y) + nindex.x;
  float image_value = (float)(in[gidx]);
  return image_value;
}
#endif // DIM_3

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

// OpenCL implementation of itk::IdentityTransform

//------------------------------------------------------------------------------
#ifdef DIM_1
float bspline_transform_point_1d(const float point,
                                 __global const INTERPOLATOR_PRECISION_TYPE* coefficients0,
                                 __constant GPUImageBase1D *coefficients_image0)
{
  return point;
}
#endif // DIM_1

#ifdef DIM_2
float2 bspline_transform_point_2d(const float2 point,
                                  __global const INTERPOLATOR_PRECISION_TYPE* coefficients0,
                                  __constant GPUImageBase2D *coefficients_image0,
                                  __global const INTERPOLATOR_PRECISION_TYPE* coefficients1,
                                  __constant GPUImageBase2D *coefficients_image1)
{
  return point;
}
#endif // DIM_2

#ifdef DIM_3
float3 bspline_transform_point_3d(const float3 point,
                                  __global const INTERPOLATOR_PRECISION_TYPE* coefficients0,
                                  __constant GPUImageBase3D *coefficients_image0,
                                  __global const INTERPOLATOR_PRECISION_TYPE* coefficients1,
                                  __constant GPUImageBase3D *coefficients_image1,
                                  __global const INTERPOLATOR_PRECISION_TYPE* coefficients2,
                                  __constant GPUImageBase3D *coefficients_image2)
{
  return point;
}
#endif // DIM_3

#ifdef DIM_1
//------------------------------------------------------------------------------
float transform_point_1d(const float point,
                         __constant GPUMatrixOffsetTransformBase1D* transform_base)
{
  return point;
}
#endif // DIM_1

#ifdef DIM_2
//------------------------------------------------------------------------------
float2 transform_point_2d(const float2 point,
                          __constant GPUMatrixOffsetTransformBase2D* transform_base)
{
  return point;
}
#endif // DIM_2

#ifdef DIM_3
//------------------------------------------------------------------------------
float3 transform_point_3d(const float3 point,
                          __constant GPUMatrixOffsetTransformBase3D* transform_base)
{
  return point;
}
#endif // DIM_3

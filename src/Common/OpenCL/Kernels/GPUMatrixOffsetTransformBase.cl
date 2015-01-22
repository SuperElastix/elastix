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
// OpenCL implementation of itk::MatrixOffsetTransformBase

//------------------------------------------------------------------------------
// Definition of GPUMatrixOffsetTransformBase 1D/2D/3D
#ifdef DIM_1
typedef struct {
  float matrix;
  float inverse_matrix;
  float offset;
} GPUMatrixOffsetTransformBase1D;
#endif // DIM_1

#ifdef DIM_2
typedef struct {
  float4 matrix;
  float4 inverse_matrix;
  float2 offset;
} GPUMatrixOffsetTransformBase2D;
#endif // DIM_2

#ifdef DIM_3
typedef struct {
  float16 matrix;         // OpenCL does not have float9
  float16 inverse_matrix; // OpenCL does not have float9
  float3  offset;
} GPUMatrixOffsetTransformBase3D;
#endif // DIM_3

//------------------------------------------------------------------------------
#ifdef DIM_1
float matrix_offset_transform_point_1d(
  const float point,
  const float matrix,
  const float offset )
{
  return mad( matrix, point, offset );
}
#endif // DIM_1

//------------------------------------------------------------------------------
#ifdef DIM_2
float2 matrix_offset_transform_point_2d(
  const float2 point,
  const float4 matrix,
  const float2 offset )
{
  float2 tpoint = offset;

  float2 rowx = matrix.s01;
  float2 rowy = matrix.s23;

  tpoint.x += dot( rowx, point );
  tpoint.y += dot( rowy, point );

  return tpoint;
}
#endif // DIM_2

//------------------------------------------------------------------------------
#ifdef DIM_3
float3 matrix_offset_transform_point_3d(
  const float3 point,
  const float16 matrix,
  const float3 offset )
{
  float3 tpoint = offset.xyz;

  float3 rowx = matrix.s012;
  float3 rowy = matrix.s345;
  float3 rowz = matrix.s678;

  tpoint.x += dot( rowx, point );
  tpoint.y += dot( rowy, point );
  tpoint.z += dot( rowz, point );

  return tpoint;
}
#endif // DIM_3

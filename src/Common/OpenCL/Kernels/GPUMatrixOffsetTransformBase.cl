/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

// OpenCL implementation of itk::MatrixOffsetTransformBase

#define _ELASTIX_USE_OPENCL_MAD 0

// Definition of GPUMatrixOffsetTransformBase 1D/2D/3D
#ifdef DIM_1
typedef struct {
  float matrix;
  float offset;
  float inverse_matrix;
} GPUMatrixOffsetTransformBase1D;
#endif // DIM_1

#ifdef DIM_2
typedef struct {
  float4 matrix;
  float2 offset;
  float4 inverse_matrix;
} GPUMatrixOffsetTransformBase2D;
#endif // DIM_2

#ifdef DIM_3
typedef struct {
  float16 matrix;         // OpenCL does not have float9
  float3  offset;
  float16 inverse_matrix; // OpenCL does not have float9
} GPUMatrixOffsetTransformBase3D;
#endif // DIM_3

//------------------------------------------------------------------------------
#ifdef DIM_1
float matrix_offset_transform_point_1d(
  const float point,
  __constant GPUMatrixOffsetTransformBase1D *transform_base )
{
  float tpoint;

#if _ELASTIX_USE_OPENCL_MAD
  tpoint = mad( transform_base->matrix, point, transform_base->offset );
#else
  tpoint =  transform_base->matrix * point;
  tpoint += transform_base->offset;
#endif

  return tpoint;
}
#endif // DIM_1

//------------------------------------------------------------------------------
#ifdef DIM_2
float2 matrix_offset_transform_point_2d(
  const float2 point,
  __constant GPUMatrixOffsetTransformBase2D *transform_base )
{
  float2 tpoint;

#if _ELASTIX_USE_OPENCL_MAD
  tpoint.x = mad( transform_base->matrix.s0, point.x,
    mad( transform_base->matrix.s1, point.y, transform_base->offset.x );

  tpoint.y = mad( transform_base->matrix.s2, point.x,
    mad( transform_base->matrix.s3, point.y, transform_base->offset.y );
#else
  tpoint.x =  transform_base->matrix.s0 * point.x;
  tpoint.x += transform_base->matrix.s1 * point.y;

  tpoint.y =  transform_base->matrix.s2 * point.x;
  tpoint.y += transform_base->matrix.s3 * point.y;

  tpoint += transform_base->offset;
#endif

  return tpoint;
}
#endif // DIM_2

//------------------------------------------------------------------------------
#ifdef DIM_3
float3 matrix_offset_transform_point_3d(
  const float3 point,
  __constant GPUMatrixOffsetTransformBase3D *transform_base )
{
  float3 tpoint;
#if _ELASTIX_USE_OPENCL_MAD
  tpoint.x = mad( transform_base->matrix.s0, point.x,
    mad( transform_base->matrix.s1, point.y,
    mad( transform_base->matrix.s2, point.z, transform_base->offset.x ) ) );

  tpoint.y = mad( transform_base->matrix.s3, point.x,
    mad( transform_base->matrix.s4, point.y,
    mad( transform_base->matrix.s5, point.z, transform_base->offset.y ) ) );

  tpoint.z = mad( transform_base->matrix.s6, point.x,
    mad( transform_base->matrix.s7, point.y,
    mad( transform_base->matrix.s8, point.z, transform_base->offset.z ) ) );
#else
  tpoint.x =  transform_base->matrix.s0 * point.x;
  tpoint.x += transform_base->matrix.s1 * point.y;
  tpoint.x += transform_base->matrix.s2 * point.z;

  tpoint.y =  transform_base->matrix.s3 * point.x;
  tpoint.y += transform_base->matrix.s4 * point.y;
  tpoint.y += transform_base->matrix.s5 * point.z;

  tpoint.z =  transform_base->matrix.s6 * point.x;
  tpoint.z += transform_base->matrix.s7 * point.y;
  tpoint.z += transform_base->matrix.s8 * point.z;

  tpoint += transform_base->offset.xyz;
#endif

  return tpoint;
}
#endif // DIM_3


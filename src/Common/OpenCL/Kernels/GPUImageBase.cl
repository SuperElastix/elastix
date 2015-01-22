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
// OpenCL implementation of itk::ImageBase

//------------------------------------------------------------------------------
// Definition of GPUImageBase 1D/2D/3D
typedef struct {
  float direction;
  float index_to_physical_point;
  float physical_point_to_index;
  float spacing;
  float origin;
  uint size;
} GPUImageBase1D;

typedef struct {
  float4 direction;
  float4 index_to_physical_point;
  float4 physical_point_to_index;
  float2 spacing;
  float2 origin;
  uint2 size;
} GPUImageBase2D;

typedef struct {
  float16 direction;                // OpenCL does not have float9
  float16 index_to_physical_point;  // OpenCL does not have float9
  float16 physical_point_to_index;  // OpenCL does not have float9
  float3 spacing;
  float3 origin;
  uint3 size;
} GPUImageBase3D;

//------------------------------------------------------------------------------
// OpenCL 1D implementation of
// itkImageBase::ComputeIndexToPhysicalPointMatrices()
#ifdef DIM_1
void compute_index_to_physical_point_matrices_1d( GPUImageBase1D *image )
{
  image->index_to_physical_point = image->direction * image->spacing;
  image->physical_point_to_index = 1.0f / image->index_to_physical_point;
}
#endif // DIM_1

//------------------------------------------------------------------------------
// OpenCL 2D implementation of
// itkImageBase::ComputeIndexToPhysicalPointMatrices()
#ifdef DIM_2
void compute_index_to_physical_point_matrices_2d( GPUImageBase2D *image )
{
  // not implemented
}
#endif // DIM_2

//------------------------------------------------------------------------------
// OpenCL 3D implementation of
// itkImageBase::ComputeIndexToPhysicalPointMatrices()
#ifdef DIM_3
void compute_index_to_physical_point_matrices_3d( GPUImageBase3D *image )
{
  // not implemented
}
#endif // DIM_3

//------------------------------------------------------------------------------
// OpenCL 1D implementation of
// itkImageBase::SetSpacing(const SpacingType &spacing)
#ifdef DIM_1
void set_spacing_1d( const float spacing, GPUImageBase1D *image )
{
  if ( image->spacing != spacing )
  {
    image->spacing = spacing;
    compute_index_to_physical_point_matrices_1d( image );
  }
}
#endif // DIM_1

//------------------------------------------------------------------------------
// OpenCL 2D implementation of
// itkImageBase::SetSpacing(const SpacingType &spacing)
#ifdef DIM_2
void set_spacing_2d( const float2 spacing, GPUImageBase2D *image )
{
  const int2 is_not_equal = isnotequal( spacing, image->spacing );

  if ( any( is_not_equal ) )
  {
    image->spacing = spacing;
    compute_index_to_physical_point_matrices_2d( image );
  }
}
#endif // DIM_2

//------------------------------------------------------------------------------
// OpenCL 3D implementation of
// itkImageBase::SetSpacing(const SpacingType &spacing)
#ifdef DIM_3
void set_spacing_3d( const float3 spacing, GPUImageBase3D *image )
{
  const int3 isNotEqual = isnotequal( spacing, image->spacing );

  if ( any( isNotEqual ) )
  {
    image->spacing = spacing;
    compute_index_to_physical_point_matrices_3d( image );
  }
}
#endif // DIM_3

//------------------------------------------------------------------------------
// OpenCL 1D implementation of
// itkImageBase::SetDirection(const DirectionType direction)
#ifdef DIM_1
void set_direction_1d( const float direction, GPUImageBase1D *image )
{
  if ( image->direction != direction )
  {
    image->direction = direction;
    compute_index_to_physical_point_matrices_1d( image );
  }
}
#endif // DIM_1

//------------------------------------------------------------------------------
// OpenCL 2D implementation of
// itkImageBase::SetDirection(const DirectionType direction)
#ifdef DIM_2
void set_direction_2d( const float3 direction, GPUImageBase2D *image )
{
  const int3 is_not_equal = isnotequal( direction, image->direction.xyz );

  if ( any( is_not_equal ) )
  {
    image->direction.xyz = direction;
    compute_index_to_physical_point_matrices_2d( image );
  }
}
#endif // DIM_2

//------------------------------------------------------------------------------
// OpenCL 3D implementation of
// itkImageBase::SetDirection(const DirectionType direction)
#ifdef DIM_3
void set_direction_3d( const float16 direction, GPUImageBase3D *image )
{
  const int16 is_not_equal = isnotequal( direction, image->direction );

  if ( any( is_not_equal ) )
  {
    image->direction = direction;
    compute_index_to_physical_point_matrices_3d( image );
  }
}
#endif // DIM_3

//------------------------------------------------------------------------------
// OpenCL 1D implementation of
// itkImageRegion::IsInside(const ContinuousIndex< TCoordRepType,
// VImageDimension > &index)
#ifdef DIM_1
bool is_continuous_index_inside_1d( const float index, const uint size )
{
  int rounded;
  rounded = round( index );
  if ( rounded < 0 ) { return false; }

  float bound;
  bound = (float)( size ) - 0.5f;
  if ( index > bound ) { return false; }

  return true;
}
#endif // DIM_1

//------------------------------------------------------------------------------
// OpenCL 2D implementation of
// itkImageRegion::IsInside(const ContinuousIndex< TCoordRepType,
// VImageDimension > &index)
#ifdef DIM_2
bool is_continuous_index_inside_2d( const float2 index, const uint2 size )
{
  int2 rounded;
  rounded.x = round( index.x );
  rounded.y = round( index.y );
  if ( rounded.x < 0 || rounded.y < 0 ) { return false; }

  float2 bound;
  bound.x = (float)( size.x ) - 0.5f;
  bound.y = (float)( size.y ) - 0.5f;
  if ( index.x > bound.x || index.y > bound.y ) { return false; }

  return true;
}
#endif // DIM_2

//------------------------------------------------------------------------------
// OpenCL 3D implementation of
// itkImageRegion::IsInside(const ContinuousIndex< TCoordRepType,
// VImageDimension > &index)
#ifdef DIM_3
bool is_continuous_index_inside_3d( const float3 index, const uint3 size )
{
  int3 rounded;
  rounded.x = round( index.x );
  rounded.y = round( index.y );
  rounded.z = round( index.z );

  if ( rounded.x < 0 ) { return false; }
  if ( rounded.y < 0 ) { return false; }
  if ( rounded.z < 0 ) { return false; }

  float3 bound;
  bound.x = (float)( size.x ) - 0.5f;
  bound.y = (float)( size.y ) - 0.5f;
  bound.z = (float)( size.z ) - 0.5f;

  if ( index.x > bound.x ) { return false; }
  if ( index.y > bound.y ) { return false; }
  if ( index.z > bound.z ) { return false; }

  return true;
}
#endif // DIM_3

//------------------------------------------------------------------------------
// OpenCL 1D implementation of
// itkImageBase::TransformIndexToPhysicalPoint()
#ifdef DIM_1
float transform_index_to_physical_point_1d(
  const uint index,
  __constant const GPUImageBase1D *image )
{
  float point = image->origin;
  point += image->index_to_physical_point * index;

  return point;
}
#endif // DIM_1

//------------------------------------------------------------------------------
// OpenCL 1D implementation of
// itkImageBase::TransformIndexToPhysicalPoint()
#ifdef DIM_1
float transform_index_to_physical_point_1d_(
  const uint index,
  const float index_to_physical_point,
  const float origin )
{
  float point = origin;
  point += index_to_physical_point * index;

  return point;
}
#endif // DIM_1

//------------------------------------------------------------------------------
// OpenCL 2D implementation of
// itkImageBase::TransformIndexToPhysicalPoint()
#ifdef DIM_2
float2 transform_index_to_physical_point_2d(
  const uint2 index,
  __constant const GPUImageBase2D *image )
{
  float2 i2pp_x = image->index_to_physical_point.s01;
  float2 i2pp_y = image->index_to_physical_point.s23;

  float2 point = image->origin;
  point.x += dot( i2pp_x, convert_float2( index ) );
  point.y += dot( i2pp_y, convert_float2( index ) );

  return point;
}
#endif // DIM_2

//------------------------------------------------------------------------------
// OpenCL 2D implementation of
// itkImageBase::TransformIndexToPhysicalPoint()
#ifdef DIM_2
float2 transform_index_to_physical_point_2d_(
  const uint2  index,
  const float4 index_to_physical_point,
  const float2 origin )
{
  float2 index_as_float = convert_float2( index );

  float2 i2pp_x = index_to_physical_point.s01;
  float2 i2pp_y = index_to_physical_point.s23;

  float2 point = origin;
  point.x += dot( i2pp_x, index_as_float );
  point.y += dot( i2pp_y, index_as_float );

  return point;
}
#endif // DIM_2

//------------------------------------------------------------------------------
// OpenCL 3D implementation of
// itkImageBase::TransformIndexToPhysicalPoint()
#ifdef DIM_3
float3 transform_index_to_physical_point_3d(
  const uint3 index,
  __constant GPUImageBase3D *image )
{
  float3 index_as_float = convert_float3( index );

  float3 i2pp_x = image->index_to_physical_point.s012;
  float3 i2pp_y = image->index_to_physical_point.s345;
  float3 i2pp_z = image->index_to_physical_point.s678;

  float3 point = image->origin;
  point.x += dot( i2pp_x, index_as_float );
  point.y += dot( i2pp_y, index_as_float );
  point.z += dot( i2pp_z, index_as_float );

  return point;
}
#endif // DIM_3

//------------------------------------------------------------------------------
// OpenCL 3D implementation of
// itkImageBase::TransformIndexToPhysicalPoint()
#ifdef DIM_3
float3 transform_index_to_physical_point_3d_(
  const uint3   index,
  const float16 index_to_physical_point,  // OpenCL does not have float9
  const float3  origin )
{
  float3 index_as_float = convert_float3( index );

  float3 i2pp_x = index_to_physical_point.s012;
  float3 i2pp_y = index_to_physical_point.s345;
  float3 i2pp_z = index_to_physical_point.s678;

  float3 point = origin;
  point.x += dot( i2pp_x, index_as_float );
  point.y += dot( i2pp_y, index_as_float );
  point.z += dot( i2pp_z, index_as_float );

  return point;
}
#endif // DIM_3

//------------------------------------------------------------------------------
// OpenCL 1D implementation of
// itkImageBase::TransformPhysicalPointToContinuousIndex()
#ifdef DIM_1
bool transform_physical_point_to_continuous_index_1d(
  const float point, float *index,
  __constant const GPUImageBase1D *image )
{
  float cvector = point - image->origin;
  float cvector1;

  cvector1 = image->physical_point_to_index * cvector;
  *index = cvector1;
  return is_continuous_index_inside_1d( cvector1, image->size );
}
#endif // DIM_1

//------------------------------------------------------------------------------
// OpenCL 2D implementation of
// itkImageBase::TransformPhysicalPointToContinuousIndex()
#ifdef DIM_2
bool transform_physical_point_to_continuous_index_2d(
  const float2 point, float2 *index,
  __constant const GPUImageBase2D *image )
{
  float2 pp2i_x = image->physical_point_to_index.s01;
  float2 pp2i_y = image->physical_point_to_index.s23;

  float2 cvector = point - image->origin;

  float2 cvector1;

  cvector1.x = dot( pp2i_x, cvector );
  cvector1.y = dot( pp2i_y, cvector );

  *index = cvector1;
  return is_continuous_index_inside_2d( cvector1, image->size );
}
#endif // DIM_2

//------------------------------------------------------------------------------
// OpenCL 3D implementation of
// itkImageBase::TransformPhysicalPointToContinuousIndex()
#ifdef DIM_3
float3 transform_physical_point_to_continuous_index_3d(
  const float3 point,
  float16 physical_point_to_index, // OpenCL does not have float9
  float3  origin )
{
  // Extract the three columns
  float3 pp2i_x = physical_point_to_index.s012;
  float3 pp2i_y = physical_point_to_index.s345;
  float3 pp2i_z = physical_point_to_index.s678;

  // Transform to continuous index
  float3 cvector = point - origin;

  float3 cindex;
  cindex.x = dot( pp2i_x, cvector );
  cindex.y = dot( pp2i_y, cvector );
  cindex.z = dot( pp2i_z, cvector );

  // The corresponding ITK function returns is_inside, but this in
  // never used, so we omit it here and just return cindex.
  //return is_continuous_index_inside_3d( cvector1, image->size );
  return cindex;
}
#endif // DIM_3

//------------------------------------------------------------------------------
// OpenCL 1D implementation (long index, float return value) version of
// itkImage::GetPixel()
#ifdef DIM_1
float get_pixel_1d(
  const long index,
  __global const INPIXELTYPE *in,
  const uint size )
{
  float value = (float)( in[index] );
  return value;
}
#endif // DIM_1

//------------------------------------------------------------------------------
// OpenCL 2D implementation (long index, float return value) version of
// itkImage::GetPixel()
#ifdef DIM_2
float get_pixel_2d(
  const long2 index,
  __global const INPIXELTYPE *in,
  const uint2 size )
{
  uint  gidx = mad24( size.x, index.y, index.x );
  float value = (float)( in[gidx] );
  return value;
}
#endif // DIM_2

//------------------------------------------------------------------------------
// OpenCL 3D implementation (long index, float return value) version of
// itkImage::GetPixel()
#ifdef DIM_3
float get_pixel_3d(
  const long3 index,
  __global const INPIXELTYPE *in,
  const uint3 size )
{
  uint gidx = mad24( size.x, mad24( index.z, size.y, index.y ), index.x );
  float value = (float)( in[gidx] );
  return value;
}
#endif // DIM_3

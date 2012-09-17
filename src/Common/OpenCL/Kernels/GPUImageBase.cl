/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

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
void set_image_base_1d(GPUImageBase1D *image,
                       const float spacing,
                       const float origin,
                       const uint size,
                       const float direction,
                       const float index_to_physical_point,
                       const float physical_point_to_index)
{
  image->spacing   = spacing;
  image->origin    = origin;
  image->size      = size;
  image->direction = direction;
  image->index_to_physical_point = index_to_physical_point;
  image->physical_point_to_index = physical_point_to_index;
}

//------------------------------------------------------------------------------
void set_image_base_2d(GPUImageBase2D *image,
                       const float2 spacing,
                       const float2 origin,
                       const uint2 size,
                       const float4 direction,
                       const float4 index_to_physical_point,
                       const float4 physical_point_to_index)
{
  image->spacing   = spacing;
  image->origin    = origin;
  image->size      = size;
  image->direction = direction;
  image->index_to_physical_point = index_to_physical_point;
  image->physical_point_to_index = physical_point_to_index;
}

//------------------------------------------------------------------------------
void set_image_base_3d(GPUImageBase3D *image,
                       const float3 spacing,
                       const float3 origin,
                       const uint3 size,
                       const float16 direction,
                       const float16 index_to_physical_point,
                       const float16 physical_point_to_index)
{
  image->spacing   = spacing;
  image->origin    = origin;
  image->size      = size;
  image->direction = direction;
  image->index_to_physical_point = index_to_physical_point;
  image->physical_point_to_index = physical_point_to_index;
}

//------------------------------------------------------------------------------
// OpenCL 1D implementation of
// itkImageBase::ComputeIndexToPhysicalPointMatrices()
void compute_index_to_physical_point_matrices_1d(GPUImageBase1D *image)
{
  image->index_to_physical_point = image->direction * image->spacing;
  image->physical_point_to_index = 1.0f / image->index_to_physical_point;
}

//------------------------------------------------------------------------------
// OpenCL 2D implementation of
// itkImageBase::ComputeIndexToPhysicalPointMatrices()
void compute_index_to_physical_point_matrices_2d(GPUImageBase2D *image)
{
  // not implemented
}

//------------------------------------------------------------------------------
// OpenCL 3D implementation of
// itkImageBase::ComputeIndexToPhysicalPointMatrices()
void compute_index_to_physical_point_matrices_3d(GPUImageBase3D *image)
{
  // not implemented
}

//------------------------------------------------------------------------------
// OpenCL 1D implementation of
// itkImageBase::SetSpacing(const SpacingType &spacing)
void set_spacing_1d(const float spacing, GPUImageBase1D *image)
{
  if(image->spacing != spacing)
  {
    image->spacing = spacing;
    compute_index_to_physical_point_matrices_1d(image);
  }
}

//------------------------------------------------------------------------------
// OpenCL 2D implementation of
// itkImageBase::SetSpacing(const SpacingType &spacing)
void set_spacing_2d(const float2 spacing, GPUImageBase2D *image)
{
  const int2 is_not_equal = isnotequal(spacing, image->spacing);

  if( any(is_not_equal) )
  {
    image->spacing = spacing;
    compute_index_to_physical_point_matrices_2d(image);
  }
}

//------------------------------------------------------------------------------
// OpenCL 3D implementation of
// itkImageBase::SetSpacing(const SpacingType &spacing)
void set_spacing_3d(const float3 spacing, GPUImageBase3D *image)
{
  const int3 isNotEqual = isnotequal(spacing, image->spacing);

  if( any(isNotEqual) )
  {
    image->spacing = spacing;
    compute_index_to_physical_point_matrices_3d(image);
  }
}

//------------------------------------------------------------------------------
// OpenCL 1D implementation of
// itkImageBase::SetDirection(const DirectionType direction)
void set_direction_1d(const float direction, GPUImageBase1D *image)
{
  if(image->direction != direction)
  {
    image->direction = direction;
    compute_index_to_physical_point_matrices_1d(image);
  }
}

//------------------------------------------------------------------------------
// OpenCL 2D implementation of
// itkImageBase::SetDirection(const DirectionType direction)
void set_direction_2d(const float3 direction, GPUImageBase2D *image)
{
  const int3 is_not_equal = isnotequal(direction, image->direction.xyz);

  if( any(is_not_equal) )
  {
    image->direction.xyz = direction;
    compute_index_to_physical_point_matrices_2d(image);
  }
}

//------------------------------------------------------------------------------
// OpenCL 3D implementation of
// itkImageBase::SetDirection(const DirectionType direction)
void set_direction_3d(const float16 direction, GPUImageBase3D *image)
{
  const int16 is_not_equal = isnotequal(direction, image->direction);

  if( any(is_not_equal) )
  {
    image->direction = direction;
    compute_index_to_physical_point_matrices_3d(image);
  }
}

//------------------------------------------------------------------------------
// OpenCL 1D implementation of
// itkImageRegion::IsInside(const ContinuousIndex< TCoordRepType,
// VImageDimension > &index)
bool is_continuous_index_inside_1d(const float index, const uint size)
{
  int round_up;

  round_up = round_half_integer_up(index);
  if(round_up < 0)
  {
    return false;
  }

  float bound;
  bound = (float)size - 0.5f;
  if( !(index <= bound) )
  {
    return false;
  }

  return true;
}

//------------------------------------------------------------------------------
// OpenCL 2D implementation of
// itkImageRegion::IsInside(const ContinuousIndex< TCoordRepType,
// VImageDimension > &index)
bool is_continuous_index_inside_2d(const float2 index, const uint2 size)
{
  int2 round_up;

  round_up.x = round_half_integer_up(index.x);
  round_up.y = round_half_integer_up(index.y);
  if(round_up.x < 0 || round_up.y < 0)
  {
    return false;
  }

  float2 bound;
  bound.x = (float)size.x - 0.5f;
  bound.y = (float)size.y - 0.5f;
  if( !(index.x <= bound.x) || !(index.y <= bound.y) )
  {
    return false;
  }

  return true;
}

//------------------------------------------------------------------------------
// OpenCL 3D implementation of
// itkImageRegion::IsInside(const ContinuousIndex< TCoordRepType,
// VImageDimension > &index)
bool is_continuous_index_inside_3d(const float3 index, const uint3 size)
{
  int3 round_up;

  round_up.x = round_half_integer_up(index.x);
  round_up.y = round_half_integer_up(index.y);
  round_up.z = round_half_integer_up(index.z);

  if(round_up.x < 0)
  {
    return false;
  }
  if(round_up.y < 0)
  {
    return false;
  }
  if(round_up.z < 0)
  {
    return false;
  }

  float3 bound;
  bound.x = (float)size.x - 0.5f;
  bound.y = (float)size.y - 0.5f;
  bound.z = (float)size.z - 0.5f;

  if( !(index.x <= bound.x) )
  {
    return false;
  }
  if( !(index.y <= bound.y) )
  {
    return false;
  }
  if( !(index.z <= bound.z) )
  {
    return false;
  }

  return true;
}

//------------------------------------------------------------------------------
// OpenCL 1D implementation of
// itkImageBase::TransformIndexToPhysicalPoint()
float transform_index_to_physical_point_1d(const uint index,
                                           __global const GPUImageBase1D *image)
{
  float point = image->origin;

  point = image->index_to_physical_point * index;
  return point;
}

//------------------------------------------------------------------------------
// OpenCL 2D implementation of
// itkImageBase::TransformIndexToPhysicalPoint()
float2 transform_index_to_physical_point_2d(const uint2 index,
                                            __global const GPUImageBase2D *image)
{
  float2 point = image->origin;

  point.x += image->index_to_physical_point.s0 * index.x;
  point.x += image->index_to_physical_point.s1 * index.y;

  point.y += image->index_to_physical_point.s2 * index.x;
  point.y += image->index_to_physical_point.s3 * index.y;

  return point;
}

//------------------------------------------------------------------------------
// OpenCL 3D implementation of
// itkImageBase::TransformIndexToPhysicalPoint()
float3 transform_index_to_physical_point_3d(const uint3 index,
                                            __constant GPUImageBase3D *image)
{
  float3 point = image->origin;

  point.x += image->index_to_physical_point.s0 * index.x;
  point.x += image->index_to_physical_point.s1 * index.y;
  point.x += image->index_to_physical_point.s2 * index.z;

  point.y += image->index_to_physical_point.s3 * index.x;
  point.y += image->index_to_physical_point.s4 * index.y;
  point.y += image->index_to_physical_point.s5 * index.z;

  point.z += image->index_to_physical_point.s6 * index.x;
  point.z += image->index_to_physical_point.s7 * index.y;
  point.z += image->index_to_physical_point.s8 * index.z;

  return point;
}

//------------------------------------------------------------------------------
// OpenCL 1D implementation of
// itkImageBase::TransformPhysicalPointToContinuousIndex()
bool transform_physical_point_to_continuous_index_1d(const float point, float *index,
                                                     __global const GPUImageBase1D *image)
{
  float cvector = point - image->origin;
  float cvector1;

  cvector1 = image->physical_point_to_index * cvector;
  *index = cvector1;
  return is_continuous_index_inside_1d(cvector1, image->size);
}

//------------------------------------------------------------------------------
// OpenCL 2D implementation of
// itkImageBase::TransformPhysicalPointToContinuousIndex()
bool transform_physical_point_to_continuous_index_2d(const float2 point, float2 *index,
                                                     __global const GPUImageBase2D *image)
{
  float2 cvector = point - image->origin;
  float2 cvector1;

  cvector1.x =  image->physical_point_to_index.s0 * cvector.x;
  cvector1.x += image->physical_point_to_index.s1 * cvector.y;

  cvector1.y =  image->physical_point_to_index.s2 * cvector.x;
  cvector1.y += image->physical_point_to_index.s3 * cvector.y;

  *index = cvector1;
  return is_continuous_index_inside_2d(cvector1, image->size);
}

//------------------------------------------------------------------------------
// OpenCL 3D implementation of
// itkImageBase::TransformPhysicalPointToContinuousIndex()
bool transform_physical_point_to_continuous_index_3d(const float3 point, float3 *index,
                                                     __constant GPUImageBase3D *image)
{
  float3 cvector = point - image->origin;
  float3 cvector1;

  cvector1.x =  image->physical_point_to_index.s0 * cvector.x;
  cvector1.x += image->physical_point_to_index.s1 * cvector.y;
  cvector1.x += image->physical_point_to_index.s2 * cvector.z;

  cvector1.y =  image->physical_point_to_index.s3 * cvector.x;
  cvector1.y += image->physical_point_to_index.s4 * cvector.y;
  cvector1.y += image->physical_point_to_index.s5 * cvector.z;

  cvector1.z =  image->physical_point_to_index.s6 * cvector.x;
  cvector1.z += image->physical_point_to_index.s7 * cvector.y;
  cvector1.z += image->physical_point_to_index.s8 * cvector.z;

  *index = cvector1;
  return is_continuous_index_inside_3d(cvector1, image->size);
}

//------------------------------------------------------------------------------
// OpenCL 1D implementation (long index, float return value) version of
// itkImage::GetPixel()
float get_pixel_1d(const long index,
                   __global const INPIXELTYPE *in,
                   __global const GPUImageBase1D *image)
{
  float value = (float)(in[index]);
  return value;
}

//------------------------------------------------------------------------------
// OpenCL 2D implementation (long index, float return value) version of
// itkImage::GetPixel()
float get_pixel_2d(const long2 index,
                   __global const INPIXELTYPE *in,
                   __global const GPUImageBase2D *image)
{
  uint  gidx = mad24(image->size.x, index.y, index.x);
  float value = (float)(in[gidx]);
  return value;
}

//------------------------------------------------------------------------------
// OpenCL 3D implementation (long index, float return value) version of
// itkImage::GetPixel()
float get_pixel_3d(const long3 index,
                   __global const INPIXELTYPE *in,
                   __constant GPUImageBase3D *image)
{
  uint  gidx = mad24(image->size.x, mad24(index.z, image->size.y, index.y), index.x);
  float value = (float)(in[gidx]);
  return value;
}

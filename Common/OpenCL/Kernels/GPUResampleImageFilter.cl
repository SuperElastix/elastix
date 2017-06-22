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
// OpenCL implementation of itk::ResampleImageFilter

//------------------------------------------------------------------------------
// Typedef for FilterParameters struct
typedef struct {
  float2 min_max;
  float2 min_max_output;
  float  default_value;
  float  dummy_for_alignment;
} FilterParameters;

//------------------------------------------------------------------------------
bool is_valid_1d( const uint index, const uint size )
{
  if( index >= size ) { return false; }
  return true;
}

//------------------------------------------------------------------------------
bool is_valid_2d( const uint2 index, const uint2 size )
{
  if( index.x >= size.x ) { return false; }
  if( index.y >= size.y ) { return false; }
  return true;
}

//------------------------------------------------------------------------------
bool is_valid_3d( const uint3 index, const uint3 size )
{
  /* NOTE: More than three-level nested conditional statements (e.g.,
  if A && B && C..) invalidates command queue during kernel
  execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
  GT). Therefore, we flattened conditional statements. */
  if( index.x >= size.x ) return false;
  if( index.y >= size.y ) return false;
  if( index.z >= size.z ) return false;
  return true;
}

//------------------------------------------------------------------------------
uint get_global_id_1d( void )
{
  uint global_id = get_global_id( 0 );
  return global_id;
}

uint get_current_image_index_1d( uint global_id )
{
  uint global_offset = get_global_offset( 0 );
  uint index = global_id - global_offset;
  return index;
}

//------------------------------------------------------------------------------
uint2 get_global_id_2d( void )
{
  uint2 global_id = (uint2)( get_global_id( 0 ), get_global_id( 1 ) );
  return global_id;
}

uint2 get_current_image_index_2d( uint2 global_id )
{
  uint2 global_offset = (uint2)( get_global_offset( 0 ), get_global_offset( 1 ) );
  uint2 index = global_id - global_offset;
  return index;
}

//------------------------------------------------------------------------------
uint3 get_global_id_3d( void )
{
  uint3 global_id = (uint3)( get_global_id( 0 ),
    get_global_id( 1 ), get_global_id( 2 ) );
  return global_id;
}

uint3 get_current_image_index_3d( uint3 global_id )
{
  uint3 global_offset = (uint3)( get_global_offset( 0 ),
    get_global_offset( 1 ), get_global_offset( 2 ) );
  uint3 index = global_id - global_offset;
  return index;
}

//------------------------------------------------------------------------------
// cast from interpolator output to pixel type
OUTPIXELTYPE cast_pixel_with_bounds_checking(
  const OUTPIXELTYPE value,
  const float2 min_max,
  const float2 min_max_output )
{
  OUTPIXELTYPE output_value;

  // check for value min/max
  if( value < min_max_output.x ){      output_value = min_max.x; }
  else if( value > min_max_output.y ){ output_value = min_max.y; }
  else{                                output_value = value; }

  return output_value;
}

//------------------------------------------------------------------------------
#if defined( DIM_1 ) && defined( RESAMPLE_PRE )
__kernel void ResampleImageFilterPre(
  /* Transformation field buffer */
  __global float *transformation_field,
  /* Transformation field size */
  uint transformation_field_size,
  /* Output image information */
  const float index_to_physical_point,
  const float origin,
  const uint size )
{
  // Get current image index
  uint global_id = get_global_id_1d();
  uint index = get_current_image_index_1d( global_id );

  if( is_valid_1d( index, transformation_field_size ) && is_valid_1d( global_id, size ) )
  {
    float point = transform_index_to_physical_point_1d_(
      global_id, index_to_physical_point, origin );

    // calculate gidx within buffer
    uint gidx = index;
    transformation_field[gidx] = point;
  }
}
#endif

//------------------------------------------------------------------------------
// This kernel executed for itk::GPUResampleImageFilter with identity transform
// \sa IdentityTransform
#if defined( DIM_1 ) && defined( RESAMPLE_LOOP ) && defined( IDENTITY_TRANSFORM )
__kernel void ResampleImageFilterLoop_IdentityTransform(
  /* Transformation field buffer */
  __global float *transformation_field,
  /* Transformation field size */
  uint transformation_field_size,
  /* Output image size */
  uint output_image_size )
{
  // Get current image index
  uint global_id = get_global_id_1d();
  uint index = get_current_image_index_1d( global_id );

  if( is_valid_1d( index, transformation_field_size ) && is_valid_1d( global_id, output_image_size ) )
  {
    // get point
    const uint tidx = index;
    float output_point = transformation_field[tidx];

    // Perform coordinate transformation
    float input_point = identity_transform_point_1d( output_point );

    // Store the result in the deformation field
    transformation_field[tidx] = input_point;
  }
}
#endif

//------------------------------------------------------------------------------
// This kernel executed for itk::GPUResampleImageFilter with matrix offset transform
// \sa MatrixOffsetTransformBase
#if defined( DIM_1 ) && defined( RESAMPLE_LOOP ) && defined( MATRIX_OFFSET_TRANSFORM )
__kernel void ResampleImageFilterLoop_MatrixOffsetTransform(
  /* Transformation field buffer */
  __global float *transformation_field,
  /* Transformation field size */
  uint transformation_field_size,
  /* Output image size */
  uint output_image_size,
  /* transform base parameters */
  __constant GPUMatrixOffsetTransformBase1D *transform_base )
{
  // Get current image index
  uint global_id = get_global_id_1d();
  uint index = get_current_image_index_1d( global_id );

  if( is_valid_1d( index, transformation_field_size ) && is_valid_1d( global_id, output_image_size ) )
  {
    // get point
    const uint tidx = index;
    float output_point = transformation_field[tidx];

    // Perform coordinate transformation
    float input_point = matrix_offset_transform_point_1d(
      output_point, transform_base->matrix, transform_base->offset );

    // Store the result in the deformation field
    transformation_field[tidx] = input_point;
  }
}
#endif

//------------------------------------------------------------------------------
// This kernel executed for itk::GPUResampleImageFilter with matrix offset transform
// \sa TranslationTransform
#if defined( DIM_1 ) && defined( RESAMPLE_LOOP ) && defined( TRANSLATION_TRANSFORM )
__kernel void ResampleImageFilterLoop_TranslationTransform(
  /* Transformation field buffer */
  __global float *transformation_field,
  /* Transformation field size */
  uint transformation_field_size,
  /* Output image size */
  uint output_image_size,
  /* transform base parameters */
  __constant GPUTranslationTransformBase1D *transform_base )
{
  // Get current image index
  uint global_id = get_global_id_1d();
  uint index = get_current_image_index_1d( global_id );

  if( is_valid_1d( index, transformation_field_size ) && is_valid_1d( global_id, output_image_size ) )
  {
    // get point
    const uint tidx = index;
    float output_point = transformation_field[tidx];

    // Perform coordinate transformation
    float input_point = translation_transform_point_1d(
      output_point, transform_base->offset );

    // Store the result in the deformation field
    transformation_field[tidx] = input_point;
  }
}
#endif

//------------------------------------------------------------------------------
// This kernel executed for itk::GPUResampleImageFilter with BSpline transform
// \sa BSplineBaseTransform
#if defined( DIM_1 ) && defined( RESAMPLE_LOOP ) && defined( BSPLINE_TRANSFORM )
__kernel void ResampleImageFilterLoop_BSplineTransform(
  /* Transformation field buffer */
  __global float *transformation_field,
  /* Transformation field size */
  uint transformation_field_size,
  /* Output image size */
  uint output_image_size,
  /* B-spline transform spline order */
  uint spline_order,
  /* B-spline transform coefficients image meta information. */
  __constant GPUImageBase1D *coefficients_image,
  /* B-spline transform coefficients images. */
  __global const float *transform_coefficients )
{
  // Get current image index
  uint global_id = get_global_id_1d();
  uint index = get_current_image_index_1d( global_id );

  if( is_valid_1d( index, transformation_field_size ) && is_valid_1d( global_id, output_image_size ) )
  {
    // get point
    const uint tidx = index;
    float output_point = transformation_field[tidx];

    // Perform coordinate transformation
    float input_point = bspline_transform_point_1d( output_point, spline_order,
      coefficients_image, transform_coefficients );

    // Store the result in the deformation field
    transformation_field[tidx] = input_point;
  }
}
#endif

//------------------------------------------------------------------------------
#if defined( DIM_1 ) && defined( RESAMPLE_POST ) && !defined( BSPLINE_INTERPOLATOR )
__kernel void ResampleImageFilterPost(
  /* Transformation field buffer */
  __global const float *transformation_field,
  /* Transformation field size */
  uint transformation_field_size,
  /* Input image buffer */
  __global const INPIXELTYPE *in,
  /* Input image meta information. */
  __constant GPUImageBase1D * input_image,
  /* Output image buffer */
  __global OUTPIXELTYPE *out,
  /* Output image size */
  uint output_image_size,
  /* Filter parameters */
  __constant FilterParameters *parameters,
  /* Image function parameters. NOTE: Should be defined as __constant, but fails on GeForce GTX 780. */
  __global GPUImageFunction1D *image_function )
{
  // Get current image index
  uint global_id = get_global_id_1d();
  uint index = get_current_image_index_1d( global_id );

  if( is_valid_1d( index, transformation_field_size ) && is_valid_1d( global_id, output_image_size ) )
  {
    // Get the transformed point
    const float transformed_point = transformation_field[index];
    // Convert to continuous index
    float continuous_index;
    transform_physical_point_to_continuous_index_1d( transformed_point, &continuous_index, input_image );

    // evaluate input at right position and copy to the output
    if( interpolator_is_inside_buffer_1d( continuous_index,
      image_function->start_continuous_index, image_function->end_continuous_index ) )
    {
      OUTPIXELTYPE value = evaluate_at_continuous_index_1d(
        continuous_index, in, input_image->size,
        image_function->start_index, image_function->end_index );
      out[global_id] = cast_pixel_with_bounds_checking(
        value, parameters->min_max, parameters->min_max_output );
    }
    else
    {
      out[global_id] = parameters->default_value;
    }
  }
}
#endif

//------------------------------------------------------------------------------
#if defined( DIM_1 ) && defined( RESAMPLE_POST ) && defined( BSPLINE_INTERPOLATOR )
__kernel void ResampleImageFilterPost_BSplineInterpolator(
  /* Transformation field buffer */
  __global const float *transformation_field,
  /* Transformation field size */
  uint transformation_field_size,
  /* B-spline interpolator coefficients image. */
  __global const float * interpolator_coefficients,
  /* B-spline interpolator coefficients image meta information. */
  __constant GPUImageBase1D * interpolator_coefficients_image,
  /* B-spline interpolator spline order */
  uint spline_order,
  /* Output image buffer */
  __global OUTPIXELTYPE *out,
  /* Output image size */
  uint output_image_size,
  /* Filter parameters */
  __constant FilterParameters *parameters,
  /* Image function parameters. NOTE: Should be defined as __constant, but fails on GeForce GTX 780. */
  __global GPUImageFunction1D *image_function )
{
  // Get current image index
  uint global_id = get_global_id_1d();
  uint index = get_current_image_index_1d( global_id );

  if( is_valid_1d( index, transformation_field_size ) && is_valid_1d( global_id, output_image_size ) )
  {
    // Get the transformed point
    const float transformed_point = transformation_field[index];
    // Convert to continuous index
    float continuous_index;
    transform_physical_point_to_continuous_index_1d(
      transformed_point, &continuous_index, interpolator_coefficients_image );

    // evaluate input at right position and copy to the output
    if( interpolator_is_inside_buffer_1d( continuous_index,
      image_function->start_continuous_index, image_function->end_continuous_index ) )
    {
      OUTPIXELTYPE value = bspline_evaluate_at_continuous_index_1d(
        continuous_index, spline_order,
        image_function->start_index,
        image_function->end_index,
        interpolator_coefficients,
        interpolator_coefficients_image->size );

      out[global_id] = cast_pixel_with_bounds_checking(
        value, parameters->min_max, parameters->min_max_output );
    }
    else
    {
      out[global_id] = parameters->default_value;
    }
  }
}
#endif

//------------------------------------------------------------------------------
#if defined( DIM_2 ) && defined( RESAMPLE_PRE )
__kernel void ResampleImageFilterPre(
  /* Transformation field buffer */
  __global float2 *transformation_field,
  /* Transformation field size */
  uint2 transformation_field_size,
  /* Output image information */
  const float4 index_to_physical_point,
  const float2 origin,
  const uint2 size )
{
  // Get current image index
  uint2 global_id = get_global_id_2d();
  uint2 index = get_current_image_index_2d( global_id );

  if( is_valid_2d( index, transformation_field_size ) && is_valid_2d( global_id, size ) )
  {
    float2 point = transform_index_to_physical_point_2d_(
      global_id, index_to_physical_point, origin );

    // calculate gidx within buffer
    uint gidx = mad24( transformation_field_size.x, index.y, index.x );
    transformation_field[gidx] = point;
  }
}
#endif

//------------------------------------------------------------------------------
// This kernel executed for itk::GPUResampleImageFilter with identity transform
// \sa IdentityTransform
#if defined( DIM_2 ) && defined( RESAMPLE_LOOP ) && defined( IDENTITY_TRANSFORM )
__kernel void ResampleImageFilterLoop_IdentityTransform(
  /* Transformation field buffer */
  __global float2 *transformation_field,
  /* Transformation field size */
  uint2 transformation_field_size,
  /* Output image size */
  uint2 output_image_size )
{
  // Get current image index
  uint2 global_id = get_global_id_2d();
  uint2 index = get_current_image_index_2d( global_id );

  if( is_valid_2d( index, transformation_field_size ) && is_valid_2d( global_id, output_image_size ) )
  {
    // get point
    const uint tidx = mad24( transformation_field_size.x, index.y, index.x );
    float2 output_point = transformation_field[tidx];

    // Perform coordinate transformation
    float2 input_point = identity_transform_point_2d( output_point );

    // Store the result in the deformation field
    transformation_field[tidx] = input_point;
  }
}
#endif

//------------------------------------------------------------------------------
// This kernel executed for itk::GPUResampleImageFilter with matrix offset transform
// \sa MatrixOffsetTransformBase
#if defined( DIM_2 ) && defined( RESAMPLE_LOOP ) && defined( MATRIX_OFFSET_TRANSFORM )
__kernel void ResampleImageFilterLoop_MatrixOffsetTransform(
  /* Transformation field buffer */
  __global float2 *transformation_field,
  /* Transformation field size */
  uint2 transformation_field_size,
  /* Output image size */
  uint2 output_image_size,
  /* transform base parameters */
  __constant GPUMatrixOffsetTransformBase2D *transform_base )
{
  // Get current image index
  uint2 global_id = get_global_id_2d();
  uint2 index = get_current_image_index_2d( global_id );

  if( is_valid_2d( index, transformation_field_size ) && is_valid_2d( global_id, output_image_size ) )
  {
    // get point
    const uint tidx = mad24( transformation_field_size.x, index.y, index.x );
    float2 output_point = transformation_field[tidx];

    // Perform coordinate transformation
    float2 input_point = matrix_offset_transform_point_2d(
      output_point, transform_base->matrix, transform_base->offset );

    // Store the result in the deformation field
    transformation_field[tidx] = input_point;
  }
}
#endif

//------------------------------------------------------------------------------
// This kernel executed for itk::GPUResampleImageFilter with matrix offset transform
// \sa TranslationTransform
#if defined( DIM_2 ) && defined( RESAMPLE_LOOP ) && defined( TRANSLATION_TRANSFORM )
__kernel void ResampleImageFilterLoop_TranslationTransform(
  /* Transformation field buffer */
  __global float2 *transformation_field,
  /* Transformation field size */
  uint2 transformation_field_size,
  /* Output image size */
  uint2 output_image_size,
  /* transform base parameters */
  __constant GPUTranslationTransformBase2D *transform_base )
{
  // Get current image index
  uint2 global_id = get_global_id_2d();
  uint2 index = get_current_image_index_2d( global_id );

  if( is_valid_2d( index, transformation_field_size ) && is_valid_2d( global_id, output_image_size ) )
  {
    // get point
    const uint tidx = mad24( transformation_field_size.x, index.y, index.x );
    float2 output_point = transformation_field[tidx];

    // Perform coordinate transformation
    float2 input_point = translation_transform_point_2d(
      output_point, transform_base->offset );

    // Store the result in the deformation field
    transformation_field[tidx] = input_point;
  }
}
#endif

//------------------------------------------------------------------------------
// This kernel executed for itk::GPUResampleImageFilter with BSpline transform
// \sa BSplineBaseTransform
#if defined( DIM_2 ) && defined( RESAMPLE_LOOP ) && defined( BSPLINE_TRANSFORM )
__kernel void ResampleImageFilterLoop_BSplineTransform(
  /* Transformation field buffer */
  __global float2 *transformation_field,
  /* Transformation field size */
  uint2 transformation_field_size,
  /* Output image size */
  uint2 output_image_size,
  /* B-spline transform spline order */
  uint spline_order,
  /* B-spline transform coefficients image meta information. */
  __constant GPUImageBase2D *coefficients_image,
  /* B-spline transform coefficients images. */
  __global const float *transform_coefficients0,
  __global const float *transform_coefficients1 )
{
  // Get current image index
  uint2 global_id = get_global_id_2d();
  uint2 index = get_current_image_index_2d( global_id );

  if( is_valid_2d( index, transformation_field_size ) && is_valid_2d( global_id, output_image_size ) )
  {
    // get point
    const uint tidx = mad24( transformation_field_size.x, index.y, index.x );
    float2 output_point = transformation_field[tidx];

    // Perform coordinate transformation
    float2 input_point = bspline_transform_point_2d( output_point, spline_order,
      coefficients_image,
      transform_coefficients0, transform_coefficients1 );

    // Store the result in the deformation field
    transformation_field[tidx] = input_point;
  }
}
#endif

//------------------------------------------------------------------------------
#if defined( DIM_2 ) && defined( RESAMPLE_POST ) && !defined( BSPLINE_INTERPOLATOR )
__kernel void ResampleImageFilterPost(
  /* Transformation field buffer */
  __global const float2 *transformation_field,
  /* Transformation field size */
  uint2 transformation_field_size,
  /* Input image buffer */
  __global const INPIXELTYPE *in,
  /* Input image meta information. */
  __constant GPUImageBase2D * input_image,
  /* Output image buffer */
  __global OUTPIXELTYPE *out,
  /* Output image size */
  uint2 output_image_size,
  /* Filter parameters */
  __constant FilterParameters *parameters,
  /* Image function parameters. NOTE: Should be defined as __constant, but fails on GeForce GTX 780. */
  __global GPUImageFunction2D *image_function )
{
  // Get current image index
  uint2 global_id = get_global_id_2d();
  uint2 index = get_current_image_index_2d( global_id );

  if( is_valid_2d( index, transformation_field_size ) && is_valid_2d( global_id, output_image_size ) )
  {
    const uint tidx = mad24( transformation_field_size.x, index.y, index.x );
    const uint gidx = mad24( output_image_size.x, global_id.y, global_id.x );

    // Get the transformed point
    const float2 transformed_point = transformation_field[tidx];
    // Convert to continuous index
    float2 continuous_index;
    transform_physical_point_to_continuous_index_2d(
      transformed_point, &continuous_index, input_image );

    // evaluate input at right position and copy to the output
    if( interpolator_is_inside_buffer_2d( continuous_index,
      image_function->start_continuous_index, image_function->end_continuous_index ) )
    {
      OUTPIXELTYPE value = evaluate_at_continuous_index_2d(
        continuous_index, in, input_image->size,
        image_function->start_index, image_function->end_index );
      out[gidx] = cast_pixel_with_bounds_checking(
        value, parameters->min_max, parameters->min_max_output );
    }
    else
    {
      out[gidx] = parameters->default_value;
    }
  }
}
#endif

//------------------------------------------------------------------------------
#if defined( DIM_2 ) && defined( RESAMPLE_POST ) && defined( BSPLINE_INTERPOLATOR )
__kernel void ResampleImageFilterPost_BSplineInterpolator(
  /* Transformation field buffer */
  __global const float2 *transformation_field,
  /* Transformation field size */
  uint2 transformation_field_size,
  /* B-spline interpolator coefficients image. */
  __global const float * interpolator_coefficients,
  /* B-spline interpolator coefficients image meta information. */
  __constant GPUImageBase2D * interpolator_coefficients_image,
  /* B-spline interpolator spline order */
  uint spline_order,
  /* Output image buffer */
  __global OUTPIXELTYPE *out,
  /* Output image size */
  uint2 output_image_size,
  /* Filter parameters */
  __constant FilterParameters *parameters,
  /* Image function parameters. NOTE: Should be defined as __constant, but fails on GeForce GTX 780. */
  __global GPUImageFunction2D *image_function )
{
  // Get current image index
  uint2 global_id = get_global_id_2d();
  uint2 index = get_current_image_index_2d( global_id );

  if( is_valid_2d( index, transformation_field_size ) && is_valid_2d( global_id, output_image_size ) )
  {
    const uint tidx = mad24( transformation_field_size.x, index.y, index.x );
    const uint gidx = mad24( output_image_size.x, global_id.y, global_id.x );

    // Get the transformed point
    const float2 transformed_point = transformation_field[tidx];
    // Convert to continuous index
    float2 continuous_index;
    transform_physical_point_to_continuous_index_2d(
      transformed_point, &continuous_index, interpolator_coefficients_image );

    // evaluate input at right position and copy to the output
    if( interpolator_is_inside_buffer_2d( continuous_index,
      image_function->start_continuous_index, image_function->end_continuous_index ) )
    {
      OUTPIXELTYPE value = bspline_evaluate_at_continuous_index_2d(
        continuous_index, spline_order,
        image_function->start_index,
        image_function->end_index,
        interpolator_coefficients,
        interpolator_coefficients_image->size );

      out[gidx] = cast_pixel_with_bounds_checking(
        value, parameters->min_max, parameters->min_max_output );
    }
    else
    {
      out[gidx] = parameters->default_value;
    }
  }
}
#endif

//------------------------------------------------------------------------------
#if defined( DIM_3 ) && defined( RESAMPLE_PRE )
__kernel void ResampleImageFilterPre(
  /* Transformation field buffer */
  __global float3 *transformation_field,
  /* Transformation field size */
  uint3 transformation_field_size,
  /* Output image information */
  const float16 index_to_physical_point, // OpenCL does not have float9
  const float3 origin,
  const uint3 size )
{
  // Get current image index
  uint3 global_id = get_global_id_3d();
  uint3 index = get_current_image_index_3d( global_id );

  if( is_valid_3d( index, transformation_field_size ) && is_valid_3d( global_id, size ) )
  {
    float3 point = transform_index_to_physical_point_3d_(
      global_id, index_to_physical_point, origin );

    // calculate gidx within buffer
    uint gidx = mad24( transformation_field_size.x, mad24( index.z, transformation_field_size.y, index.y ), index.x );
    transformation_field[gidx] = point;
  }
}
#endif

//------------------------------------------------------------------------------
// This kernel executed for itk::GPUResampleImageFilter with identity transform
// \sa IdentityTransform
#if defined( DIM_3 ) && defined( RESAMPLE_LOOP ) && defined( IDENTITY_TRANSFORM )
__kernel void ResampleImageFilterLoop_IdentityTransform(
  /* Transformation field buffer */
  __global float3 *transformation_field,
  /* Transformation field size */
  uint3 transformation_field_size,
  /* Output image size */
  uint3 output_image_size )
{
  // Get current image index
  uint3 global_id = get_global_id_3d();
  uint3 index = get_current_image_index_3d( global_id );

  if( is_valid_3d( index, transformation_field_size ) && is_valid_3d( global_id, output_image_size ) )
  {
    // get point
    const uint tidx = mad24( transformation_field_size.x,
      mad24( index.z, transformation_field_size.y, index.y ), index.x );
    float3 output_point = transformation_field[tidx];

    // Perform coordinate transformation
    float3 input_point = identity_transform_point_3d( output_point );

    // Store the result in the deformation field
    transformation_field[tidx] = input_point;
  }
}
#endif

//------------------------------------------------------------------------------
// This kernel executed for itk::GPUResampleImageFilter with matrix offset transform
// \sa MatrixOffsetTransformBase
#if defined( DIM_3 ) && defined( RESAMPLE_LOOP ) && defined( MATRIX_OFFSET_TRANSFORM )
__kernel void ResampleImageFilterLoop_MatrixOffsetTransform(
  /* Transformation field buffer */
  __global float3 *transformation_field,
  /* Transformation field size */
  uint3 transformation_field_size,
  /* Output image size */
  uint3 output_image_size,
  /* transform base parameters */
  __constant GPUMatrixOffsetTransformBase3D *transform_base )
{
  // Get current image index
  uint3 global_id = get_global_id_3d();
  uint3 index = get_current_image_index_3d( global_id );

  if( is_valid_3d( index, transformation_field_size ) && is_valid_3d( global_id, output_image_size ) )
  {
    // get point
    const uint tidx = mad24( transformation_field_size.x,
      mad24( index.z, transformation_field_size.y, index.y ), index.x );
    float3 output_point = transformation_field[tidx];

    // Perform coordinate transformation
    float3 input_point = matrix_offset_transform_point_3d(
      output_point, transform_base->matrix, transform_base->offset );

    // Store the result in the deformation field
    transformation_field[tidx] = input_point;
  }
}
#endif

//------------------------------------------------------------------------------
// This kernel executed for itk::GPUResampleImageFilter with matrix offset transform
// \sa TranslationTransform
#if defined( DIM_3 ) && defined( RESAMPLE_LOOP ) && defined( TRANSLATION_TRANSFORM )
__kernel void ResampleImageFilterLoop_TranslationTransform(
  /* Transformation field buffer */
  __global float3 *transformation_field,
  /* Transformation field size */
  uint3 transformation_field_size,
  /* Output image size */
  uint3 output_image_size,
  /* transform base parameters */
  __constant GPUTranslationTransformBase3D *transform_base )
{
  // Get current image index
  uint3 global_id = get_global_id_3d();
  uint3 index = get_current_image_index_3d( global_id );

  if( is_valid_3d( index, transformation_field_size ) && is_valid_3d( global_id, output_image_size ) )
  {
    // get point
    const uint tidx = mad24( transformation_field_size.x,
      mad24( index.z, transformation_field_size.y, index.y ), index.x );
    float3 output_point = transformation_field[tidx];

    // Perform coordinate transformation
    float3 input_point = translation_transform_point_3d(
      output_point, transform_base->offset );

    // Store the result in the deformation field
    transformation_field[tidx] = input_point;
  }
}
#endif

//------------------------------------------------------------------------------
// This kernel executed for itk::GPUResampleImageFilter with BSpline transform
// \sa BSplineBaseTransform
#if defined( DIM_3 ) && defined( RESAMPLE_LOOP ) && defined( BSPLINE_TRANSFORM )
__kernel void ResampleImageFilterLoop_BSplineTransform(
  /* Transformation field buffer */
  __global float3 *transformation_field,
  /* Transformation field size */
  uint3 transformation_field_size,
  /* Output image size */
  uint3 output_image_size,
  /* B-spline transform spline order */
  uint spline_order,
  /* B-spline transform coefficients image meta information. */
  __constant GPUImageBase3D *coefficients_image, // only PhysicalPointToIndex, Origin, Size needed
  /* B-spline transform coefficients images. */
  __global const float *transform_coefficients0,
  __global const float *transform_coefficients1,
  __global const float *transform_coefficients2 )
{
  // Get current image index
  uint3 global_id = get_global_id_3d();
  uint3 index = get_current_image_index_3d( global_id );

  if( is_valid_3d( index, transformation_field_size ) && is_valid_3d( global_id, output_image_size ) )
  {
    // get point
    const uint tidx = mad24( transformation_field_size.x,
      mad24( index.z, transformation_field_size.y, index.y ), index.x );
    float3 output_point = transformation_field[tidx];

    // Perform coordinate transformation
    float3 input_point = bspline_transform_point_3d( output_point, spline_order,
      coefficients_image,
      transform_coefficients0, transform_coefficients1, transform_coefficients2 );

    // Store the result in the deformation field
    transformation_field[tidx] = input_point;
  }
}
#endif

//------------------------------------------------------------------------------
#if defined( DIM_3 ) && defined( RESAMPLE_POST ) && !defined( BSPLINE_INTERPOLATOR )
__kernel void ResampleImageFilterPost(
  /* Transformation field buffer */
  __global const float3 *transformation_field,
  /* Transformation field size */
  uint3 transformation_field_size,
  /* Input image buffer */
  __global const INPIXELTYPE *in,
  /* Input image meta information. */
  __constant GPUImageBase3D * input_image,
  /* Output image buffer */
  __global OUTPIXELTYPE *out,
  /* Output image size */
  uint3 output_image_size,
  /* Filter parameters */
  __constant FilterParameters *parameters,
  /* Image function parameters. NOTE: Should be defined as __constant, but fails on GeForce GTX 780. */
  __global GPUImageFunction3D *image_function )
{
  // Get current image index
  uint3 global_id = get_global_id_3d();
  uint3 index = get_current_image_index_3d( global_id );

  if( is_valid_3d( index, transformation_field_size ) && is_valid_3d( global_id, output_image_size ) )
  {
    const uint tidx = mad24( transformation_field_size.x,
      mad24( index.z, transformation_field_size.y, index.y ), index.x );
    const uint gidx = mad24( output_image_size.x,
      mad24( global_id.z, output_image_size.y, global_id.y ), global_id.x );

    // Get the transformed point
    const float3 transformed_point = transformation_field[tidx];
    // Convert to continuous index
    float3 continuous_index
      = transform_physical_point_to_continuous_index_3d( transformed_point,
      input_image->physical_point_to_index, input_image->origin );

    // evaluate input at right position and copy to the output
    if( interpolator_is_inside_buffer_3d( continuous_index,
      image_function->start_continuous_index, image_function->end_continuous_index ) )
    {
      OUTPIXELTYPE value = evaluate_at_continuous_index_3d(
        continuous_index, in, input_image->size,
        image_function->start_index, image_function->end_index );

      out[gidx] = cast_pixel_with_bounds_checking(
        value, parameters->min_max, parameters->min_max_output );
    }
    else
    {
      out[gidx] = parameters->default_value;
    }
  }
}
#endif

//------------------------------------------------------------------------------
#if defined( DIM_3 ) && defined( RESAMPLE_POST ) && defined( BSPLINE_INTERPOLATOR )
__kernel void ResampleImageFilterPost_BSplineInterpolator(
  /* Transformation field buffer */
  __global const float3 *transformation_field,
  /* Transformation field size */
  uint3 transformation_field_size,
  /* B-spline interpolator coefficients image. */
  __global const float * interpolator_coefficients,
  /* B-spline interpolator coefficients image meta information. */
  __constant GPUImageBase3D * interpolator_coefficients_image,
  /* B-spline interpolator spline order */
  uint spline_order,
  /* Output image buffer */
  __global OUTPIXELTYPE *out,
  /* Output image size */
  uint3 output_image_size,
  /* Filter parameters */
  __constant FilterParameters *parameters,
  /* Image function parameters. NOTE: Should be defined as __constant, but fails on GeForce GTX 780. */
  __global GPUImageFunction3D *image_function )
{
  // Get current image index
  uint3 global_id = get_global_id_3d();
  uint3 index = get_current_image_index_3d( global_id );

  if( is_valid_3d( index, transformation_field_size ) && is_valid_3d( global_id, output_image_size ) )
  {
    const uint tidx = mad24( transformation_field_size.x,
      mad24( index.z, transformation_field_size.y, index.y ), index.x );
    const uint gidx = mad24( output_image_size.x,
      mad24( global_id.z, output_image_size.y, global_id.y ), global_id.x );

    // Get the transformed point
    const float3 transformed_point = transformation_field[tidx];
    // Convert to continuous index
    float3 continuous_index
      = transform_physical_point_to_continuous_index_3d( transformed_point,
      interpolator_coefficients_image->physical_point_to_index,
      interpolator_coefficients_image->origin );

    // evaluate input at right position and copy to the output
    if( interpolator_is_inside_buffer_3d( continuous_index,
      image_function->start_continuous_index, image_function->end_continuous_index ) )
    {
      OUTPIXELTYPE value = bspline_evaluate_at_continuous_index_3d(
        continuous_index, spline_order,
        image_function->start_index,
        image_function->end_index,
        interpolator_coefficients,
        interpolator_coefficients_image->size );

      out[ gidx ] = cast_pixel_with_bounds_checking(
        value, parameters->min_max, parameters->min_max_output );
    }
    else
    {
      out[ gidx ] = parameters->default_value;
    }
  }
}
#endif

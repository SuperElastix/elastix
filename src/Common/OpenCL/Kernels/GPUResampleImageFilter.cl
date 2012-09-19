/*======================================================================

This file is part of the elastix software.

Copyright (c) University Medical Center Utrecht. All rights reserved.
See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the above copyright notices for more information.

======================================================================*/

//------------------------------------------------------------------------------
// Typedef for FilterParameters struct
typedef struct {
  int transform_linear;
  int interpolator_is_bspline;
  int transform_is_bspline;
  float default_value;
  float2 min_max;
  float2 min_max_output;
  float3 delta;
} FilterParameters;

//------------------------------------------------------------------------------
// Apple OpenCL 1.0 support function
bool is_valid(const uint3 index, const uint3 size)
{
  /* NOTE: More than three-level nested conditional statements (e.g.,
  if A && B && C..) invalidates command queue during kernel
  execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
  GT). Therefore, we flattened conditional statements. */
  if(index.x >= size.x){ return false; }
  if(index.y >= size.y){ return false; }
  if(index.z >= size.z){ return false; }
  return true;
}

//------------------------------------------------------------------------------
// cast from interpolator output to pixel type
OUTPIXELTYPE cast_pixel_with_bounds_checking(
  const OUTPIXELTYPE value,
  const float2 min_max,
  const float2 min_max_output)
{
  OUTPIXELTYPE output_value;
  // check for value min/max
  if(value < min_max_output.x)
  {
    output_value = min_max.x;
  }
  else if(value > min_max_output.y)
  {
    output_value = min_max.y;
  }
  else
  {
    output_value = value;
  }
  return output_value;
}

//------------------------------------------------------------------------------
#if defined( DIM_3 ) && defined( RESAMPLE_PRE )
__kernel void ResampleImageFilterPre(
  /* output ImageBase information */
  __constant GPUImageBase3D *output_image,
  /* */
  __global float3 *tout,
  uint3 tsize,
  /* filter parameters */
  __constant FilterParameters *parameters)
{
  uint3 global_id = (uint3)( get_global_id(0), get_global_id(1), get_global_id(2) );
  uint3 global_offset = (uint3)( get_global_offset(0), get_global_offset(1), get_global_offset(2) );

  // recalculate index
  uint3 index = global_id - global_offset;

  if(is_valid(index, tsize) && is_valid(global_id, output_image->size))
  {
    float3 point;
    if(parameters->transform_linear)
    {
      // compute continuous index for the first index
      uint3 first_index = (uint3)(0, global_id.y, global_id.z);
      point = transform_index_to_physical_point_3d(first_index, output_image);
    }
    else
    {
      point = transform_index_to_physical_point_3d(global_id, output_image);
    }

    // calculate gidx within buffer
    uint gidx = mad24(tsize.x, mad24(index.z, tsize.y, index.y), index.x);
    tout[gidx] = point;
  }
}
#endif

//------------------------------------------------------------------------------
// This kernel executed for itk::GPUResampleImageFilter with identity transform
// \sa IdentityTransform
#if defined( DIM_3 ) && defined( RESAMPLE_LOOP ) && defined( IDENTITY_TRANSFORM )
__kernel void ResampleImageFilterLoop_IdentityTransform(
  /* input ImageBase information */
  __constant GPUImageBase3D *input_image,
  /* output ImageBase information */
  __constant GPUImageBase3D *output_image,
  /* */
  __global float3 *tout,
  uint3 tsize,
  uint combo,
  /* filter parameters */
  __constant FilterParameters *parameters)
{
  uint3 global_id = (uint3)( get_global_id(0), get_global_id(1), get_global_id(2) );
  uint3 global_offset = (uint3)( get_global_offset(0), get_global_offset(1), get_global_offset(2) );

  // recalculate index
  uint3 index = global_id - global_offset;

  if(is_valid(index, tsize) && is_valid(global_id, output_image->size))
  {
    // get point
    uint tidx = mad24(tsize.x, mad24(index.z, tsize.y, index.y), index.x);
    float3 output_point = tout[tidx];

    // IdentityTransform is linear transform, execute linear call
    // \sa IdentityTransform::IsLinear()
    float3 input_point = identity_transform_point_3d(output_point);

    if(combo)
    {
      // roll it back
      tout[tidx] = input_point;
    }
    else
    {
      float3 input_index;
      transform_physical_point_to_continuous_index_3d(input_point, &input_index, input_image);
      float3 continuous_index = input_index + index.x * parameters->delta;

      // roll it back
      tout[tidx] = continuous_index;
    }
  }
}
#endif

//------------------------------------------------------------------------------
// This kernel executed for itk::GPUResampleImageFilter with matrix offset transform
// \sa MatrixOffsetTransformBase
#if defined( DIM_3 ) && defined( RESAMPLE_LOOP ) && defined( MATRIX_OFFSET_TRANSFORM )
__kernel void ResampleImageFilterLoop_MatrixOffsetTransform(
  /* input ImageBase information */
  __constant GPUImageBase3D *input_image,
  /* output ImageBase information */
  __constant GPUImageBase3D *output_image,
  /* */
  __global float3 *tout,
  uint3 tsize,
  uint combo,
  /* filter parameters */
  __constant FilterParameters *parameters,
  /* transform base parameters */
  __constant GPUMatrixOffsetTransformBase3D *transform_base)
{
  uint3 global_id = (uint3)( get_global_id(0), get_global_id(1), get_global_id(2) );
  uint3 global_offset = (uint3)( get_global_offset(0), get_global_offset(1), get_global_offset(2) );

  // recalculate index
  uint3 index = global_id - global_offset;

  if(is_valid(index, tsize) && is_valid(global_id, output_image->size))
  {
    // get point
    uint tidx = mad24(tsize.x, mad24(index.z, tsize.y, index.y), index.x);
    float3 output_point = tout[tidx];

    // MatrixOffsetTransformBase is linear transform, execute linear call
    // \sa MatrixOffsetTransformBase::IsLinear()
    float3 input_point = matrix_offset_transform_point_3d(output_point, transform_base);

    if(combo)
    {
      // roll it back
      tout[tidx] = input_point;
    }
    else
    {
      float3 input_index;
      transform_physical_point_to_continuous_index_3d(input_point, &input_index, input_image);
      float3 continuous_index = input_index + index.x * parameters->delta;

      // roll it back
      tout[tidx] = continuous_index;
    }
  }
}
#endif

//------------------------------------------------------------------------------
// This kernel executed for itk::GPUResampleImageFilter with matrix offset transform
// \sa TranslationTransform
#if defined( DIM_3 ) && defined( RESAMPLE_LOOP ) && defined( TRANSLATION_TRANSFORM )
__kernel void ResampleImageFilterLoop_TranslationTransform(
  /* input ImageBase information */
  __constant GPUImageBase3D *input_image,
  /* output ImageBase information */
  __constant GPUImageBase3D *output_image,
  /* */
  __global float3 *tout,
  uint3 tsize,
  uint combo,
  /* filter parameters */
  __constant FilterParameters *parameters,
  /* transform base parameters */
  __constant GPUTranslationTransformBase3D *transform_base)
{
  uint3 global_id = (uint3)( get_global_id(0), get_global_id(1), get_global_id(2) );
  uint3 global_offset = (uint3)( get_global_offset(0), get_global_offset(1), get_global_offset(2) );

  // recalculate index
  uint3 index = global_id - global_offset;

  if(is_valid(index, tsize) && is_valid(global_id, output_image->size))
  {
    // get point
    uint tidx = mad24(tsize.x, mad24(index.z, tsize.y, index.y), index.x);
    float3 output_point = tout[tidx];

    // TranslationTransform is linear transform, execute linear call
    // \sa TranslationTransform::IsLinear()
    float3 input_point = translation_transform_point_3d(output_point, transform_base);

    if(combo)
    {
      // roll it back
      tout[tidx] = input_point;
    }
    else
    {
      float3 input_index;
      transform_physical_point_to_continuous_index_3d(input_point, &input_index, input_image);
      float3 continuous_index = input_index + index.x * parameters->delta;

      // roll it back
      tout[tidx] = continuous_index;
    }
  }
}
#endif

//------------------------------------------------------------------------------
// This kernel executed for itk::GPUResampleImageFilter with BSpline transform
// \sa BSplineBaseTransform
#if defined( DIM_3 ) && defined( RESAMPLE_LOOP ) && defined( BSPLINE_TRANSFORM )
__kernel void ResampleImageFilterLoop_BSplineTransform(
  /* input ImageBase information */
  __constant GPUImageBase3D *input_image,
  /* output ImageBase information */
  __constant GPUImageBase3D *output_image,
  /* */
  __global float3 *tout,
  uint3 tsize,
  uint combo,
  /* filter parameters */
  __constant FilterParameters *parameters,
  /* transform coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE *transform_coefficients0,
  __constant GPUImageBase3D *transform_coefficients_image0,
  /* transform coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE *transform_coefficients1,
  __constant GPUImageBase3D *transform_coefficients_image1,
  /* transform coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE *transform_coefficients2,
  __constant GPUImageBase3D *transform_coefficients_image2)
{
  uint3 global_id = (uint3)( get_global_id(0), get_global_id(1), get_global_id(2) );
  uint3 global_offset = (uint3)( get_global_offset(0), get_global_offset(1), get_global_offset(2) );

  // recalculate index
  uint3 index = global_id - global_offset;

  if(is_valid(index, tsize) && is_valid(global_id, output_image->size))
  {
    // get point
    uint tidx = mad24(tsize.x, mad24(index.z, tsize.y, index.y), index.x);
    float3 output_point = tout[tidx];

    // BSplineBaseTransform is not linear transform, execute non linear call
    // \sa BSplineBaseTransform::IsLinear()
    float3 input_point = bspline_transform_point_3d(output_point,
      transform_coefficients0, transform_coefficients_image0,
      transform_coefficients1, transform_coefficients_image1,
      transform_coefficients2, transform_coefficients_image2);

    if(combo)
    {
      // roll it back
      tout[tidx] = input_point;
    }
    else
    {
      float3 input_index;
      transform_physical_point_to_continuous_index_3d(input_point, &input_index, input_image);

      // roll it back
      tout[tidx] = input_index;
    }
  }
}
#endif

//------------------------------------------------------------------------------
#if defined( DIM_3 ) && defined( RESAMPLE_POST )
void resample_3d_post(__global const INPIXELTYPE *in,
  __constant GPUImageBase3D *input_image,
  __global OUTPIXELTYPE *out,
  __constant GPUImageBase3D *output_image,
  __global const float3 *tin,
  const uint3 tsize,
  const uint3 global_id,
  const uint3 index,
  __constant FilterParameters *parameters,
  __constant GPUImageFunction3D *image_function,
  __global const INTERPOLATOR_PRECISION_TYPE *interpolator_coefficients,
  __constant GPUImageBase3D *interpolator_coefficients_image)
{
  const uint tidx = mad24(tsize.x, mad24(index.z, tsize.y, index.y), index.x);
  const uint gidx = mad24(output_image->size.x, mad24(global_id.z, output_image->size.y, global_id.y), global_id.x);

  const float3 continuous_index = tin[tidx];

  // evaluate input at right position and copy to the output
  if(interpolator_is_inside_buffer_3d(continuous_index, image_function))
  {
    OUTPIXELTYPE value;
    if(parameters->interpolator_is_bspline)
    {
      value = bspline_evaluate_at_continuous_index_3d(continuous_index,
        in,
        input_image,
        image_function,
        interpolator_coefficients,
        interpolator_coefficients_image);
    }
    else
    {
      value = evaluate_at_continuous_index_3d(continuous_index, in, input_image, image_function);
    }
    out[gidx] = cast_pixel_with_bounds_checking(value, parameters->min_max, parameters->min_max_output);
  }
  else
  {
    out[gidx] = parameters->default_value;
  }
}
#endif

//------------------------------------------------------------------------------
#if defined( DIM_3 ) && defined( RESAMPLE_POST )
__kernel void ResampleImageFilterPost(
  /* input ImageBase information */
  __global const INPIXELTYPE *in,
  __constant GPUImageBase3D *input_image,
  /* output ImageBase information */
  __global OUTPIXELTYPE *out,
  __constant GPUImageBase3D *output_image,
  /* */
  __global const float3 *tin,
  uint3 tsize,
  /* filter parameters */
  __constant FilterParameters *parameters,
  /* image function parameters */
  __constant GPUImageFunction3D *image_function)
{
  uint3 global_id = (uint3)( get_global_id(0), get_global_id(1), get_global_id(2) );
  uint3 global_offset = (uint3)( get_global_offset(0), get_global_offset(1), get_global_offset(2) );

  // recalculate index
  uint3 index = global_id - global_offset;

  if(is_valid(index, tsize) && is_valid(global_id, output_image->size))
  {
    // resample 3d post
    resample_3d_post(in, input_image, out, output_image, tin, tsize,
      global_id, index, parameters, image_function, 0, 0);
  }
}

//------------------------------------------------------------------------------
__kernel void ResampleImageFilterPost_InterpolatorBSpline(
  /* input ImageBase information */
  __global const INPIXELTYPE *in,
  __constant GPUImageBase3D *input_image,
  /* output ImageBase information */
  __global OUTPIXELTYPE *out,
  __constant GPUImageBase3D *output_image,
  /* */
  __global const float3 *tin,
  uint3 tsize,
  /* filter parameters */
  __constant FilterParameters *parameters,
  /* image function parameters */
  __constant GPUImageFunction3D *image_function,
  /* interpolator coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE *interpolator_coefficients,
  __constant GPUImageBase3D *interpolator_coefficients_image)
{
  uint3 global_id = (uint3)( get_global_id(0), get_global_id(1), get_global_id(2) );
  uint3 global_offset = (uint3)( get_global_offset(0), get_global_offset(1), get_global_offset(2) );

  // recalculate index
  uint3 index = global_id - global_offset;

  if(is_valid(index, tsize) && is_valid(global_id, output_image->size))
  {
    // resample 3d post
    resample_3d_post(in, input_image, out, output_image, tin, tsize,
      global_id, index, parameters, image_function,
      interpolator_coefficients, interpolator_coefficients_image);
  }
}
#endif

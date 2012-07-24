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

// Typedef for FilterParameters struct
typedef struct{
  int transform_linear;
  int interpolator_is_bspline;
  int transform_is_bspline;
  float default_value;
  float2 min_max;
  float2 min_max_output;
  float3 delta;
} FilterParameters;

// cast from interpolator output to pixel type
OUTPIXELTYPE cast_pixel_with_bounds_checking(const OUTPIXELTYPE value,
                                             const float2 min_max,
                                             const float2 min_max_output)
{
  OUTPIXELTYPE output_value;
  // check for value min/max
  if (value < min_max_output.x)
  {
    output_value = min_max.x;
  }
  else if (value > min_max_output.y)
  {
    output_value = min_max.y;
  }
  else
  {
    output_value = value;
  }
  return output_value;
}

#ifdef DIM_1
void resample_1d(__global const INPIXELTYPE* in,
                 __constant GPUImageBase1D *inputImage,
                 __global OUTPIXELTYPE* out,
                 __constant GPUImageBase1D *outputImage,
                 const uint index,
                 __constant FilterParameters* parameters,
                 __constant GPUImageFunction1D* image_function,
                 __constant GPUMatrixOffsetTransformBase1D* transform_base,
                 __global const INTERPOLATOR_PRECISION_TYPE* interpolator_coefficients,
                 __constant GPUImageBase1D *interpolator_coefficientsImage,
                 __global const INTERPOLATOR_PRECISION_TYPE* transform_coefficients,
                 __constant GPUImageBase1D *transform_coefficientsImage)
{
  // compute continuous index
  float continuous_index;
  if( parameters->transform_linear )
  {
    // compute continuous index for the first index
    uint first_index = 0;
    float point = transform_index_to_physical_point_1d(first_index, outputImage);
    float tpoint;
    if( parameters->transform_is_bspline )
    {
      tpoint = bspline_transform_point_1d(point,
        transform_coefficients, transform_coefficientsImage);
    }
    else
    {
      tpoint = transform_point_1d(point, transform_base);
    }
    float first_continuous_index;
    transform_physical_point_to_continuous_index_1d(tpoint, &first_continuous_index, inputImage);
    continuous_index = first_continuous_index + index * parameters->delta.x;
  }
  else
  {
    float point = transform_index_to_physical_point_1d(index, outputImage);
    float tpoint;
    if( parameters->transform_is_bspline )
    {
      tpoint = bspline_transform_point_1d(point,
        transform_coefficients, transform_coefficientsImage);
    }
    else
    {
      tpoint = transform_point_1d(point, transform_base);
    }
    transform_physical_point_to_continuous_index_1d(tpoint, &continuous_index, inputImage);
  }

  // evaluate input at right position and copy to the output
  if ( interpolator_is_inside_buffer_1d(continuous_index, image_function) )
  {
    OUTPIXELTYPE value;
    if ( parameters->interpolator_is_bspline )
    {
      value = bspline_evaluate_at_continuous_index_1d(continuous_index,
        in, inputImage, image_function, interpolator_coefficients, interpolator_coefficientsImage);
    }
    else
    {
      value = evaluate_at_continuous_index_1d(continuous_index, in, inputImage, image_function);
    }
    out[index] = cast_pixel_with_bounds_checking(value, parameters->min_max, parameters->min_max_output);
  }
  else
  {
    out[index] = parameters->default_value;
  }
}
#endif

#ifdef DIM_1
// General ResampleImageFilter implementation in 1D
__kernel void ResampleImageFilter(
  /* input ImageBase information */
  __global const INPIXELTYPE* in,
  __constant GPUImageBase1D* inputImage,
  /* output ImageBase information */
  __global OUTPIXELTYPE* out,
  __constant GPUImageBase1D* outputImage,
  /* filter parameters */
  __constant FilterParameters* parameters,
  /* image function parameters */
  __constant GPUImageFunction1D* imageFunction,
  /* transform base parameters */
  __constant GPUMatrixOffsetTransformBase1D* transformBase)
{
  uint index = get_global_id(0);

  if(index < outputImage->Size)
  {
    // resample
    resample_1d(in, inputImage, out, outputImage, index,
      parameters, imageFunction, transformBase,
      0, 0,  // interpolate coefficients are null
      0, 0); // transform coefficients are null
  }
}

// ResampleImageFilter with BSplineInterpolateImageFunction implementation in 1D
__kernel void ResampleImageFilter_InterpolatorBSpline(
  /* input ImageBase information */
  __global const INPIXELTYPE* in,
  __constant GPUImageBase1D* inputImage,
  /* output ImageBase information */
  __global OUTPIXELTYPE* out,
  __constant GPUImageBase1D* outputImage,
  /* filter parameters */
  __constant FilterParameters* parameters,
  /* image function parameters */
  __constant GPUImageFunction1D* imageFunction,
  /* interpolator coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE* interpolator_coefficients,
  __constant GPUImageBase1D* interpolatorCoefficientsImage,
  /* transform base parameters */
  __constant GPUMatrixOffsetTransformBase1D* transformBase)
{
  uint index = get_global_id(0);

  if(index < outputImage->Size)
  {
    // resample
    resample_1d(in, inputImage, out, outputImage, index,
      parameters, imageFunction, transformBase,
      interpolator_coefficients, interpolatorCoefficientsImage,
      0, 0); // transform coefficients are null
  }
}

// ResampleImageFilter with BSplineTransform implementation in 1D
__kernel void ResampleImageFilter_TransformBSpline(
  /* input ImageBase information */
  __global const INPIXELTYPE* in,
  __constant GPUImageBase1D* inputImage,
  /* output ImageBase information */
  __global OUTPIXELTYPE* out,
  __constant GPUImageBase1D* outputImage,
  /* filter parameters */
  __constant FilterParameters* parameters,
  /* image function parameters */
  __constant GPUImageFunction1D* imageFunction,
  /* transform coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE* transform_coefficients,
  __constant GPUImageBase1D* transformCoefficientsImage)
{
  uint index = get_global_id(0);

  if(index < outputImage->Size)
  {
    // resample
    resample_1d(in, inputImage, out, outputImage, index,
      parameters, imageFunction,
      0, // transform base is null
      0, 0, // interpolate coefficients are null
      transform_coefficients, transformCoefficientsImage);
  }
}

// ResampleImageFilter with BSplineInterpolateImageFunction and
// BSplineTransform implementation in 1D
__kernel void ResampleImageFilter_InterpolatorBSpline_TransformBSpline(
  /* input ImageBase information */
  __global const INPIXELTYPE* in,
  __constant GPUImageBase1D* inputImage,
  /* output ImageBase information */
  __global OUTPIXELTYPE* out,
  __constant GPUImageBase1D* outputImage,
  /* filter parameters */
  __constant FilterParameters* parameters,
  /* image function parameters */
  __constant GPUImageFunction1D* imageFunction,
  /* interpolator coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE* interpolator_coefficients,
  __constant GPUImageBase1D* interpolatorCoefficientsImage,
  /* transform coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE* transform_coefficients,
  __constant GPUImageBase1D* transformCoefficientsImage)
{
  uint index = get_global_id(0);

  if(index < outputImage->Size)
  {
    // resample
    resample_1d(in, inputImage, out, outputImage, index,
      parameters, imageFunction,
      0, // transform base is null
      interpolator_coefficients, interpolatorCoefficientsImage,
      transform_coefficients, transformCoefficientsImage);
  }
}
#endif

//------------------------------------------------------------------------------
#ifdef DIM_2
void resample_2d(__global const INPIXELTYPE* in,
                 __constant GPUImageBase2D *inputImage,
                 __global OUTPIXELTYPE* out,
                 __constant GPUImageBase2D *outputImage,
                 const uint2 index,
                 __constant FilterParameters* parameters,
                 __constant GPUImageFunction2D* image_function,
                 __constant GPUMatrixOffsetTransformBase2D* transform_base,
                 __global const INTERPOLATOR_PRECISION_TYPE* interpolator_coefficients,
                 __constant GPUImageBase2D *interpolator_coefficientsImage,
                 __global const INTERPOLATOR_PRECISION_TYPE* transform_coefficients0,
                 __constant GPUImageBase2D *transform_coefficientsImage0,
                 __global const INTERPOLATOR_PRECISION_TYPE* transform_coefficients1,
                 __constant GPUImageBase2D *transform_coefficientsImage1)
{
  uint gidx = outputImage->Size.x * index.y + index.x;

  // compute continuous index
  float2 continuous_index;
  if( parameters->transform_linear )
  {
    // compute continuous index for the first index
    uint2 first_index = (uint2)( 0, index.y );
    float2 point = transform_index_to_physical_point_2d(first_index, outputImage);
    float2 tpoint;
    if( parameters->transform_is_bspline )
    {
      tpoint = bspline_transform_point_2d(point,
        transform_coefficients0, transform_coefficientsImage0,
        transform_coefficients1, transform_coefficientsImage1);
    }
    else
    {
      tpoint = transform_point_2d(point, transform_base);
    }
    float2 first_continuous_index;
    transform_physical_point_to_continuous_index_2d(tpoint, &first_continuous_index, inputImage);
    continuous_index = first_continuous_index + index.x * parameters->delta.xy;
  }
  else
  {
    float2 point = transform_index_to_physical_point_2d(index, outputImage);
    float2 tpoint;
    if( parameters->transform_is_bspline )
    {
      tpoint = bspline_transform_point_2d(point,
        transform_coefficients0, transform_coefficientsImage0,
        transform_coefficients1, transform_coefficientsImage1);
    }
    else
    {
      tpoint = transform_point_2d(point, transform_base);
    }
    transform_physical_point_to_continuous_index_2d(tpoint, &continuous_index, inputImage);
  }

  // evaluate input at right position and copy to the output
  if ( interpolator_is_inside_buffer_2d(continuous_index, image_function) )
  {
    OUTPIXELTYPE value;
    if ( parameters->interpolator_is_bspline )
    {
      value = bspline_evaluate_at_continuous_index_2d(continuous_index,
        in, inputImage, image_function, interpolator_coefficients, interpolator_coefficientsImage);
    }
    else
    {
      value = evaluate_at_continuous_index_2d(continuous_index, in, inputImage, image_function);
    }
    out[gidx] = cast_pixel_with_bounds_checking(value, parameters->min_max, parameters->min_max_output);
  }
  else
  {
    out[gidx] = parameters->default_value;
  }
}
#endif

#ifdef DIM_2
// General ResampleImageFilter implementation in 2D
__kernel void ResampleImageFilter(
  /* input ImageBase information */
  __global const INPIXELTYPE* in,
  __constant GPUImageBase2D* inputImage,
  /* output ImageBase information */
  __global OUTPIXELTYPE* out,
  __constant GPUImageBase2D* outputImage,
  /* filter parameters */
  __constant FilterParameters* parameters,
  /* image function parameters */
  __constant GPUImageFunction2D* imageFunction,
  /* transform base parameters */
  __constant GPUMatrixOffsetTransformBase2D* transformBase)
{
  uint2 index = (uint2)(get_global_id(0), get_global_id(1));
  if(index.x < outputImage->Size.x && index.y < outputImage->Size.y)
  {
    // resample
    resample_2d(in, inputImage, out, outputImage, index,
      parameters, imageFunction, transformBase,
      0, 0,   // interpolate coefficients are null
      0, 0,   // transform coefficients0 are null
      0, 0);  // transform coefficients1 are null
  }
}

// ResampleImageFilter with BSplineInterpolateImageFunction implementation in 2D
__kernel void ResampleImageFilter_InterpolatorBSpline(
  /* input ImageBase information */
  __global const INPIXELTYPE* in,
  __constant GPUImageBase2D* inputImage,
  /* output ImageBase information */
  __global OUTPIXELTYPE* out,
  __constant GPUImageBase2D* outputImage,
  /* filter parameters */
  __constant FilterParameters* parameters,
  /* image function parameters */
  __constant GPUImageFunction2D* imageFunction,
  /* interpolator coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE* interpolator_coefficients,
  __constant GPUImageBase2D* interpolatorCoefficientsImage,
    /* transform base parameters */
  __constant GPUMatrixOffsetTransformBase2D* transformBase)
{
  uint2 index = (uint2)(get_global_id(0), get_global_id(1));
  if(index.x < outputImage->Size.x && index.y < outputImage->Size.y)
  {
    // resample
    resample_2d(in, inputImage, out, outputImage, index,
      parameters, imageFunction, transformBase,
      interpolator_coefficients, interpolatorCoefficientsImage,
      0, 0,  // transform coefficients0 are null
      0, 0); // transform coefficients1 are null
  }
}

// ResampleImageFilter with BSplineTransform implementation in 2D
__kernel void ResampleImageFilter_TransformBSpline(
  /* input ImageBase information */
  __global const INPIXELTYPE* in,
  __constant GPUImageBase2D* inputImage,
  /* output ImageBase information */
  __global OUTPIXELTYPE* out,
  __constant GPUImageBase2D* outputImage,
  /* filter parameters */
  __constant FilterParameters* parameters,
  /* image function parameters */
  __constant GPUImageFunction2D* imageFunction,
  /* transform coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE* transform_coefficients0,
  __constant GPUImageBase2D* transformCoefficientsImage0,
  /* transform coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE* transform_coefficients1,
  __constant GPUImageBase2D* transformCoefficientsImage1)
{
  uint2 index = (uint2)(get_global_id(0), get_global_id(1));
  if(index.x < outputImage->Size.x && index.y < outputImage->Size.y)
  {
    // resample
    resample_2d(in, inputImage, out, outputImage, index,
      parameters, imageFunction,
      0, // transform base is null
      0, 0, // interpolate coefficients are null
      transform_coefficients0, transformCoefficientsImage0,
      transform_coefficients1, transformCoefficientsImage1);
  }
}

// ResampleImageFilter with BSplineInterpolateImageFunction and
// BSplineTransform implementation in 2D
__kernel void ResampleImageFilter_InterpolatorBSpline_TransformBSpline(
  /* input ImageBase information */
  __global const INPIXELTYPE* in,
  __constant GPUImageBase2D* inputImage,
  /* output ImageBase information */
  __global OUTPIXELTYPE* out,
  __constant GPUImageBase2D* outputImage,
  /* filter parameters */
  __constant FilterParameters* parameters,
  /* image function parameters */
  __constant GPUImageFunction2D* imageFunction,
  /* interpolator coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE* interpolator_coefficients,
  __constant GPUImageBase2D* interpolatorCoefficientsImage,
  /* transform coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE* transform_coefficients0,
  __constant GPUImageBase2D* transformCoefficientsImage0,
  /* transform coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE* transform_coefficients1,
  __constant GPUImageBase2D* transformCoefficientsImage1)
{
  uint2 index = (uint2)(get_global_id(0), get_global_id(1));
  if(index.x < outputImage->Size.x && index.y < outputImage->Size.y)
  {
    // resample
    resample_2d(in, inputImage, out, outputImage, index,
      parameters, imageFunction,
      0, // transform base is null
      interpolator_coefficients, interpolatorCoefficientsImage,
      transform_coefficients0, transformCoefficientsImage0,
      transform_coefficients1, transformCoefficientsImage1);
  }
}
#endif

//------------------------------------------------------------------------------
#ifdef DIM_3
void resample_3d(__global const INPIXELTYPE* in,
                 __constant GPUImageBase3D *inputImage,
                 __global OUTPIXELTYPE* out,
                 __constant GPUImageBase3D *outputImage,
                 const uint3 index,
                 __constant FilterParameters* parameters,
                 __constant GPUImageFunction3D* image_function,
                 __constant GPUMatrixOffsetTransformBase3D* transform_base,
                 __global const INTERPOLATOR_PRECISION_TYPE* interpolator_coefficients,
                 __constant GPUImageBase3D *interpolator_coefficientsImage,
                 __global const INTERPOLATOR_PRECISION_TYPE* transform_coefficients0,
                 __constant GPUImageBase3D *transform_coefficientsImage0,
                 __global const INTERPOLATOR_PRECISION_TYPE* transform_coefficients1,
                 __constant GPUImageBase3D *transform_coefficientsImage1,
                 __global const INTERPOLATOR_PRECISION_TYPE* transform_coefficients2,
                 __constant GPUImageBase3D *transform_coefficientsImage2)
{
  uint gidx = outputImage->Size.x *(index.z * outputImage->Size.y + index.y) + index.x;

  // compute continuous index
  float3 continuous_index;
  if( parameters->transform_linear )
  {
    // compute continuous index for the first index
    uint3 first_index = (uint3)(0, index.y, index.z);
    float3 point = transform_index_to_physical_point_3d(first_index, outputImage);
    float3 tpoint;
    if( parameters->transform_is_bspline )
    {
      tpoint = bspline_transform_point_3d(point,
        transform_coefficients0, transform_coefficientsImage0,
        transform_coefficients1, transform_coefficientsImage1,
        transform_coefficients2, transform_coefficientsImage2);
    }
    else
    {
      tpoint = transform_point_3d(point, transform_base);
    }
    float3 first_continuous_index;
    transform_physical_point_to_continuous_index_3d(tpoint, &first_continuous_index, inputImage);
    continuous_index = first_continuous_index + index.x * parameters->delta;
  }
  else
  {
    float3 point = transform_index_to_physical_point_3d(index, outputImage);
    float3 tpoint;
    if( parameters->transform_is_bspline )
    {
      tpoint = bspline_transform_point_3d(point,
        transform_coefficients0, transform_coefficientsImage0,
        transform_coefficients1, transform_coefficientsImage1,
        transform_coefficients2, transform_coefficientsImage2);
    }
    else
    {
      tpoint = transform_point_3d(point, transform_base);
    }
    transform_physical_point_to_continuous_index_3d(tpoint, &continuous_index, inputImage);
  }

  // evaluate input at right position and copy to the output
  if ( interpolator_is_inside_buffer_3d(continuous_index, image_function) )
  {
    OUTPIXELTYPE value;
    if ( parameters->interpolator_is_bspline )
    {
      value = bspline_evaluate_at_continuous_index_3d(continuous_index,
        in, inputImage, image_function, interpolator_coefficients, interpolator_coefficientsImage);
    }
    else
    {
      value = evaluate_at_continuous_index_3d(continuous_index, in, inputImage, image_function);
    }
    out[gidx] = cast_pixel_with_bounds_checking(value, parameters->min_max, parameters->min_max_output);
  }
  else
  {
    out[gidx] = parameters->default_value;
  }
}
#endif

#ifdef DIM_3
// General ResampleImageFilter implementation in 3D
__kernel void ResampleImageFilter(
  /* input ImageBase information */
  __global const INPIXELTYPE* in,
  __constant GPUImageBase3D* inputImage,
  /* output ImageBase information */
  __global OUTPIXELTYPE* out,
  __constant GPUImageBase3D* outputImage,
  /* filter parameters */
  __constant FilterParameters* parameters,
  /* image function parameters */
  __constant GPUImageFunction3D* imageFunction,
  /* transform base parameters */
  __constant GPUMatrixOffsetTransformBase2D* transformBase)
{
  uint3 index = (uint3)(get_global_id(0), get_global_id(1), get_global_id(2));

  /* NOTE: More than three-level nested conditional statements (e.g.,
  if A && B && C..) invalidates command queue during kernel
  execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
  GT). Therefore, we flattened conditional statements. */
  bool isValid = true;
  if(index.x >= outputImage->Size.x) isValid = false;
  if(index.y >= outputImage->Size.y) isValid = false;
  if(index.z >= outputImage->Size.z) isValid = false;

  if( isValid )
  {
    // resample
    resample_3d(in, inputImage, out, outputImage, index,
      parameters, imageFunction, transformBase,
      0, 0,  // interpolate coefficients are null
      0, 0,  // transform coefficients0 are null
      0, 0,  // transform coefficients1 are null
      0, 0); // transform coefficients2 are null
  }
}

// ResampleImageFilter with BSplineInterpolateImageFunction implementation in 3D
__kernel void ResampleImageFilter_InterpolatorBSpline(
  /* input ImageBase information */
  __global const INPIXELTYPE* in,
  __constant GPUImageBase3D* inputImage,
  /* output ImageBase information */
  __global OUTPIXELTYPE* out,
  __constant GPUImageBase3D* outputImage,
  /* filter parameters */
  __constant FilterParameters* parameters,
  /* image function parameters */
  __constant GPUImageFunction3D* imageFunction,
  /* interpolator coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE* interpolator_coefficients,
  __constant GPUImageBase3D* interpolatorCoefficientsImage,
  /* transform base parameters */
  __constant GPUMatrixOffsetTransformBase3D* transformBase)
{
  uint3 index = (uint3)(get_global_id(0), get_global_id(1), get_global_id(2));

  /* NOTE: More than three-level nested conditional statements (e.g.,
  if A && B && C..) invalidates command queue during kernel
  execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
  GT). Therefore, we flattened conditional statements. */
  bool isValid = true;
  if(index.x >= outputImage->Size.x) isValid = false;
  if(index.y >= outputImage->Size.y) isValid = false;
  if(index.z >= outputImage->Size.z) isValid = false;

  if( isValid )
  {
    // resample
    resample_3d(in, inputImage, out, outputImage, index,
      parameters, imageFunction, transformBase,
      interpolator_coefficients, interpolatorCoefficientsImage,
      0, 0,  // transform coefficients0 are null
      0, 0,  // transform coefficients1 are null
      0, 0); // transform coefficients2 are null
  }
}

// ResampleImageFilter with BSplineTransform implementation in 3D
__kernel void ResampleImageFilter_TransformBSpline(
  /* input ImageBase information */
  __global const INPIXELTYPE* in,
  __constant GPUImageBase3D* inputImage,
  /* output ImageBase information */
  __global OUTPIXELTYPE* out,
  __constant GPUImageBase3D* outputImage,
  /* filter parameters */
  __constant FilterParameters* parameters,
  /* image function parameters */
  __constant GPUImageFunction3D* imageFunction,
  /* transform coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE* transform_coefficients0,
  __constant GPUImageBase3D* transformCoefficientsImage0,
  /* transform coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE* transform_coefficients1,
  __constant GPUImageBase3D* transformCoefficientsImage1,
  /* transform coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE* transform_coefficients2,
  __constant GPUImageBase3D* transformCoefficientsImage2)
{
  uint3 index = (uint3)(get_global_id(0), get_global_id(1), get_global_id(2));

  /* NOTE: More than three-level nested conditional statements (e.g.,
  if A && B && C..) invalidates command queue during kernel
  execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
  GT). Therefore, we flattened conditional statements. */
  bool isValid = true;
  if(index.x >= outputImage->Size.x) isValid = false;
  if(index.y >= outputImage->Size.y) isValid = false;
  if(index.z >= outputImage->Size.z) isValid = false;

  if( isValid )
  {
    // resample
    resample_3d(in, inputImage, out, outputImage, index,
      parameters, imageFunction,
      0, // transform base is null
      transform_coefficients0, transformCoefficientsImage0,
      transform_coefficients0, transformCoefficientsImage0,
      transform_coefficients1, transformCoefficientsImage1,
      transform_coefficients2, transformCoefficientsImage2);
  }
}

// ResampleImageFilter with BSplineInterpolateImageFunction and
// BSplineTransform implementation in 3D
__kernel void ResampleImageFilter_InterpolatorBSpline_TransformBSpline(
  /* input ImageBase information */
  __global const INPIXELTYPE* in,
  __constant GPUImageBase3D* inputImage,
  /* output ImageBase information */
  __global OUTPIXELTYPE* out,
  __constant GPUImageBase3D* outputImage,
  /* filter parameters */
  __constant FilterParameters* parameters,
  /* image function parameters */
  __constant GPUImageFunction3D* imageFunction,
  /* interpolator coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE* interpolator_coefficients,
  __constant GPUImageBase3D* interpolatorCoefficientsImage,
  /* transform coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE* transform_coefficients0,
  __constant GPUImageBase3D* transformCoefficientsImage0,
  /* transform coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE* transform_coefficients1,
  __constant GPUImageBase3D* transformCoefficientsImage1,
  /* transform coefficients ImageBase information */
  __global const INTERPOLATOR_PRECISION_TYPE* transform_coefficients2,
  __constant GPUImageBase3D* transformCoefficientsImage2)
{
  uint3 index = (uint3)(get_global_id(0), get_global_id(1), get_global_id(2));

  /* NOTE: More than three-level nested conditional statements (e.g.,
  if A && B && C..) invalidates command queue during kernel
  execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
  GT). Therefore, we flattened conditional statements. */
  bool isValid = true;
  if(index.x >= outputImage->Size.x) isValid = false;
  if(index.y >= outputImage->Size.y) isValid = false;
  if(index.z >= outputImage->Size.z) isValid = false;

  if( isValid )
  {
    // resample
    resample_3d(in, inputImage, out, outputImage, index,
      parameters, imageFunction,
      0, // transform base is null
      interpolator_coefficients, interpolatorCoefficientsImage,
      transform_coefficients0, transformCoefficientsImage0,
      transform_coefficients1, transformCoefficientsImage1,
      transform_coefficients2, transformCoefficientsImage2);
  }
}
#endif

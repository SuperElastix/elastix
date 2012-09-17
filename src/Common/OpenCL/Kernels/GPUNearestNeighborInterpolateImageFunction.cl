/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

// OpenCL implementation of itk::NearestNeighborInterpolateImageFunction

//------------------------------------------------------------------------------
float bspline_evaluate_at_continuous_index_1d(const float index,
                                              __global const INPIXELTYPE *in,
                                              __constant GPUImageBase1D *in_image,
                                              __constant GPUImageFunction1D *image_function,
                                              __global const INTERPOLATOR_PRECISION_TYPE *coefficients,
                                              __constant GPUImageBase1D *coefficients_image)
{
  return 0.0f;
}

//------------------------------------------------------------------------------
float bspline_evaluate_at_continuous_index_2d(const float2 index,
                                              __global const INPIXELTYPE *in,
                                              __constant GPUImageBase2D *in_image,
                                              __constant GPUImageFunction2D *image_function,
                                              __global const INTERPOLATOR_PRECISION_TYPE *coefficients,
                                              __constant GPUImageBase2D *coefficients_image)
{
  return 0.0f;
}

//------------------------------------------------------------------------------
float bspline_evaluate_at_continuous_index_3d(const float3 index,
                                              __global const INPIXELTYPE *in,
                                              __constant GPUImageBase3D *in_image,
                                              __constant GPUImageFunction3D *image_function,
                                              __global const INTERPOLATOR_PRECISION_TYPE *coefficients,
                                              __constant GPUImageBase3D *coefficients_image)
{
  return 0.0f;
}

//------------------------------------------------------------------------------
#ifdef DIM_1
float evaluate_at_continuous_index_1d(const float index,
                                      __global const INPIXELTYPE *in,
                                      __constant GPUImageBase1D *image,
                                      __constant GPUImageFunction1D *image_function)
{
  uint         nindex = convert_continuous_index_to_nearest_index_1d(index);
  uint gidx = nindex;
  float        image_value = (float)(in[gidx]);

  return image_value;
}

#endif // DIM_1

//------------------------------------------------------------------------------
#ifdef DIM_2
float evaluate_at_continuous_index_2d(const float2 index,
                                      __global const INPIXELTYPE *in,
                                      __constant GPUImageBase2D *image,
                                      __constant GPUImageFunction2D *image_function)
{
  uint2 nindex = convert_continuous_index_to_nearest_index_2d(index);
  uint  gidx = mad24(image->size.x, nindex.y, nindex.x);
  float image_value = (float)(in[gidx]);

  return image_value;
}

#endif // DIM_2

//------------------------------------------------------------------------------
#ifdef DIM_3
float evaluate_at_continuous_index_3d(const float3 index,
                                      __global const INPIXELTYPE *in,
                                      __constant GPUImageBase3D *image,
                                      __constant GPUImageFunction3D *image_function)
{
  uint3       nindex = convert_continuous_index_to_nearest_index_3d(index);
  uint gidx = mad24(image->size.x, mad24(nindex.z, image->size.y, nindex.y), nindex.x);
  float       image_value = (float)(in[gidx]);

  return image_value;
}

#endif // DIM_3

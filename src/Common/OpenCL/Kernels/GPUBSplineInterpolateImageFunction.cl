/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

// OpenCL implementation of itk::BSplineInterpolateImageFunction

//------------------------------------------------------------------------------
// purposely not implemented. Supporting OpenCL compilation.
float evaluate_at_continuous_index_1d(const float index,
                                      __global const INPIXELTYPE *in,
                                      __constant GPUImageBase1D *image,
                                      __constant GPUImageFunction1D *image_function)
{
  return 0.0f;
}

//------------------------------------------------------------------------------
// purposely not implemented. Supporting OpenCL compilation.
float evaluate_at_continuous_index_2d(const float2 index,
                                      __global const INPIXELTYPE *in,
                                      __constant GPUImageBase2D *image,
                                      __constant GPUImageFunction2D *image_function)
{
  return 0.0f;
}

//------------------------------------------------------------------------------
// purposely not implemented. Supporting OpenCL compilation.
float evaluate_at_continuous_index_3d(const float3 index,
                                      __global const INPIXELTYPE *in,
                                      __constant GPUImageBase3D *image,
                                      __constant GPUImageFunction3D *image_function)
{
  return 0.0f;
}

//------------------------------------------------------------------------------
void determine(long *evaluate_index, const float index,
               const uint offset, const float half_offset,
               const uint spline_order)
{
  long indx = (long)(floor(index + half_offset) - spline_order / 2);

  for( uint k = 0; k <= spline_order; k++ )
  {
    uint eindx = offset + k;
    evaluate_index[eindx] = indx++;
  }
}

//------------------------------------------------------------------------------
#ifdef DIM_1
void determine_region_of_support_1d(long *evaluate_index,
                                    const float continuous_index,
                                    const uint spline_order)
{
  const float half_offset = spline_order & 1 ? 0.0 : 0.5;

  determine(evaluate_index, continuous_index, 0, half_offset, spline_order);
}
#endif // DIM_1

//------------------------------------------------------------------------------
#ifdef DIM_2
void determine_region_of_support_2d(long *evaluate_index,
                                    const float2 continuous_index,
                                    const uint spline_order)
{
  const float half_offset = spline_order & 1 ? 0.0 : 0.5;

  determine( evaluate_index, continuous_index.x, 0, half_offset, spline_order );
  determine( evaluate_index, continuous_index.y, (spline_order + 1), half_offset, spline_order );
}
#endif // DIM_2

//------------------------------------------------------------------------------
#ifdef DIM_3
void determine_region_of_support_3d(long *evaluate_index,
                                    const float3 continuous_index,
                                    const uint spline_order)
{
  const float half_offset = spline_order & 1 ? 0.0 : 0.5;

  determine( evaluate_index, continuous_index.x, 0, half_offset, spline_order );
  determine( evaluate_index, continuous_index.y, (spline_order + 1), half_offset, spline_order );
  determine( evaluate_index, continuous_index.z, (spline_order + 1) * 2, half_offset, spline_order );
}
#endif // DIM_3

//------------------------------------------------------------------------------
// get offset in array
uint get_array_offset( const uint x, const uint y, const uint width )
{
  uint idx = mad24( width, y, x );
  return idx;
}

//------------------------------------------------------------------------------
#ifdef DIM_1
void set_interpolation_weights_1d(const float index,
                                  const long *evaluate_index,
                                  float *weights,
                                  const uint spline_order)
{
  // For speed improvements we could make each case a separate function and use
  // function pointers to reference the correct weight order.
  // Left as is for now for readability.
  float w, w2, w4, t, t0, t1, t2;

  // create float x from float, makes it easy to use
  float x = index;
  uint  width = spline_order + 1;

  // spline_order must be between 0 and 5.
  if( spline_order == 3 )
  {
    w = x - (float)evaluate_index[ 1 ];
    weights[ 3 ] = (1.0 / 6.0) * w * w * w;
    weights[ 0 ] = (1.0 / 6.0) + 0.5 * w * (w - 1.0) - weights[ 3 ];
    weights[ 2 ] = w + weights[ 0 ] - 2.0 * weights[ 3 ];
    weights[ 1 ] = 1.0 - weights[ 0 ] - weights[ 2 ] - weights[ 3 ];
  }
  else if( spline_order == 0 )
  {
    weights[ 0 ] = 1; // implements nearest neighbor
  }
  else if( spline_order == 1 )
  {
    w = x - (float)evaluate_index[ 1 ];
    weights[ 1 ] = w;
    weights[ 0 ] = 1.0 - w;
  }
  else if( spline_order == 2 )
  {
    w = x - (float)evaluate_index[ 1 ];
    weights[ 1 ] = 0.75 - w * w;
    weights[ 2 ] = 0.5 * ( w - weights[ 1 ] + 1.0 );
    weights[ 0 ] = 1.0 - weights[ 1 ] - weights[ 2 ];
  }
  else if( spline_order == 4 )
  {
    w = x - (float)evaluate_index[ 2 ];
    w2 = w * w;
    t2 = ( 0.5 - w ); t2 *= t2; t2 *= t2;
    weights[ 0 ] = (1.0 / 24.0) * t2;
    t = (1.0 / 6.0) * w2;
    t0 = w * ( t - 11.0 / 24.0 );
    t1 = 19.0 / 96.0 + w2 * ( 0.25 - t );
    weights[ 1 ] = t1 + t0;
    weights[ 3 ] = t1 - t0;
    weights[ 4 ] = weights[ 0 ] + t0 + 0.5 * w;
    weights[ 2 ] = 1.0 - weights[ 0 ] - weights[ 1 ] - weights[ 3 ] - weights[ 4 ];
  }
  else if( spline_order == 5 )
  {
    w = x - (float)evaluate_index[ 2 ];
    w2 = w * w;
    weights[ 5 ] = ( 1.0 / 120.0 ) * w * w2 * w2;
    w2 -= w;
    w4 = w2 * w2;
    w -= 0.5;
    t = w2 * ( w2 - 3.0 );
    weights[ 0 ] = ( 1.0 / 24.0 ) * ( 1.0 / 5.0 + w2 + w4 ) - weights[ 5 ];
    t0 = ( 1.0 / 24.0 ) * ( w2 * ( w2 - 5.0 ) + 46.0 / 5.0 );
    t1 = ( -1.0 / 12.0 ) * w * ( t + 4.0 );
    weights[ 2 ] = t0 + t1;
    weights[ 3 ] = t0 - t1;
    t0 = ( 1.0 / 16.0 ) * ( 9.0 / 5.0 - t );
    t1 = ( 1.0 / 24.0 ) * w * ( w4 - w2 - 5.0 );
    weights[ 1 ] = t0 + t1;
    weights[ 4 ] = t0 - t1;
  }
}
#endif // DIM_1

//------------------------------------------------------------------------------
#ifdef DIM_2
void set_interpolation_weights_2d(const float2 index,
                                  const long *evaluate_index,
                                  float *weights,
                                  const uint spline_order)
{
  // For speed improvements we could make each case a separate function and use
  // function pointers to reference the correct weight order.
  // Left as is for now for readability.
  float w, w2, w4, t, t0, t1, t2;

  // create float x[2] from float2, makes it easy to use in loops
  float x[2];
  x[0] = index.x;
  x[1] = index.y;
  uint width = spline_order + 1;

  // spline_order must be between 0 and 5.
  if( spline_order == 3 )
  {
    for( uint n = 0; n < 2; n++ )
    {
      uint ao0 = get_array_offset( 0, n, width );
      uint ao1 = get_array_offset( 1, n, width );
      uint ao2 = get_array_offset( 2, n, width );
      uint ao3 = get_array_offset( 3, n, width );

      w = x[n] - (float)evaluate_index[ ao1 ];
      weights[ ao3 ] = (1.0 / 6.0) * w * w * w;
      weights[ ao0 ] = (1.0 / 6.0) + 0.5 * w * ( w - 1.0 ) - weights[ ao3 ];
      weights[ ao2 ] = w + weights[ ao0 ] - 2.0 * weights[ ao3 ];
      weights[ ao1 ] = 1.0 - weights[ ao0 ] - weights[ ao2 ] - weights[ ao3 ];
    }
  }
  else if( spline_order == 0 )
  {
    for( uint n = 0; n < 2; n++ )
    {
      uint ao0 = get_array_offset( 0, n, width );

      weights[ ao0 ] = 1;
    }
  }
  else if( spline_order == 1 )
  {
    for( uint n = 0; n < 2; n++ )
    {
      uint ao0 = get_array_offset( 0, n, width );
      uint ao1 = get_array_offset( 1, n, width );

      w = x[n] - (float)evaluate_index[ ao0 ];
      weights[ ao1 ] = w;
      weights[ ao0 ] = 1.0 - w;
    }
  }
  else if( spline_order == 2 )
  {
    for( uint n = 0; n < 2; n++ )
    {
      uint ao0 = get_array_offset( 0, n, width );
      uint ao1 = get_array_offset( 1, n, width );
      uint ao2 = get_array_offset( 2, n, width );

      w = x[n] - (float)evaluate_index[ ao1 ];
      weights[ ao1 ] = 0.75 - w * w;
      weights[ ao2 ] = 0.5 * ( w - weights[ ao1 ] + 1.0 );
      weights[ ao0 ] = 1.0 - weights[ ao1 ] - weights[ ao2 ];
    }
  }
  else if( spline_order == 4 )
  {
    for( uint n = 0; n < 2; n++ )
    {
      uint ao0 = get_array_offset( 0, n, width );
      uint ao1 = get_array_offset( 1, n, width );
      uint ao2 = get_array_offset( 2, n, width );
      uint ao3 = get_array_offset( 3, n, width );
      uint ao4 = get_array_offset( 4, n, width );

      w = x[n] - (float)evaluate_index[ ao2 ];
      w2 = w * w;
      t2 = ( 0.5 - w ); t2 *= t2; t2 *= t2;
      weights[ ao0 ] = (1.0 / 24.0) * t2;
      t = (1.0 / 6.0) * w2;
      t0 = w * ( t - 11.0 / 24.0 );
      t1 = 19.0 / 96.0 + w2 * ( 0.25 - t );
      weights[ ao1 ] = t1 + t0;
      weights[ ao3 ] = t1 - t0;
      weights[ ao4 ] = weights[ ao0 ] + t0 + 0.5 * w;
      weights[ ao2 ] = 1.0 - weights[ ao0 ] - weights[ ao1 ] - weights[ ao3 ] - weights[ ao4 ];
    }
  }
  else if( spline_order == 5 )
  {
    for( uint n = 0; n < 2; n++ )
    {
      uint ao0 = get_array_offset( 0, n, width );
      uint ao1 = get_array_offset( 1, n, width );
      uint ao2 = get_array_offset( 2, n, width );
      uint ao3 = get_array_offset( 3, n, width );
      uint ao4 = get_array_offset( 4, n, width );
      uint ao5 = get_array_offset( 5, n, width );

      w = x[n] - (float)evaluate_index[ ao2 ];
      w2 = w * w;
      weights[ ao5 ] = ( 1.0 / 120.0 ) * w * w2 * w2;
      w2 -= w;
      w4 = w2 * w2;
      w -= 0.5;
      t = w2 * ( w2 - 3.0 );
      weights[ ao0 ] = ( 1.0 / 24.0 ) * ( 1.0 / 5.0 + w2 + w4 ) - weights[ ao5 ];
      t0 = ( 1.0 / 24.0 ) * ( w2 * ( w2 - 5.0 ) + 46.0 / 5.0 );
      t1 = ( -1.0 / 12.0 ) * w * ( t + 4.0 );
      weights[ ao2 ] = t0 + t1;
      weights[ ao3 ] = t0 - t1;
      t0 = ( 1.0 / 16.0 ) * ( 9.0 / 5.0 - t );
      t1 = ( 1.0 / 24.0 ) * w * ( w4 - w2 - 5.0 );
      weights[ ao1 ] = t0 + t1;
      weights[ ao4 ] = t0 - t1;
    }
  }
}
#endif // DIM_2

//------------------------------------------------------------------------------
#ifdef DIM_3
void set_interpolation_weights_3d(const float3 index,
                                  const long *evaluate_index,
                                  float *weights,
                                  const uint spline_order)
{
  // For speed improvements we could make each case a separate function and use
  // function pointers to reference the correct weight order.
  // Left as is for now for readability.
  float w, w2, w4, t, t0, t1, t2;

  // create float x[3] from float3, makes it easy to use in loops
  float x[3];
  x[0] = index.x;
  x[1] = index.y;
  x[2] = index.z;
  uint width = spline_order + 1;

  // spline_order must be between 0 and 5.
  if( spline_order == 3 )
  {
    for( uint n = 0; n < 3; n++ )
    {
      uint ao0 = get_array_offset( 0, n, width );
      uint ao1 = get_array_offset( 1, n, width );
      uint ao2 = get_array_offset( 2, n, width );
      uint ao3 = get_array_offset( 3, n, width );

      w = x[n] - (float)evaluate_index[ ao1 ];
      weights[ ao3 ] = (1.0 / 6.0) * w * w * w;
      weights[ ao0 ] = (1.0 / 6.0) + 0.5 * w * ( w - 1.0 ) - weights[ ao3 ];
      weights[ ao2 ] = w + weights[ ao0 ] - 2.0 * weights[ ao3 ];
      weights[ ao1 ] = 1.0 - weights[ ao0 ] - weights[ ao2 ] - weights[ ao3 ];
    }
  }
  else if( spline_order == 0 )
  {
    for( uint n = 0; n < 3; n++ )
    {
      uint ao0 = get_array_offset( 0, n, width );

      weights[ ao0 ] = 1;
    }
  }
  else if( spline_order == 1 )
  {
    for( uint n = 0; n < 3; n++ )
    {
      uint ao0 = get_array_offset( 0, n, width );
      uint ao1 = get_array_offset( 1, n, width );

      w = x[n] - (float)evaluate_index[ ao0 ];
      weights[ ao1 ] = w;
      weights[ ao0 ] = 1.0 - w;
    }
  }
  else if( spline_order == 2 )
  {
    for( uint n = 0; n < 3; n++ )
    {
      uint ao0 = get_array_offset( 0, n, width );
      uint ao1 = get_array_offset( 1, n, width );
      uint ao2 = get_array_offset( 2, n, width );

      w = x[n] - (float)evaluate_index[ ao1 ];
      weights[ ao1 ] = 0.75 - w * w;
      weights[ ao2 ] = 0.5 * ( w - weights[ ao1 ] + 1.0 );
      weights[ ao0 ] = 1.0 - weights[ ao1 ] - weights[ ao2 ];
    }
  }
  else if( spline_order == 4 )
  {
    for( uint n = 0; n < 3; n++ )
    {
      uint ao0 = get_array_offset( 0, n, width );
      uint ao1 = get_array_offset( 1, n, width );
      uint ao2 = get_array_offset( 2, n, width );
      uint ao3 = get_array_offset( 3, n, width );
      uint ao4 = get_array_offset( 4, n, width );

      w = x[n] - (float)evaluate_index[ ao2 ];
      w2 = w * w;
      t2 = ( 0.5 - w ); t2 *= t2; t2 *= t2;
      weights[ ao0 ] = (1.0 / 24.0) * t2;
      t = (1.0 / 6.0) * w2;
      t0 = w * ( t - 11.0 / 24.0 );
      t1 = 19.0 / 96.0 + w2 * ( 0.25 - t );
      weights[ ao1 ] = t1 + t0;
      weights[ ao3 ] = t1 - t0;
      weights[ ao4 ] = weights[ ao0 ] + t0 + 0.5 * w;
      weights[ ao2 ] = 1.0 - weights[ ao0 ] - weights[ ao1 ] - weights[ ao3 ] - weights[ ao4 ];
    }
  }
  else if( spline_order == 5 )
  {
    for( uint n = 0; n < 3; n++ )
    {
      uint ao0 = get_array_offset( 0, n, width );
      uint ao1 = get_array_offset( 1, n, width );
      uint ao2 = get_array_offset( 2, n, width );
      uint ao3 = get_array_offset( 3, n, width );
      uint ao4 = get_array_offset( 4, n, width );
      uint ao5 = get_array_offset( 5, n, width );

      w = x[n] - (float)evaluate_index[ ao2 ];
      w2 = w * w;
      weights[ ao5 ] = ( 1.0 / 120.0 ) * w * w2 * w2;
      w2 -= w;
      w4 = w2 * w2;
      w -= 0.5;
      t = w2 * ( w2 - 3.0 );
      weights[ ao0 ] = ( 1.0 / 24.0 ) * ( 1.0 / 5.0 + w2 + w4 ) - weights[ ao5 ];
      t0 = ( 1.0 / 24.0 ) * ( w2 * ( w2 - 5.0 ) + 46.0 / 5.0 );
      t1 = ( -1.0 / 12.0 ) * w * ( t + 4.0 );
      weights[ ao2 ] = t0 + t1;
      weights[ ao3 ] = t0 - t1;
      t0 = ( 1.0 / 16.0 ) * ( 9.0 / 5.0 - t );
      t1 = ( 1.0 / 24.0 ) * w * ( w4 - w2 - 5.0 );
      weights[ ao1 ] = t0 + t1;
      weights[ ao4 ] = t0 - t1;
    }
  }
}
#endif // DIM_3

//------------------------------------------------------------------------------
#ifdef DIM_1
void apply_mirror_boundary_conditions_1d(long *evaluate_index,
                                         __constant GPUImageBase1D *coefficients_image,
                                         __constant GPUImageFunction1D *image_function,
                                         const uint spline_order)
{
  long start_index = image_function->start_index;
  long end_index = image_function->end_index;
  uint data_length = coefficients_image->size;
  uint width = spline_order + 1;

  if( data_length == 1 )
  {
    for( uint k = 0; k <= spline_order; k++ )
    {
      evaluate_index[ k ] = 0;
    }
  }
  else
  {
    for( uint k = 0; k <= spline_order; k++ )
    {
      if( evaluate_index[ k ] < start_index )
      {
        evaluate_index[ k ] = start_index + (start_index - evaluate_index[ k ]);
      }
      else if( evaluate_index[ k ] >= end_index )
      {
        evaluate_index[ k ] = end_index - (evaluate_index[ k ] - end_index);
      }
    }
  }
}
#endif // DIM_1

//------------------------------------------------------------------------------
#ifdef DIM_2
void apply_mirror_boundary_conditions_2d(long *evaluate_index,
                                         __constant GPUImageBase2D *coefficients_image,
                                         __constant GPUImageFunction2D *image_function,
                                         const uint spline_order)
{
  long start_index[2];
  start_index[0] = image_function->start_index.x;
  start_index[1] = image_function->start_index.y;

  long end_index[2];
  end_index[0] = image_function->end_index.x;
  end_index[1] = image_function->end_index.y;

  uint data_length[2];
  data_length[0] = coefficients_image->size.x;
  data_length[1] = coefficients_image->size.y;

  uint width = spline_order + 1;

  for( uint n = 0; n < 2; n++ )
  {
    if( data_length[n] == 1 )
    {
      for( uint k = 0; k <= spline_order; k++ )
      {
        evaluate_index[ get_array_offset(k, n, width) ] = 0;
      }
    }
    else
    {
      for( uint k = 0; k <= spline_order; k++ )
      {
        uint ao = get_array_offset( k, n, width );
        if( evaluate_index[ ao ] < start_index[ n ] )
        {
          evaluate_index[ ao ] = start_index[ n ] + (start_index[ n ] - evaluate_index[ ao ] );
        }
        else if( evaluate_index[ ao ] >= end_index[ n ] )
        {
          evaluate_index[ ao ] = end_index[ n ] - (evaluate_index[ ao ] - end_index[ n ] );
        }
      }
    }
  }
}
#endif // DIM_2

//------------------------------------------------------------------------------
#ifdef DIM_3
void apply_mirror_boundary_conditions_3d(long *evaluate_index,
                                         __constant GPUImageBase3D *coefficients_image,
                                         __constant GPUImageFunction3D *image_function,
                                         const uint spline_order)
{
  long start_index[3];
  start_index[0] = image_function->start_index.x;
  start_index[1] = image_function->start_index.y;
  start_index[2] = image_function->start_index.z;

  long end_index[3];
  end_index[0] = image_function->end_index.x;
  end_index[1] = image_function->end_index.y;
  end_index[2] = image_function->end_index.z;

  uint data_length[3];
  data_length[0] = coefficients_image->size.x;
  data_length[1] = coefficients_image->size.y;
  data_length[2] = coefficients_image->size.z;

  uint width = spline_order + 1;

  for( uint n = 0; n < 3; n++ )
  {
    // MS: is this check needed?
    if( data_length[ n ] == 1 )
    {
      for( uint k = 0; k <= spline_order; k++ )
      {
        evaluate_index[ get_array_offset( k, n, width ) ] = 0;
      }
    }
    else
    {
      for( uint k = 0; k <= spline_order; k++ )
      {
        uint ao = get_array_offset( k, n, width );
        if( evaluate_index[ ao ] < start_index[ n ] )
        {
          evaluate_index[ ao ] = start_index[ n ] + (start_index[ n ] - evaluate_index[ ao ] );
        }
        else if( evaluate_index[ ao ] >= end_index[ n ] )
        {
          evaluate_index[ ao ] = end_index[ n ] - (evaluate_index[ ao ] - end_index[ n ] );
        }
      }
    }
  }
}
#endif // DIM_3

//------------------------------------------------------------------------------
#ifdef DIM_1
float bspline_evaluate_at_continuous_index_1d(const float index,
                                              __global const INPIXELTYPE *in,
                                              __constant GPUImageBase1D *in_image,
                                              __constant GPUImageFunction1D *image_function,
                                              __global const INTERPOLATOR_PRECISION_TYPE *coefficients,
                                              __constant GPUImageBase1D *coefficients_image)
{
  // variable length array declaration not allowed in OpenCL,
  // therefore we are using #define GPUBSplineOrder num
  long  evaluate_index[GPUBSplineOrder + 1];
  float weights[GPUBSplineOrder + 1];
  ulong width = (ulong)(GPUBSplineOrder + 1);

  // compute the interpolation indexes
  determine_region_of_support_1d( evaluate_index, index, GPUBSplineOrder );
  // determine weights
  set_interpolation_weights_1d( index, evaluate_index, weights, GPUBSplineOrder );
  // modify evaluateIndex at the boundaries using mirror boundary conditions
  apply_mirror_boundary_conditions_1d( evaluate_index, coefficients_image, image_function, GPUBSplineOrder );

  // define points_to_index
  ulong index_factor;
  uint  points_to_index[GPUMaxNumberInterpolationPoints];
  for( uint p = 0; p < GPUMaxNumberInterpolationPoints; p++ )
  {
    points_to_index[p] = p;
  }

  // perform interpolation
  float interpolated = 0.0;
  uint  coefficient_index;

  // Step through each point in the N-dimensional interpolation cube.
  for( uint p = 0; p < GPUMaxNumberInterpolationPoints; p++ )
  {
    float w = 1.0;
    uint  indx = points_to_index[p];
    w *= weights[indx];
    coefficient_index = evaluate_index[indx];

    uint gidx = coefficient_index;
    interpolated += w * (float)(coefficients[gidx]);
  }

  return interpolated;
}
#endif // DIM_1

//------------------------------------------------------------------------------
#ifdef DIM_2
float bspline_evaluate_at_continuous_index_2d(const float2 index,
                                              __global const INPIXELTYPE *in,
                                              __constant GPUImageBase2D *in_image,
                                              __constant GPUImageFunction2D *image_function,
                                              __global const INTERPOLATOR_PRECISION_TYPE *coefficients,
                                              __constant GPUImageBase2D *coefficients_image)
{
  // variable length array declaration not allowed in OpenCL,
  // therefore we are using #define GPUBSplineOrder num
  long  evaluate_index[2][GPUBSplineOrder + 1];
  float weights[2][GPUBSplineOrder + 1];
  ulong width = (ulong)(GPUBSplineOrder + 1);

  // compute the interpolation indexes
  determine_region_of_support_2d( evaluate_index, index, GPUBSplineOrder );
  // determine weights
  set_interpolation_weights_2d( index, evaluate_index, weights, GPUBSplineOrder );
  // modify evaluateIndex at the boundaries using mirror boundary conditions
  apply_mirror_boundary_conditions_2d( evaluate_index, coefficients_image, image_function, GPUBSplineOrder );

  // define points_to_index
  ulong2 index_factor;
  uint   points_to_index[GPUMaxNumberInterpolationPoints][2];
  for( uint p = 0; p < GPUMaxNumberInterpolationPoints; p++ )
  {
    int    pp = p;
    ulong2 index_factor = (ulong2)(1, width);
    points_to_index[p][1] = pp / index_factor.s1;
    pp = pp % index_factor.s1;
    points_to_index[p][0] = pp / index_factor.s0;
  }

  // perform interpolation
  float interpolated = 0.0;
  uint  coefficient_index[2];

  // Step through each point in the N-dimensional interpolation cube.
  for( uint p = 0; p < GPUMaxNumberInterpolationPoints; p++ )
  {
    float w = 1.0;
    for( uint n = 0; n < 2; n++ )
    {
      uint indx = points_to_index[p][n];
      w *= weights[n][indx];
      coefficient_index[n] = evaluate_index[n][indx];
    }
    uint gidx = mad24( coefficients_image->size.x, coefficient_index[1], coefficient_index[0] );
    interpolated += w * (float)(coefficients[gidx]);
  }

  return interpolated;
}
#endif // DIM_2

//------------------------------------------------------------------------------
#ifdef DIM_3
float bspline_evaluate_at_continuous_index_3d(const float3 index,
                                              __global const INPIXELTYPE *in,
                                              __constant GPUImageBase3D *in_image,
                                              __constant GPUImageFunction3D *image_function,
                                              __global const INTERPOLATOR_PRECISION_TYPE *coefficients,
                                              __constant GPUImageBase3D *coefficients_image)
{
  // variable length array declaration not allowed in OpenCL,
  // therefore we are using #define GPUBSplineOrder num
  long  evaluate_index[3][GPUBSplineOrder + 1];
  float weights[3][GPUBSplineOrder + 1];
  ulong width = (ulong)(GPUBSplineOrder + 1);

  // compute the interpolation indexes
  determine_region_of_support_3d( evaluate_index, index, GPUBSplineOrder );
  // determine weights
  set_interpolation_weights_3d( index, evaluate_index, weights, GPUBSplineOrder );
  // modify evaluateIndex at the boundaries using mirror boundary conditions
  apply_mirror_boundary_conditions_3d( evaluate_index, coefficients_image, image_function, GPUBSplineOrder );

  // define points_to_index
  ulong3 index_factor;
  uint   points_to_index[GPUMaxNumberInterpolationPoints][3];
  for( uint p = 0; p < GPUMaxNumberInterpolationPoints; p++ )
  {
    int    pp = p;
    ulong3 index_factor = (ulong3)(1, width, width * width);

    points_to_index[p][2] = pp / index_factor.s2;
    pp = pp % index_factor.s2;
    points_to_index[p][1] = pp / index_factor.s1;
    pp = pp % index_factor.s1;
    points_to_index[p][0] = pp / index_factor.s0;
  }

  // perform interpolation
  float interpolated = 0.0;
  uint  coefficient_index[3];

  // Step through each point in the N-dimensional interpolation cube.
  for( uint p = 0; p < GPUMaxNumberInterpolationPoints; p++ )
  {
    float w = 1.0;
    for( uint n = 0; n < 3; n++ )
    {
      uint indx = points_to_index[p][n];
      w *= weights[n][indx];
      coefficient_index[n] = evaluate_index[n][indx];
    }
    uint gidx
      = mad24( coefficients_image->size.x,
          mad24( coefficient_index[2], coefficients_image->size.y, coefficient_index[1] ),
          coefficient_index[0] );
    interpolated += w * (float)(coefficients[gidx]);
  }

  return interpolated;
}
#endif // DIM_3


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
// OpenCL implementation of itk::BSplineInterpolateImageFunction

//------------------------------------------------------------------------------
// get offset in array
uint get_array_offset( const uint x, const uint y, const uint width )
{
  return mad24( width, y, x );
}

//------------------------------------------------------------------------------
#ifdef DIM_1
int apply_mirror_boundary_conditions_1d(
  int evaluate_index,
  const int start_index_image,
  const int end_index_image )
{
  if( evaluate_index < start_index_image )
  {
    evaluate_index = 2 * start_index_image - evaluate_index;
  }
  else if( evaluate_index > end_index_image )
  {
    evaluate_index = 2 * end_index_image - evaluate_index;
  }

  return evaluate_index;
}
#endif // DIM_1

//------------------------------------------------------------------------------
#ifdef DIM_2
int2 apply_mirror_boundary_conditions_2d(
  int2 evaluate_index,
  const int2 start_index_image,
  const int2 end_index_image )
{
  if( evaluate_index.x < start_index_image.x )
  {
    evaluate_index.x = 2 * start_index_image.x - evaluate_index.x;
  }
  else if( evaluate_index.x > end_index_image.x )
  {
    evaluate_index.x = 2 * end_index_image.x - evaluate_index.x;
  }

  if( evaluate_index.y < start_index_image.y )
  {
    evaluate_index.y = 2 * start_index_image.y - evaluate_index.y;
  }
  else if( evaluate_index.y > end_index_image.y )
  {
    evaluate_index.y = 2 * end_index_image.y - evaluate_index.y;
  }

  return evaluate_index;
}
#endif // DIM_2

//------------------------------------------------------------------------------
#ifdef DIM_3
int3 apply_mirror_boundary_conditions_3d(
  int3 evaluate_index,
  const int3 start_index_image,
  const int3 end_index_image )
{
  if( evaluate_index.x < start_index_image.x )
  {
    evaluate_index.x = 2 * start_index_image.x - evaluate_index.x;
  }
  else if( evaluate_index.x > end_index_image.x )
  {
    evaluate_index.x = 2 * end_index_image.x - evaluate_index.x;
  }

  if( evaluate_index.y < start_index_image.y )
  {
    evaluate_index.y = 2 * start_index_image.y - evaluate_index.y;
  }
  else if( evaluate_index.y > end_index_image.y )
  {
    evaluate_index.y = 2 * end_index_image.y - evaluate_index.y;
  }

  if( evaluate_index.z < start_index_image.z )
  {
    evaluate_index.z = 2 * start_index_image.z - evaluate_index.z;
  }
  else if( evaluate_index.z > end_index_image.z )
  {
    evaluate_index.z = 2 * end_index_image.z - evaluate_index.z;
  }

  return evaluate_index;
}
#endif // DIM_3

//------------------------------------------------------------------------------
#ifdef DIM_1
void set_interpolation_weights_1d( const float cindex, float *weights,
  const uint spline_order, const float half_offset, const uint support_size )
{
  // For speed improvements we could make each case a separate function and use
  // function pointers to reference the correct weight order.
  // now for readability use following float per different spline order:
  // float w, w2, w4, t, t0, t1, t2;

  // spline_order must be between 0 and 5.
  if( spline_order == 3 )
  {
    float tmp = floor( cindex + half_offset ) - spline_order / 2 + 1;
    float w = cindex - tmp;
    weights[3] = ( 1.0f / 6.0f ) * pown(w, 3);
    weights[0] = ( 1.0f / 6.0f ) + 0.5f * w * ( w - 1.0f ) - weights[3];
    weights[2] = w + weights[0] - 2.0f * weights[3];
    weights[1] = 1.0f - weights[0] - weights[2] - weights[3];
    return;
  }
  else if( spline_order == 0 )
  {
    weights[0] = 1.0f; // implements nearest neighbor
    return;
  }
  else if( spline_order == 1 )
  {
    float tmp = floor( cindex + half_offset ) - spline_order / 2;
    float w = cindex - tmp;
    weights[1] = w;
    weights[0] = 1.0f - w;
    return;
  }
  else if( spline_order == 2 )
  {
    float tmp = floor( cindex + half_offset ) - spline_order / 2 + 1;
    float w = cindex - tmp;
    weights[1] = 0.75f - pown(w, 2);
    weights[2] = 0.5f * ( w - weights[1] + 1.0f );
    weights[0] = 1.0f - weights[1] - weights[2];
    return;
  }
  else if( spline_order == 4 )
  {
    float tmp = floor( cindex + half_offset ) - spline_order / 2 + 2;
    float w = cindex - tmp;
    float w2 = pown(w, 2);
    float t2 = ( 0.5f - w ); t2 *= t2; t2 *= t2;
    weights[0] = ( 1.0f / 24.0f ) * t2;
    float t = ( 1.0f / 6.0f ) * w2;
    float t0 = w * ( t - 11.0f / 24.0f );
    float t1 = 19.0f / 96.0f + w2 * ( 0.25f - t );
    weights[1] = t1 + t0;
    weights[3] = t1 - t0;
    weights[4] = weights[0] + t0 + 0.5f * w;
    weights[2] = 1.0f - weights[0] - weights[1] - weights[3] - weights[4];
    return;
  }
  else if( spline_order == 5 )
  {
    float tmp = floor( cindex + half_offset ) - spline_order / 2 + 2;
    float w = cindex - tmp;
    float w2 = pown(w, 2);
    weights[5] = ( 1.0f / 120.0f ) * w * pown(w2, 2);
    w2 -= w;
    float w4 = pown(w2, 2);
    w -= 0.5f;
    float t = w2 * ( w2 - 3.0f );
    weights[0] = ( 1.0f / 24.0f ) * ( 1.0f / 5.0f + w2 + w4 ) - weights[5];
    float t0 = ( 1.0f / 24.0f ) * ( w2 * ( w2 - 5.0f ) + 46.0f / 5.0f );
    float t1 = ( -1.0f / 12.0f ) * w * ( t + 4.0f );
    weights[2] = t0 + t1;
    weights[3] = t0 - t1;
    t0 = ( 1.0f / 16.0f ) * ( 9.0f / 5.0f - t );
    t1 = ( 1.0f / 24.0f ) * w * ( w4 - w2 - 5.0f );
    weights[1] = t0 + t1;
    weights[4] = t0 - t1;
    return;
  }
}
#endif // DIM_1

//------------------------------------------------------------------------------
#ifdef DIM_2
void set_interpolation_weights_2d( const float2 cindex, float *weights,
  const uint spline_order, const float half_offset, const uint support_size )
{
  // For speed improvements we could make each case a separate function and use
  // function pointers to reference the correct weight order.
  // now for readability use following float per different spline order:
  // float w, w2, w4, t, t0, t1, t2;

  // create float x[2] from float2, makes it easy to use in loops
  float x[2];
  x[0] = cindex.x;
  x[1] = cindex.y;

  // spline_order must be between 0 and 5.
  if( spline_order == 3 )
  {
    float w, tmp;
    for( uint n = 0; n < 2; ++n )
    {
      uint ao0 = get_array_offset( 0, n, support_size );
      uint ao1 = get_array_offset( 1, n, support_size );
      uint ao2 = get_array_offset( 2, n, support_size );
      uint ao3 = get_array_offset( 3, n, support_size );

      tmp = floor( x[n] + half_offset ) - spline_order / 2 + 1;
      w = x[n] - tmp;
      weights[ao3] = ( 1.0f / 6.0f ) * pown(w, 3);
      weights[ao0] = ( 1.0f / 6.0f ) + 0.5f * w * ( w - 1.0f ) - weights[ao3];
      weights[ao2] = w + weights[ao0] - 2.0f * weights[ao3];
      weights[ao1] = 1.0f - weights[ao0] - weights[ao2] - weights[ao3];
    }
    return;
  }
  else if( spline_order == 0 )
  {
    for( uint n = 0; n < 2; ++n )
    {
      uint ao0 = get_array_offset( 0, n, support_size );
      weights[ao0] = 1.0f;
    }
    return;
  }
  else if( spline_order == 1 )
  {
    float w, tmp;
    for( uint n = 0; n < 2; ++n )
    {
      uint ao0 = get_array_offset( 0, n, support_size );
      uint ao1 = get_array_offset( 1, n, support_size );

      tmp = floor( x[n] + half_offset ) - spline_order / 2;
      w = x[n] - tmp;
      weights[ao1] = w;
      weights[ao0] = 1.0f - w;
    }
    return;
  }
  else if( spline_order == 2 )
  {
    float w, tmp;
    for( uint n = 0; n < 2; ++n )
    {
      uint ao0 = get_array_offset( 0, n, support_size );
      uint ao1 = get_array_offset( 1, n, support_size );
      uint ao2 = get_array_offset( 2, n, support_size );

      tmp = floor( x[n] + half_offset ) - spline_order / 2 + 1;
      w = x[n] - tmp;
      weights[ao1] = 0.75f - pown(w, 2);
      weights[ao2] = 0.5f * ( w - weights[ao1] + 1.0f );
      weights[ao0] = 1.0f - weights[ao1] - weights[ao2];
    }
    return;
  }
  else if( spline_order == 4 )
  {
    float w, w2, t, t0, t1, t2, tmp;
    for( uint n = 0; n < 2; ++n )
    {
      uint ao0 = get_array_offset( 0, n, support_size );
      uint ao1 = get_array_offset( 1, n, support_size );
      uint ao2 = get_array_offset( 2, n, support_size );
      uint ao3 = get_array_offset( 3, n, support_size );
      uint ao4 = get_array_offset( 4, n, support_size );

      tmp = floor( x[n] + half_offset ) - spline_order / 2 + 2;
      w = x[n] - tmp;
      w2 = pown(w, 2);
      t2 = ( 0.5f - w ); t2 *= t2; t2 *= t2;
      weights[ao0] = ( 1.0f / 24.0f ) * t2;
      t = ( 1.0f / 6.0f ) * w2;
      t0 = w * ( t - 11.0f / 24.0f );
      t1 = 19.0f / 96.0f + w2 * ( 0.25f - t );
      weights[ao1] = t1 + t0;
      weights[ao3] = t1 - t0;
      weights[ao4] = weights[ao0] + t0 + 0.5f * w;
      weights[ao2] = 1.0f - weights[ao0] - weights[ao1] - weights[ao3] - weights[ao4];
    }
    return;
  }
  else if( spline_order == 5 )
  {
    float w, w2, w4, t, t0, t1, t2, tmp;
    for( uint n = 0; n < 2; ++n )
    {
      uint ao0 = get_array_offset( 0, n, support_size );
      uint ao1 = get_array_offset( 1, n, support_size );
      uint ao2 = get_array_offset( 2, n, support_size );
      uint ao3 = get_array_offset( 3, n, support_size );
      uint ao4 = get_array_offset( 4, n, support_size );
      uint ao5 = get_array_offset( 5, n, support_size );

      tmp = floor( x[n] + half_offset ) - spline_order / 2 + 2;
      w = x[n] - tmp;
      w2 = pown(w, 2);
      weights[ao5] = ( 1.0f / 120.0f ) * w * pown(w2, 2);
      w2 -= w;
      w4 = pown(w2, 2);
      w -= 0.5f;
      t = w2 * ( w2 - 3.0f );
      weights[ao0] = ( 1.0f / 24.0f ) * ( 1.0f / 5.0f + w2 + w4 ) - weights[ao5];
      t0 = ( 1.0f / 24.0f ) * ( w2 * ( w2 - 5.0f ) + 46.0f / 5.0f );
      t1 = ( -1.0f / 12.0f ) * w * ( t + 4.0f );
      weights[ao2] = t0 + t1;
      weights[ao3] = t0 - t1;
      t0 = ( 1.0f / 16.0f ) * ( 9.0f / 5.0f - t );
      t1 = ( 1.0f / 24.0f ) * w * ( w4 - w2 - 5.0f );
      weights[ao1] = t0 + t1;
      weights[ao4] = t0 - t1;
    }
    return;
  }
}
#endif // DIM_2

//------------------------------------------------------------------------------
#ifdef DIM_3
void set_interpolation_weights_3d( const float3 cindex, float * weights,
  const uint spline_order, const float half_offset, const uint support_size )
{
  // For speed improvements we could make each case a separate function and use
  // function pointers to reference the correct spline order. Left as is for
  // now for readability use following float per different spline order:
  // float w, w2, w4, t, t0, t1, t2;
  
  // create float x[3] from float3, makes it easy to use in loops
  float x[3];
  x[0] = cindex.x;
  x[1] = cindex.y;
  x[2] = cindex.z;

  // spline_order must be between 0 and 5.
  if( spline_order == 3 )
  {
    float w, tmp;
    for( uint n = 0; n < 3; ++n )
    {
      uint ao0 = get_array_offset( 0, n, support_size );
      uint ao1 = get_array_offset( 1, n, support_size );
      uint ao2 = get_array_offset( 2, n, support_size );
      uint ao3 = get_array_offset( 3, n, support_size );

      tmp = floor( x[n] + half_offset ) - spline_order / 2 + 1;
      w = x[n] - tmp;
      weights[ao3] = ( 1.0f / 6.0f ) * pown(w, 3);
      weights[ao0] = ( 1.0f / 6.0f ) + 0.5f * w * ( w - 1.0f ) - weights[ao3];
      weights[ao2] = w + weights[ao0] - 2.0f * weights[ao3];
      weights[ao1] = 1.0f - weights[ao0] - weights[ao2] - weights[ao3];
    }
  }
  else if( spline_order == 0 )
  {
    for( uint n = 0; n < 3; ++n )
    {
      uint ao0 = get_array_offset( 0, n, support_size );

      weights[ao0] = 1.0f;
    }
  }
  else if( spline_order == 1 )
  {
    float w, tmp;
    for( uint n = 0; n < 3; ++n )
    {
      uint ao0 = get_array_offset( 0, n, support_size );
      uint ao1 = get_array_offset( 1, n, support_size );

      tmp = floor( x[n] + half_offset ) - spline_order / 2;
      w = x[n] - tmp;
      weights[ao1] = w;
      weights[ao0] = 1.0f - w;
    }
  }
  else if( spline_order == 2 )
  {
    float w, tmp;
    for( uint n = 0; n < 3; ++n )
    {
      uint ao0 = get_array_offset( 0, n, support_size );
      uint ao1 = get_array_offset( 1, n, support_size );
      uint ao2 = get_array_offset( 2, n, support_size );

      tmp = floor( x[n] + half_offset ) - spline_order / 2 + 1;
      w = x[n] - tmp;
      weights[ao1] = 0.75f - pown(w, 2);
      weights[ao2] = 0.5f * ( w - weights[ao1] + 1.0f );
      weights[ao0] = 1.0f - weights[ao1] - weights[ao2];
    }
  }
  else if( spline_order == 4 )
  {
    float w, w2, t, t0, t1, t2, tmp;
    for( uint n = 0; n < 3; ++n )
    {
      uint ao0 = get_array_offset( 0, n, support_size );
      uint ao1 = get_array_offset( 1, n, support_size );
      uint ao2 = get_array_offset( 2, n, support_size );
      uint ao3 = get_array_offset( 3, n, support_size );
      uint ao4 = get_array_offset( 4, n, support_size );

      tmp = floor( x[n] + half_offset ) - spline_order / 2 + 2;
      w = x[n] - tmp;
      w2 = pown(w, 2);
      t2 = ( 0.5f - w ); t2 *= t2; t2 *= t2;
      weights[ao0] = ( 1.0f / 24.0f ) * t2;
      t = ( 1.0f / 6.0f ) * w2;
      t0 = w * ( t - 11.0f / 24.0f );
      t1 = 19.0f / 96.0f + w2 * ( 0.25f - t );
      weights[ao1] = t1 + t0;
      weights[ao3] = t1 - t0;
      weights[ao4] = weights[ao0] + t0 + 0.5f * w;
      weights[ao2] = 1.0f - weights[ao0] - weights[ao1] - weights[ao3] - weights[ao4];
    }
  }
  else if( spline_order == 5 )
  {
    float w, w2, w4, t, t0, t1, t2, tmp;
    for( uint n = 0; n < 3; ++n )
    {
      uint ao0 = get_array_offset( 0, n, support_size );
      uint ao1 = get_array_offset( 1, n, support_size );
      uint ao2 = get_array_offset( 2, n, support_size );
      uint ao3 = get_array_offset( 3, n, support_size );
      uint ao4 = get_array_offset( 4, n, support_size );
      uint ao5 = get_array_offset( 5, n, support_size );

      tmp = floor( x[n] + half_offset ) - spline_order / 2 + 2;
      w = x[n] - tmp;
      w2 = pown(w, 2);
      weights[ao5] = ( 1.0f / 120.0f ) * w * pown(w2, 2);
      w2 -= w;
      w4 = pown(w2, 2);
      w -= 0.5f;
      t = w2 * ( w2 - 3.0f );
      weights[ao0] = ( 1.0f / 24.0f ) * ( 1.0f / 5.0f + w2 + w4 ) - weights[ao5];
      t0 = ( 1.0f / 24.0f ) * ( w2 * ( w2 - 5.0f ) + 46.0f / 5.0f );
      t1 = ( -1.0f / 12.0f ) * w * ( t + 4.0f );
      weights[ao2] = t0 + t1;
      weights[ao3] = t0 - t1;
      t0 = ( 1.0f / 16.0f ) * ( 9.0f / 5.0f - t );
      t1 = ( 1.0f / 24.0f ) * w * ( w4 - w2 - 5.0f );
      weights[ao1] = t0 + t1;
      weights[ao4] = t0 - t1;
    }
  }
}
#endif // DIM_3

//------------------------------------------------------------------------------
#ifdef DIM_1
float bspline_evaluate_at_continuous_index_1d(
  const float cindex,
  const uint spline_order,
  const int start_index_image,
  const int end_index_image,
  __global const float *coefficients,
  const uint coef_image_size ) // coef_image_size == image_size
{
  // Some local variables
  const uint  support_size = spline_order + 1;
  const float half_offset  = spline_order & 1 ? 0.0f : 0.5f;

  // number of weights in 1d computed using formula (spline_order + 1).
  // we allocate the maximum to avoid using if's for all spline orders.
  float weights[6];
  set_interpolation_weights_1d( cindex, weights, spline_order, half_offset, support_size );

  // Determine start index of region of support, uncorrected (no mirroring)
  int start_index_roi;
  start_index_roi = floor( cindex + half_offset ) - spline_order / 2;

  // Variables need for interpolation
  uint gidx;
  int ind;
  float interpolated = 0.0f;
  float w = 0.0f;

  // Calculate maximum number of interpolation points
  const uint maxNumberInterpolationPoints = support_size;

  // Perform interpolation:
  // Step through each point in the N-dimensional interpolation cube.
  for( uint p = 0; p < maxNumberInterpolationPoints; ++p )
  {
    // Get the local index of the point corresponding to this weight
    ind = ( p % support_size );

    // Get the total weight for this point
    w = weights[ ind ];

    // Get the global index of the point corresponding to this weight
    ind += start_index_roi;

    // Apply mirroring boundary conditions
    ind = apply_mirror_boundary_conditions_1d( ind, start_index_image, end_index_image );

    // Get global memory index and update interpolated value
    gidx = ind;

    // Summation
    interpolated += w * coefficients[ gidx ];
  }

  return interpolated;
}
#endif // DIM_1

//------------------------------------------------------------------------------
#ifdef DIM_2
float bspline_evaluate_at_continuous_index_2d(
  const float2 cindex,
  const uint spline_order,
  const int2 start_index_image,
  const int2 end_index_image,
  __global const float *coefficients,
  const uint2 coef_image_size ) // coef_image_size == image_size
{
  // Some local variables
  const uint  support_size = spline_order + 1;
  const float half_offset  = spline_order & 1 ? 0.0f : 0.5f;

  // number of weights in 2d computed using formula 2 * (spline_order + 1).
  // we allocate the maximum to avoid using if's for all spline orders.
  float weights[12];
  set_interpolation_weights_2d( cindex, weights, spline_order, half_offset, support_size );

  // Determine start index of region of support, uncorrected (no mirroring)
  int2 start_index_roi;
  start_index_roi.x = floor( cindex.x + half_offset ) - spline_order / 2;
  start_index_roi.y = floor( cindex.y + half_offset ) - spline_order / 2;

  // Variables need for interpolation
  uint gidx;
  int2 ind;
  float interpolated = 0.0f;
  float w = 0.0f;

  // Calculate maximum number of interpolation points
  const uint maxNumberInterpolationPoints = support_size * support_size;

  // Perform interpolation:
  // Step through each point in the N-dimensional interpolation cube.
  for( uint p = 0; p < maxNumberInterpolationPoints; ++p )
  {
    // Get the local index of the point corresponding to this weight
    ind.x = ( p % support_size );
    ind.y = ( p / support_size ) % support_size;

    // Get the total weight for this point
    w = weights[ ind.x ] * weights[ support_size + ind.y ];

    // Get the global index of the point corresponding to this weight
    ind += start_index_roi;

    // Apply mirroring boundary conditions
    ind = apply_mirror_boundary_conditions_2d( ind, start_index_image, end_index_image );

    // Get global memory index and update interpolated value
    //gidx = mad24( coef_image_size.x, ind.y, ind.x );
    gidx = coef_image_size.x * ind.y + ind.x;

    // Summation
    interpolated += w * coefficients[ gidx ];
  }

  return interpolated;
}
#endif // DIM_2

//------------------------------------------------------------------------------
#ifdef DIM_3
float bspline_evaluate_at_continuous_index_3d(
  const float3 cindex,
  const uint spline_order,
  const int3 start_index_image,
  const int3 end_index_image,
  __global const float *coefficients,
  const uint3 coef_image_size ) // coef_image_size == image_size
{
  // Some local variables
  const uint  support_size = spline_order + 1;
  const float half_offset  = spline_order & 1 ? 0.0f : 0.5f;

  // number of weights in 3d computed using formula 3 * (spline_order + 1).
  // we allocate the maximum to avoid using if's for all spline orders.
  float weights[18];
  set_interpolation_weights_3d( cindex, weights, spline_order, half_offset, support_size );

  // Determine start index of region of support, uncorrected (no mirroring)
  int3 start_index_roi;
  start_index_roi.x = floor( cindex.x + half_offset ) - spline_order / 2;
  start_index_roi.y = floor( cindex.y + half_offset ) - spline_order / 2;
  start_index_roi.z = floor( cindex.z + half_offset ) - spline_order / 2;

  // Variables need for interpolation
  uint gidx;
  int3 ind;
  float interpolated = 0.0f;
  float w = 0.0f;

  // Calculate maximum number of interpolation points
  const uint maxNumberInterpolationPoints = support_size * support_size * support_size;

  // Perform interpolation:
  // Step through each point in the N-dimensional interpolation cube.
  for( uint p = 0; p < maxNumberInterpolationPoints; ++p )
  {
    // Get the local index of the point corresponding to this weight
    ind.x = ( p % support_size );
    ind.y = ( p / support_size ) % support_size;
    ind.z = ( p / support_size / support_size ) % support_size;

    // Get the total weight for this point
    w = weights[ ind.x ] * weights[ support_size + ind.y ] * weights[ 2 * support_size + ind.z ];

    // Get the global index of the point corresponding to this weight
    ind += start_index_roi;

    // Apply mirroring boundary conditions
    ind = apply_mirror_boundary_conditions_3d( ind, start_index_image, end_index_image );

    // Get global memory index and update interpolated value
    //gidx = mad24( coef_image_size.x, mad24( ind.z, coef_image_size.y, ind.y ), ind.x );
    gidx = coef_image_size.x * ( ind.z * coef_image_size.y + ind.y ) + ind.x;

    // Summation
    interpolated += w * coefficients[ gidx ];
  }

  return interpolated;
}
#endif // DIM_3

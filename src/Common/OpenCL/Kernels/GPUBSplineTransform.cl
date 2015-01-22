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
// OpenCL implementation of itk::BSplineTransform

//------------------------------------------------------------------------------
void set_weights( const float cindex,
  const uint spline_order, const long startindex,
  const uint offset, float * weights )
{
  // The code below was taken from:
  //   elastix/src/Common/Transforms/itkBSplineKernelFunction2.h
  // Compared to the ITK version this code assigns to the entire
  // weights vector at once, instead of in a loop, thereby avoiding
  // the excessive use of if statements.
  const float u = cindex - (float)( startindex );

  if( spline_order == 3 )
  {
    const float uu  = pown(u, 2);
    const float uuu = uu * u;

    weights[offset    ] = ( 8.0f - 12.0f * u +  6.0f * uu - uuu ) / 6.0f;
    weights[offset + 1] = ( mad(21.0f, u, -5.0f) - 15.0f * uu + 3.0f * uuu ) / 6.0f;
    weights[offset + 2] = ( 4.0f - 12.0f * u + 12.0f * uu - 3.0f * uuu ) / 6.0f;
    weights[offset + 3] = ( mad(3.0f, u, -1.0f) -  3.0f * uu + uuu ) / 6.0f;

    return;
  }
  else if( spline_order == 0 )
  {
    if( u < 0.5f ) { weights[offset + 0] = 1.0f; }
    else { weights[offset] = 0.5f; }

    return;
  }
  else if( spline_order == 1 )
  {
    weights[offset    ] = 1.0f - u;
    weights[offset + 1] = u - 1.0f;

    return;
  }
  else if( spline_order == 2 )
  {
    const float uu = u * u;

    weights[offset    ] = ( 9.0f  - 12.0f * u + 4.0f * uu ) / 8.0f;
    weights[offset + 1] = mad(2.0f, u, -0.25f) - uu;
    weights[offset + 2] = ( 1.0f  -  4.0f * u + 4.0f * uu ) / 8.0f;

    return;
  }
}

//------------------------------------------------------------------------------
#ifdef DIM_1
bool inside_valid_region_1d( float * cindex, const uint spline_order,
  uint coefficients_image_size_x )
{
  const float min_limit  = 0.5f * (float)( spline_order - 1 );
  const float max_helper = 0.5f * (float)( spline_order - 1 ) + 1.0f;
  const float eps = 1.0e-6f;

  // x
  float max_limit = coefficients_image_size_x - max_helper;
  float cind = (*cindex); float diff = cind - max_limit;
  if( diff > 0.0f && diff < eps ){ (*cindex) -= eps; }
  else if( cind >= max_limit ) return false;
  else if( cind <  min_limit ) return false;

  return true;
}
#endif // DIM_1

//------------------------------------------------------------------------------
#ifdef DIM_2
bool inside_valid_region_2d( float2 * cindex, const uint spline_order,
  uint2 coefficients_image_size )
{
  const float min_limit  = 0.5f * (float)( spline_order - 1 );
  const float max_helper = 0.5f * (float)( spline_order - 1 ) + 1.0f;
  const float eps = 1.0e-6f;

  // x
  float max_limit = coefficients_image_size.x - max_helper;
  float cind = (*cindex).x; float diff = cind - max_limit;
  if( diff > 0.0f && diff < eps ){ (*cindex).x -= eps; }
  else if( cind >= max_limit ) return false;
  else if( cind <  min_limit ) return false;

  // y
  max_limit = coefficients_image_size.y - max_helper;
  cind = (*cindex).y; diff = cind - max_limit;
  if( diff > 0.0f && diff < eps ){ (*cindex).y -= eps; }
  else if( cind >= max_limit ) return false;
  else if( cind <  min_limit ) return false;

  return true;
}
#endif // DIM_2

//------------------------------------------------------------------------------
#ifdef DIM_3
bool inside_valid_region_3d( float3 * cindex, const uint spline_order,
  uint3 coefficients_image_size )
{
  const float min_limit  = 0.5f * (float)( spline_order - 1 );
  const float max_helper = 0.5f * (float)( spline_order - 1 ) + 1.0f;
  const float eps = 1.0e-6f;

  // x
  float max_limit = coefficients_image_size.x - max_helper;
  float cind = (*cindex).x; float diff = cind - max_limit;
  if( diff > 0.0f && diff < eps ){ (*cindex).x -= eps; }
  else if( cind >= max_limit ) return false;
  else if( cind <  min_limit ) return false;

  // y
  max_limit = coefficients_image_size.y - max_helper;
  cind = (*cindex).y; diff = cind - max_limit;
  if( diff > 0.0f && diff < eps ){ (*cindex).y -= eps; }
  else if( cind >= max_limit ) return false;
  else if( cind <  min_limit ) return false;

  // z
  max_limit = coefficients_image_size.z - max_helper;
  cind = (*cindex).z; diff = cind - max_limit;
  if( diff > 0.0f && diff < eps ){ (*cindex).z -= eps; }
  else if( cind >= max_limit ) return false;
  else if( cind <  min_limit ) return false;

  return true;
}
#endif // DIM_3

//------------------------------------------------------------------------------
#ifdef DIM_1
long evaluate_1d( const float index,
  const uint spline_order, const uint support_size,
  const uint number_of_weights, float * weights )
{
  // find the starting index of the support region
  const long startindex = (long)( floor( index - (float)( spline_order - 1 ) / 2.0f ) );

  // number of elements in offset_to_index_table in 1d computed using formula spline_order + 1.
  // we allocate the maximum to avoid using if's for all spline orders.
  ulong offset_to_index_table[4];
  for( uint i = 0; i < support_size; ++i )
  {
    offset_to_index_table[i] = i;
  }

  // number of weights1D in 1d computed using formula spline_order + 1.
  // we allocate the maximum to avoid using if's for all spline orders.
  float weights1D[4];
  set_weights( index, spline_order, startindex, 0, weights1D );

  // compute all possible products of the 1D weights
  for( uint k = 0; k < number_of_weights; ++k )
  {
    weights[k] = weights1D[offset_to_index_table[k]];
  }

  // return start index
  return startindex;
}
#endif // DIM_1

//------------------------------------------------------------------------------
#ifdef DIM_2
long2 evaluate_2d( const float2 cindex,
  const uint spline_order, const uint support_size,
  const uint number_of_weights, float * weights )
{
  // find the starting index of the support region
  long2 startIndex;
  const float tmp = (float)( spline_order - 1 ) / 2.0f;
  startIndex.x = (long)( floor( cindex.x - tmp ) );
  startIndex.y = (long)( floor( cindex.y - tmp ) );

  // number of weights1D in 2d computed using formula 2 * (spline_order + 1).
  // we allocate the maximum to avoid using if's for all spline orders.
  float weights1D[8];
  set_weights( cindex.x, spline_order, startIndex.x, 0,            weights1D );
  set_weights( cindex.y, spline_order, startIndex.y, support_size, weights1D );

  // compute all possible products of the 1D weights
  uint x, y;
  for( uint k = 0; k < number_of_weights; ++k )
  {
    x = k % support_size;
    y = ( k / support_size ) % support_size;
    weights[ k ] = weights1D[ x ] * weights1D[ 1 * support_size + y ];
  }

  // return start index
  return startIndex;
}
#endif // DIM_2

//------------------------------------------------------------------------------
#ifdef DIM_3
long3 evaluate_3d( const float3 cindex,
  const uint spline_order, const uint support_size,
  const uint number_of_weights, float * weights )
{
  // find the starting index of the support region
  long3 startIndex;
  const float tmp = (float)( spline_order - 1 ) / 2.0f;
  startIndex.x = (long)( floor( cindex.x - tmp ) );
  startIndex.y = (long)( floor( cindex.y - tmp ) );
  startIndex.z = (long)( floor( cindex.z - tmp ) );

  // number of weights1D in 3d computed using formula 3 * (spline_order + 1).
  // we allocate the maximum to avoid using if's for all spline orders.
  float weights1D[12];
  set_weights( cindex.x, spline_order, startIndex.x, 0,                weights1D );
  set_weights( cindex.y, spline_order, startIndex.y, support_size,     weights1D );
  set_weights( cindex.z, spline_order, startIndex.z, support_size * 2, weights1D );

  // compute all possible products of the 1D weights
  uint x, y, z;
  for( uint k = 0; k < number_of_weights; ++k )
  {
    x = k % support_size;
    y = ( k / support_size ) % support_size;
    z = ( k / support_size / support_size ) % support_size;
    weights[ k ] = weights1D[ x ] * weights1D[ 1 * support_size + y ] * weights1D[ 2 * support_size + z ];
  }

  // return start index
  return startIndex;
}
#endif // DIM_3

//------------------------------------------------------------------------------
#ifdef DIM_1
float bspline_transform_point_1d( const float point,
  const uint spline_order,
  __constant GPUImageBase1D *coefficients_image,
  __global const float *coefficients )
{
  float tpoint = 0;
  float cindex;

  // \todo:
  // only needs PhysicalPointToIndex, Origin, Size
  // not Direction, IndexToPhysicalPoint, Spacing
  // Memory passing reduces to 8/18 x 100%.
  transform_physical_point_to_continuous_index_1d( point, &cindex, coefficients_image );

  const bool inside = inside_valid_region_1d( &cindex, spline_order, coefficients_image->size );
  if ( !inside )
  {
    tpoint = point;
    return tpoint;
  }

  // support region and coefficient image size, equals B-spline order + 1
  const uint support_size = spline_order + 1;

  // number of weights in 1d computed using formula pow(spline_order + 1, 1).
  // we allocate the maximum to avoid using if's for all spline orders.
  float weights[4];
  const uint number_of_weights = support_size;

  const long support_index = evaluate_1d( cindex, spline_order, support_size,
    number_of_weights, weights );

  const uint support_region = support_index + support_size;

  // multiply weight with coefficient
  uint counter = 0;
  float c, w;
  for ( uint i = (uint)( support_index ); i < support_region; ++i )
  {
    c = coefficients[i];
    w = weights[counter];
    tpoint = mad( c, w, tpoint );
    ++counter;
  }

  tpoint += point;
  return tpoint;
}
#endif // DIM_1

//------------------------------------------------------------------------------
#ifdef DIM_2
float2 bspline_transform_point_2d( const float2 point,
  const uint spline_order,
  __constant GPUImageBase2D *coefficients_image,
  __global const float *coefficients0,
  __global const float *coefficients1 )
{
  float2 tpoint = (float2)( 0, 0 );
  float2 cindex;

  // \todo:
  // only needs PhysicalPointToIndex, Origin, Size
  // not Direction, IndexToPhysicalPoint, Spacing
  // Memory passing reduces to 8/18 x 100%.
  transform_physical_point_to_continuous_index_2d( point, &cindex, coefficients_image );

  const bool inside = inside_valid_region_2d( &cindex, spline_order, coefficients_image->size );
  if( !inside ) return point;

  // support region and coefficient image size, equals B-spline order + 1
  const uint support_size = spline_order + 1;

  // number of weights in 2d computed using formula pow(spline_order + 1, 2).
  // we allocate the maximum to avoid using if's for all spline orders.
  float weights[16];
  const uint number_of_weights = support_size * support_size;

  const long2 start_index = evaluate_2d( cindex, spline_order, support_size,
    number_of_weights, weights );

  // copy kernel parameter from const memory to local memory for speedup
  const uint coefficients_image_size_x = coefficients_image->size.x;

  // multiply weight with coefficient
  float cx, cy, w;
  uint x, y, gidx;
  for( uint k = 0; k < number_of_weights; ++k )
  {
    // Get the index of the point corresponding to this weight
    // Extra computation to avoid triple loop
    x = start_index.x + ( k % support_size );
    y = start_index.y + ( k / support_size ) % support_size;

    // Get the global index of the coefficient image
    gidx = mad24( coefficients_image_size_x, y, x );

    // Get the coefficients and weight from memory
    cx = coefficients0[ gidx ];
    cy = coefficients1[ gidx ];
    w = weights[ k ];

    // Perform the multiplication and update the output point
    tpoint.x = mad( cx, w, tpoint.x );
    tpoint.y = mad( cy, w, tpoint.y );
  }

  // transformation = deformation + input point
  tpoint += point;
  return tpoint;
}
#endif // DIM_2

//------------------------------------------------------------------------------
#ifdef DIM_3
float3 bspline_transform_point_3d( const float3 point,
  const uint spline_order,
  __constant GPUImageBase3D *coefficients_image, // only partially needed
  __global const float *coefficients0,
  __global const float *coefficients1,
  __global const float *coefficients2 )
{
  float3 tpoint = (float3)( 0, 0, 0 );

  // convert point to continuous index
  float3 cindex = transform_physical_point_to_continuous_index_3d( point,
    coefficients_image->physical_point_to_index, coefficients_image->origin );

  // check if inside
  const bool inside = inside_valid_region_3d( &cindex, spline_order, coefficients_image->size );
  if( !inside ) return point;

  // support region and coefficient image size, equals B-spline order + 1
  const uint support_size = spline_order + 1;

  // number of weights in 3d computed using formula pow(spline_order + 1, 3).
  // we allocate the maximum to avoid using if's for all spline orders.
  float weights[64];
  const uint number_of_weights = support_size * support_size * support_size;

  const long3 start_index = evaluate_3d( cindex, spline_order, support_size,
    number_of_weights, weights );

  // copy kernel parameter from const memory to local memory for speedup
  const uint coefficients_image_size_x = coefficients_image->size.x;
  const uint coefficients_image_size_y = coefficients_image->size.y;

  // multiply weight with coefficient
  float cx, cy, cz, w;
  uint x, y, z, gidx;
  for( uint k = 0; k < number_of_weights; ++k )
  {
    // Get the index of the point corresponding to this weight
    // Extra computation to avoid triple loop
    x = start_index.x + ( k % support_size );
    y = start_index.y + ( k / support_size ) % support_size;
    z = start_index.z + ( k / support_size / support_size ) % support_size;

    // Get the global index of the coefficient image
    gidx = mad24( coefficients_image_size_x,
      mad24( z, coefficients_image_size_y, y ), x );

    // Get the coefficients and weight from memory
    cx = coefficients0[ gidx ];
    cy = coefficients1[ gidx ];
    cz = coefficients2[ gidx ];
    w = weights[ k ];

    // Perform the multiplication and update the output point
    tpoint.x = mad( cx, w, tpoint.x );
    tpoint.y = mad( cy, w, tpoint.y );
    tpoint.z = mad( cz, w, tpoint.z );
  }

  // transformation = deformation + input point
  tpoint += point;
  return tpoint;
}
#endif // DIM_3

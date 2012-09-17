/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

// OpenCL implementation of itk::BSplineTransform

//------------------------------------------------------------------------------
// Apple OpenCL 1.0 support function
bool bspline_transform_is_valid(const uint i, const uint j, const uint k, const uint3 size)
{
  /* NOTE: More than three-level nested conditional statements (e.g.,
  if A && B && C..) invalidates command queue during kernel
  execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
  GT). Therefore, we flattened conditional statements. */
  if(i >= size.x){ return false; }
  if(j >= size.y){ return false; }
  if(k >= size.z){ return false; }
  return true;
}

//------------------------------------------------------------------------------
bool is_inside(const uint grid_size, const float ind)
{
  float       index = ind;
  const float min_limit = 0.5 * (float)(GPUBSplineTransformOrder - 1);
  const float max_limit = (float)(grid_size) - 0.5 * (float)(GPUBSplineTransformOrder - 1) - 1.0;

  // originally if( index[j] == max_limit ) has been changed to
  if(fabs(index - max_limit) < 1e-6)
  {
    index -= 1e-6;
  }
  else if(index >= max_limit)
  {
    return false;
  }
  else if(index < min_limit)
  {
    return false;
  }
  return true;
}

//------------------------------------------------------------------------------
float kernel_evaluate(float u)
{
  // third order spline.
  const float absValue = fabs(u);
  const float sqrValue = u * u;

  if(absValue < 1.0f)
  {
    return (4.0f - 6.0f * sqrValue + 3.0f * sqrValue * absValue) / 6.0f;
  }
  else if(absValue < 2.0)
  {
    return (8.0f - 12.0f * absValue + 6.0f * sqrValue - sqrValue * absValue) / 6.0f;
  }
  else
  {
    return 0.0f;
  }
}

//------------------------------------------------------------------------------
void set_weights(const float index, const long startindex,
                 const uint i, float *weights)
{
  float x = index - (float)(startindex);

  for(uint k = 0; k <= GPUBSplineTransformOrder; k++)
  {
    weights[k + i] = kernel_evaluate(x);
    x -= 1.0f;
  }
}

//------------------------------------------------------------------------------
#ifdef DIM_1
bool inside_valid_region_1d(const float ind,
                            __constant GPUImageBase1D *coefficients_image)
{
  if( !is_inside(coefficients_image->size, ind) )
  {
    return false;
  }
  return true;
}

#endif // DIM_1

//------------------------------------------------------------------------------
#ifdef DIM_2
bool inside_valid_region_2d(const float2 ind,
                            __constant GPUImageBase2D *coefficients_image)
{
  if( !is_inside(coefficients_image->size.x, ind.x) )
  {
    return false;
  }
  if( !is_inside(coefficients_image->size.y, ind.y) )
  {
    return false;
  }
  return true;
}

#endif // DIM_2

//------------------------------------------------------------------------------
#ifdef DIM_3
bool inside_valid_region_3d(const float3 ind,
                            __constant GPUImageBase3D *coefficients_image)
{
  if( !is_inside(coefficients_image->size.x, ind.x) )
  {
    return false;
  }
  if( !is_inside(coefficients_image->size.y, ind.y) )
  {
    return false;
  }
  if( !is_inside(coefficients_image->size.z, ind.z) )
  {
    return false;
  }
  return true;
}

#endif // DIM_3

//------------------------------------------------------------------------------
#ifdef DIM_1
long evaluate_1d(const float index, float *weights)
{
  // define offset_to_index_table
  ulong offset_to_index_table[GPUBSplineTransformNumberOfWeights];
  uint  support_size = GPUBSplineTransformOrder + 1;

  for(uint i = 0; i < support_size; i++)
  {
    offset_to_index_table[i] = i;
  }

  // find the starting index of the support region
  long startindex = (long)( floor(index - (float)(GPUBSplineTransformOrder - 1) / 2.0) );

  // compute the weights
  float weights1D[GPUBSplineTransformOrder + 1];
  set_weights(index, startindex, 0, weights1D);

  for(uint k = 0; k < GPUBSplineTransformNumberOfWeights; k++)
  {
    weights[k] = 1.0;
    weights[k] *= weights1D[offset_to_index_table[k]];
  }

  // return start index
  return startindex;
}

#endif // DIM_1

//------------------------------------------------------------------------------
#ifdef DIM_2
long2 evaluate_2d(const float2 index, float *weights)
{
  // define offset_to_index_table
  ulong offset_to_index_table[GPUBSplineTransformNumberOfWeights][2];
  uint  support_size = GPUBSplineTransformOrder + 1;
  ulong counter = 0;

  for(uint j = 0; j < support_size; j++)
  {
    for(uint i = 0; i < support_size; i++)
    {
      offset_to_index_table[counter][0] = i;
      offset_to_index_table[counter][1] = j;
      ++counter;
    }
  }

  // find the starting index of the support region
  long2 startindex;
  startindex.x = (long)( floor(index.x - (float)(GPUBSplineTransformOrder - 1) / 2.0) );
  startindex.y = (long)( floor(index.y - (float)(GPUBSplineTransformOrder - 1) / 2.0) );

  // compute the weights
  float weights1D[2][GPUBSplineTransformOrder + 1];
  set_weights(index.x, startindex.x, 0, weights1D);
  set_weights(index.y, startindex.y, GPUBSplineTransformOrder + 1, weights1D);

  for(uint k = 0; k < GPUBSplineTransformNumberOfWeights; k++)
  {
    weights[k] = 1.0;
    weights[k] *= weights1D[0][offset_to_index_table[k][0]];
    weights[k] *= weights1D[1][offset_to_index_table[k][1]];
  }

  // return start index
  return startindex;
}

#endif // DIM_2

//------------------------------------------------------------------------------
#ifdef DIM_3
long3 evaluate_3d(const float3 index, float *weights)
{
  // define offset_to_index_table
  ulong      offset_to_index_table[GPUBSplineTransformNumberOfWeights][3];
  const uint support_size = GPUBSplineTransformOrder + 1;
  ulong      counter = 0;

  for(uint k = 0; k < support_size; k++)
  {
    for(uint j = 0; j < support_size; j++)
    {
      for(uint i = 0; i < support_size; i++)
      {
        offset_to_index_table[counter][0] = i;
        offset_to_index_table[counter][1] = j;
        offset_to_index_table[counter][2] = k;
        ++counter;
      }
    }
  }

  // find the starting index of the support region
  long3 startindex;
  startindex.x = (long)( floor(index.x - (float)(GPUBSplineTransformOrder - 1) / 2.0) );
  startindex.y = (long)( floor(index.y - (float)(GPUBSplineTransformOrder - 1) / 2.0) );
  startindex.z = (long)( floor(index.z - (float)(GPUBSplineTransformOrder - 1) / 2.0) );

  // compute the weights
  float weights1D[3][GPUBSplineTransformOrder + 1];
  set_weights(index.x, startindex.x, 0, weights1D);
  set_weights(index.y, startindex.y, GPUBSplineTransformOrder + 1, weights1D);
  set_weights(index.z, startindex.z, (GPUBSplineTransformOrder + 1) * 2, weights1D);

  for(uint k = 0; k < GPUBSplineTransformNumberOfWeights; k++)
  {
    weights[k] = 1.0;
    weights[k] *= weights1D[0][offset_to_index_table[k][0]];
    weights[k] *= weights1D[1][offset_to_index_table[k][1]];
    weights[k] *= weights1D[2][offset_to_index_table[k][2]];
  }

  // return start index
  return startindex;
}

#endif // DIM_3

//------------------------------------------------------------------------------
#ifdef DIM_1
float bspline_transform_point_1d(const float point,
                                 __global const INTERPOLATOR_PRECISION_TYPE *coefficients,
                                 __constant GPUImageBase1D *coefficients_image)
{
  float tpoint = 0;
  float index;

  transform_physical_point_to_continuous_index_1d(point, &index, coefficients_image);

  bool inside = inside_valid_region_1d(index, coefficients_image);
  if(!inside)
  {
    tpoint = point;
    return tpoint;
  }

  // evaluate
  float weights[GPUBSplineTransformNumberOfWeights];
  long  support_index = evaluate_1d(index, weights);
  uint  support_size = (uint)(GPUBSplineTransformOrder + 1);
  uint  support_region = support_index + support_size;

  // multiply weight with coefficient
  ulong counter = 0;
  for(uint i = (uint)(support_index); i < support_region; i++)
  {
    if(i < coefficients_image->size)
    {
      uint gidx = i;

      float c = coefficients[gidx];
      tpoint = mad(c, weights[counter], tpoint);

      ++counter;
    }
  }

  tpoint += point;
  return tpoint;
}

#endif // DIM_1

//------------------------------------------------------------------------------
#ifdef DIM_2
float2 bspline_transform_point_2d(const float2 point,
                                  __global const INTERPOLATOR_PRECISION_TYPE *coefficients0,
                                  __constant GPUImageBase2D *coefficients_image0,
                                  __global const INTERPOLATOR_PRECISION_TYPE *coefficients1,
                                  __constant GPUImageBase2D *coefficients_image1)
{
  float2 tpoint = (float2)(0, 0);
  float2 index;

  transform_physical_point_to_continuous_index_2d(point, &index, coefficients_image0);

  bool inside = inside_valid_region_2d(index, coefficients_image0);
  if(!inside)
  {
    tpoint = point;
    return tpoint;
  }

  // evaluate
  float weights[GPUBSplineTransformNumberOfWeights];
  long2 support_index = evaluate_2d(index, weights);
  uint  support_size = (uint)(GPUBSplineTransformOrder + 1);
  uint2 support_region;
  support_region.x = support_index.x + support_size;
  support_region.y = support_index.y + support_size;

  // multiply weight with coefficient
  ulong counter = 0;
  for(uint j = (uint)(support_index.y); j < support_region.y; j++)
  {
    for(uint i = (uint)(support_index.x); i < support_region.x; i++)
    {
      if(i < coefficients_image0->size.x && j < coefficients_image0->size.y)
      {
        uint gidx = mad24(coefficients_image0->size.x, j, i);

        float c0 = coefficients0[gidx];
        float c1 = coefficients1[gidx];

        tpoint.x = mad(c0, weights[counter], tpoint.x);
        tpoint.y = mad(c1, weights[counter], tpoint.y);

        ++counter;
      }
    }
  }

  tpoint += point;
  return tpoint;
}

#endif // DIM_2

//------------------------------------------------------------------------------
#ifdef DIM_3
float3 bspline_transform_point_3d(const float3 point,
                                  __global const INTERPOLATOR_PRECISION_TYPE *coefficients0,
                                  __constant GPUImageBase3D *coefficients_image0,
                                  __global const INTERPOLATOR_PRECISION_TYPE *coefficients1,
                                  __constant GPUImageBase3D *coefficients_image1,
                                  __global const INTERPOLATOR_PRECISION_TYPE *coefficients2,
                                  __constant GPUImageBase3D *coefficients_image2)
{
  float3 tpoint = (float3)(0, 0, 0);
  float3 index;

  transform_physical_point_to_continuous_index_3d(point, &index, coefficients_image0);

  bool inside = inside_valid_region_3d(index, coefficients_image0);
  if(!inside)
  {
    tpoint = point;
    return tpoint;
  }

  // evaluate
  float weights[GPUBSplineTransformNumberOfWeights];
  long3 support_index = evaluate_3d(index, weights);
  uint  support_size = (uint)(GPUBSplineTransformOrder + 1);
  uint3 support_region;
  support_region.x = support_index.x + support_size;
  support_region.y = support_index.y + support_size;
  support_region.z = support_index.z + support_size;

  // multiply weight with coefficient
  ulong counter = 0;
  for(uint k = (uint)(support_index.z); k < support_region.z; k++)
  {
    for(uint j = (uint)(support_index.y); j < support_region.y; j++)
    {
      for(uint i = (uint)(support_index.x); i < support_region.x; i++)
      {
        if( bspline_transform_is_valid(i, j, k, coefficients_image0->size) )
        {
          uint gidx = mad24(coefficients_image0->size.x, mad24(k, coefficients_image0->size.y, j), i);

          float c0 = coefficients0[gidx];
          float c1 = coefficients1[gidx];
          float c2 = coefficients2[gidx];

          tpoint.x = mad(c0, weights[counter], tpoint.x);
          tpoint.y = mad(c1, weights[counter], tpoint.y);
          tpoint.z = mad(c2, weights[counter], tpoint.z);

          ++counter;
        }
      }
    }
  }

  tpoint += point;
  return tpoint;
}

#endif // DIM_3

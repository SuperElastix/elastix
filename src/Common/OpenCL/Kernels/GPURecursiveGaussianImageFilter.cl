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
// OpenCL implementation of itk::NearestNeighborInterpolateImageFunction

#define _ELASTIX_USE_OPENCL_OPTIMIZATIONS 0

//------------------------------------------------------------------------------
// Exact copy of FilterDataArray from RecursiveSeparableImageFilter
void filter_data_array( BUFFPIXELTYPE *outs,
                        const BUFFPIXELTYPE *data,
                        BUFFPIXELTYPE *scratch,
                        const unsigned int ln,
                        const float4 N, const float4 D, const float4 M,
                        const float4 BN, const float4 BM )
{
  /**
  * Causal direction pass
  */
  // this value is assumed to exist from the border to infinity.
  const BUFFPIXELTYPE outV1 = data[0];

  /**
  * Initialize borders
  */
  scratch[0] = ( outV1   * N.x +   outV1 * N.y + outV1   * N.z + outV1 * N.w );
  scratch[1] = ( data[1] * N.x +   outV1 * N.y + outV1   * N.z + outV1 * N.w );
  scratch[2] = ( data[2] * N.x + data[1] * N.y + outV1   * N.z + outV1 * N.w );
  scratch[3] = ( data[3] * N.x + data[2] * N.y + data[1] * N.z + outV1 * N.w );

  // note that the outV1 value is multiplied by the Boundary coefficients m_BNi
  scratch[0] -= ( outV1      * BN.x + outV1      * BN.y + outV1      * BN.z + outV1 * BN.w );
  scratch[1] -= ( scratch[0] * D.x  + outV1      * BN.y + outV1      * BN.z + outV1 * BN.w );
  scratch[2] -= ( scratch[1] * D.x  + scratch[0] * D.y  + outV1      * BN.z + outV1 * BN.w );
  scratch[3] -= ( scratch[2] * D.x  + scratch[1] * D.y  + scratch[0] * D.z  + outV1 * BN.w );

  /**
  * Recursively filter the rest
  */
  float4 data_small, scratch_small;
  for ( uint i = 4; i < ln; ++i )
  {
#if _ELASTIX_USE_OPENCL_OPTIMIZATIONS
    data_small    = (float4)( data[i], data[i - 1], data[i - 2], data[i - 3] );
    scratch_small = (float4)( scratch[i - 1], scratch[i - 2], scratch[i - 3], scratch[i - 4] );
    scratch[i]  = dot( data_small, N );
    scratch[i] -= dot( scratch_small, D );
#else
    scratch[i]  = data[i]        * N.x + data[i - 1]    * N.y + data[i - 2]    * N.z + data[i - 3]    * N.w;
    scratch[i] -= scratch[i - 1] * D.x + scratch[i - 2] * D.y + scratch[i - 3] * D.z + scratch[i - 4] * D.w;
#endif
  }

  /**
  * Store the causal result
  */
  for ( uint i = 0; i < ln; ++i )
  {
    outs[i] = scratch[i];
  }

  /**
  * AntiCausal direction pass
  */
  // this value is assumed to exist from the border to infinity.
  const BUFFPIXELTYPE outV2 = data[ln - 1];

  /**
  * Initialize borders
  */
  scratch[ln - 1] = ( outV2        * M.x + outV2        * M.y + outV2        * M.z + outV2 * M.w );
  scratch[ln - 2] = ( data[ln - 1] * M.x + outV2        * M.y + outV2        * M.z + outV2 * M.w );
  scratch[ln - 3] = ( data[ln - 2] * M.x + data[ln - 1] * M.y + outV2        * M.z + outV2 * M.w );
  scratch[ln - 4] = ( data[ln - 3] * M.x + data[ln - 2] * M.y + data[ln - 1] * M.z + outV2 * M.w );

  // note that the outV2value is multiplied by the Boundary coefficients m_BMi
  scratch[ln - 1] -= ( outV2           * BM.x + outV2           * BM.y + outV2           * BM.z + outV2 * BM.w );
  scratch[ln - 2] -= ( scratch[ln - 1] * D.x  + outV2           * BM.y + outV2           * BM.z + outV2 * BM.w );
  scratch[ln - 3] -= ( scratch[ln - 2] * D.x  + scratch[ln - 1] * D.y  + outV2           * BM.z + outV2 * BM.w );
  scratch[ln - 4] -= ( scratch[ln - 3] * D.x  + scratch[ln - 2] * D.y  + scratch[ln - 1] * D.z  + outV2 * BM.w );

  /**
  * Recursively filter the rest
  */
  for ( unsigned int i = ln - 4; i > 0; i-- )
  {
#if _ELASTIX_USE_OPENCL_OPTIMIZATIONS
    data_small    = (float4)( data[i], data[i + 1], data[i + 2], data[i + 3] );
    scratch_small = (float4)( scratch[i], scratch[i + 1], scratch[i + 2], scratch[i + 3] );
    scratch[i - 1]  = dot( data_small, M );
    scratch[i - 1] -= dot( scratch_small, D );
#else
    scratch[i - 1] = data[i]     * M.x + data[i + 1]    * M.y + data[i + 2]    * M.z + data[i + 3]    * M.w;
    scratch[i - 1] -= scratch[i] * D.x + scratch[i + 1] * D.y + scratch[i + 2] * D.z + scratch[i + 3] * D.w;
#endif
  }

  /**
  * Roll the antiCausal part into the output
  */
  for ( uint i = 0; i < ln; ++i )
  {
    outs[i] += scratch[i];
  }
}

//------------------------------------------------------------------------------
// Get global memory offset
uint get_image_offset( const uint gix,
                       const uint giy,
                       const uint giz,
                       const uint width, const uint height )
{
  uint gidx = mad24( width, mad24( giz, height, giy ), gix );

  return gidx;
}

//------------------------------------------------------------------------------
#ifdef DIM_1
__kernel void RecursiveGaussianImageFilter( __global const INPIXELTYPE *in,
                                            __global OUTPIXELTYPE *out,
                                            unsigned int ln, int direction,
                                            float4 N, float4 D, float4 M,
                                            float4 BN, float4 BM,
                                            uint width, uint height )
{
  uint index = get_global_id( 0 );

  // Define length
  uint length = 0;

  if ( direction == 0 )
  {
    length = width;
  }
  else if ( direction == 1 )
  {
    length = height;
  }

  if ( index < length )
  {
    // local buffers
    BUFFPIXELTYPE inps[BUFFSIZE];
    BUFFPIXELTYPE outs[BUFFSIZE];
    BUFFPIXELTYPE scratch[BUFFSIZE];

    // fill local input buffer
    uint id = 0;
    uint lidx = 0;

    // Having if() statement inside loop will cost performance penalty.
    // Therefore we are moving for() loop inside the if () statement.
    // The code become less compact, but faster.
    if ( height != 0 )
    {
      if ( direction == 0 )
      {
        for ( uint i = 0; i < length; ++i )
        {
          lidx = get_image_offset( i, 0, index, width, 1 );
          inps[id++] = (BUFFPIXELTYPE)( in[lidx] );
        }
      }
      else if ( direction == 1 )
      {
        for ( uint i = 0; i < length; ++i )
        {
          lidx = get_image_offset( index, 0, i, width, 1 );
          inps[id++] = (BUFFPIXELTYPE)( in[lidx] );
        }
      }
    }
    else
    {
      for ( uint i = 0; i < length; ++i )
      {
        inps[id++] = (BUFFPIXELTYPE)( in[i] );
      }
    }

    // Apply the recursive Filter to an array of data. This method is called
    // for each line of the volume. Parameter "scratch" is a scratch
    // area used for internal computations that is the same size as the
    // parameters "outs" and "data".
    filter_data_array( outs, inps, scratch, ln, N, D, M, BN, BM );

    // copy to output
    id = 0;

    // Having if() statement inside loop will cost performance penalty.
    // Therefore we are moving for() loop inside the if () statement.
    // The code become less compact, but faster.
    if ( height != 0 )
    {
      if ( direction == 0 )
      {
        for ( uint i = 0; i < length; ++i )
        {
          lidx = get_image_offset( i, 0, index, width, 1 );
          out[lidx] = (OUTPIXELTYPE)( outs[id++] );
        }
      }
      else if ( direction == 1 )
      {
        for ( uint i = 0; i < length; ++i )
        {
          lidx = get_image_offset( index, 0, i, width, 1 );
          out[lidx] = (OUTPIXELTYPE)( outs[id++] );
        }
      }
    }
    else
    {
      for ( uint i = 0; i < length; ++i )
      {
        out[i] = (OUTPIXELTYPE)( outs[id++] );
      }
    }
  }
}

#endif

//------------------------------------------------------------------------------
#ifdef DIM_2
__kernel void RecursiveGaussianImageFilter( __global const INPIXELTYPE *in,
                                            __global OUTPIXELTYPE *out,
                                            unsigned int ln, int direction,
                                            float4 N, float4 D, float4 M,
                                            float4 BN, float4 BM,
                                            uint width, uint height, uint depth )
{
  uint2 index = (uint2)( get_global_id( 0 ), get_global_id( 1 ) );

  // 0 (direction x) : y/z
  // 1 (direction y) : x/z
  // 2 (direction z) : x/y
  uint3 length;

  if ( direction == 0 )
  {
    length.x = height;
    length.y = depth;
    length.z = width;  // looping over
  }
  else if ( direction == 1 )
  {
    length.x = width;
    length.y = depth;
    length.z = height; // looping over
  }
  else if ( direction == 2 )
  {
    length.x = width;
    length.y = height;
    length.z = depth;  // looping over
  }

  if ( index.x < length.x && index.y < length.y )
  {
    // local buffers
    BUFFPIXELTYPE inps[BUFFSIZE];
    BUFFPIXELTYPE outs[BUFFSIZE];
    BUFFPIXELTYPE scratch[BUFFSIZE];

    // fill local input buffer
    uint id = 0;
    uint lidx = 0;

    // Having if() statement inside loop will cost performance penalty.
    // Therefore we are moving for() loop inside the if () statement.
    // The code become less compact, but faster.
    if ( direction == 0 )
    {
      for ( uint i = 0; i < length.z; ++i )
      {
        lidx = get_image_offset( i, index.x, index.y, width, height );
        inps[id++] = (BUFFPIXELTYPE)( in[lidx] );
      }
    }
    else if ( direction == 1 )
    {
      for ( uint i = 0; i < length.z; ++i )
      {
        lidx = get_image_offset( index.x, i, index.y, width, height );
        inps[id++] = (BUFFPIXELTYPE)( in[lidx] );
      }
    }
    else if ( direction == 2 )
    {
      for ( uint i = 0; i < length.z; ++i )
      {
        lidx = get_image_offset( index.x, index.y, i, width, height );
        inps[id++] = (BUFFPIXELTYPE)( in[lidx] );
      }
    }

    // Apply the recursive Filter to an array of data. This method is called
    // for each line of the volume. Parameter "scratch" is a scratch
    // area used for internal computations that is the same size as the
    // parameters "outs" and "data".
    filter_data_array( outs, inps, scratch, ln, N, D, M, BN, BM );

    // copy to output
    id = 0;

    // Having if() statement inside loop will cost performance penalty.
    // Therefore we are moving for() loop inside the if () statement.
    // The code become less compact, but faster.
    if ( direction == 0 )
    {
      for ( uint i = 0; i < length.z; ++i )
      {
        lidx = get_image_offset( i, index.x, index.y, width, height );
        out[lidx] = (OUTPIXELTYPE)( outs[id++] );
      }
    }
    else if ( direction == 1 )
    {
      for ( uint i = 0; i < length.z; ++i )
      {
        lidx = get_image_offset( index.x, i, index.y, width, height );
        out[lidx] = (OUTPIXELTYPE)( outs[id++] );
      }
    }
    else if ( direction == 2 )
    {
      for ( uint i = 0; i < length.z; ++i )
      {
        lidx = get_image_offset( index.x, index.y, i, width, height );
        out[lidx] = (OUTPIXELTYPE)( outs[id++] );
      }
    }
  }
}

#endif

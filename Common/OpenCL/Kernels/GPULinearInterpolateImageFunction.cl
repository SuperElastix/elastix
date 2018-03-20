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
// OpenCL implementation of itk::LinearInterpolateImageFunction

//------------------------------------------------------------------------------
#ifdef DIM_1
float evaluate_at_continuous_index_1d(
  const float cindex,
  __global const INPIXELTYPE *in,
  const uint size,
  const int start_index,
  const int end_index )
{
  long basei = (long)( floor( cindex ) );

  if ( basei < start_index )
  {
    basei = start_index;
  }

  const float distance = cindex - (float)( basei );
  const float val0 = get_pixel_1d( basei, in, size );
  if ( distance <= 0. )
  {
    return val0;
  }
  basei = basei + 1;
  if ( basei > end_index )
  {
    return val0;
  }
  const float val1 = get_pixel_1d( basei, in, size );
  return ( mad( distance, ( val1 - val0 ), val0 ) );
}
#endif // DIM_1

//------------------------------------------------------------------------------
#ifdef DIM_2
float evaluate_at_continuous_index_2d(
  const float2 cindex,
  __global const INPIXELTYPE *in,
  const uint2 size,
  const int2 start_index,
  const int2 end_index )
{
  long2 basei = (long2)( (long)( floor( cindex.x ) ), (long)( floor( cindex.y ) ) );

  if ( basei.x < start_index.x )
  {
    basei.x = start_index.x;
  }
  const float distance0 = cindex.x - (float)( basei.x );
  if ( basei.y < start_index.y )
  {
    basei.y = start_index.y;
  }
  const float distance1 = cindex.y - (float)( basei.y );
  const float val00 = get_pixel_2d( basei, in, size );

  if ( distance0 <= 0. && distance1 <= 0. )
  {
    return val00;
  }
  else if ( distance1 <= 0. ) // if they have the same "y"
  {
    basei.x = basei.x + 1; // then interpolate across "x"
    if ( basei.x > end_index.x )
    {
      return val00;
    }
    const float val10 = get_pixel_2d( basei, in, size );
    return ( mad( distance0, ( val10 - val00 ), val00 ) );
  }
  else if ( distance0 <= 0. ) // if they have the same "x"
  {
    basei.y = basei.y + 1; // then interpolate across "y"
    if ( basei.y > end_index.y )
    {
      return val00;
    }
    const float val01 = get_pixel_2d( basei, in, size );
    return ( mad( distance1, ( val01 - val00 ), val00 ) );
  }
  // fall-through case:
  // interpolate across "xy"
  basei.x = basei.x + 1;
  if ( basei.x > end_index.x ) // interpolate across "y"
  {
    basei.x = basei.x - 1;
    basei.y = basei.y + 1;
    if ( basei.y > end_index.y )
    {
      return val00;
    }
    const float val01 = get_pixel_2d( basei, in, size );
    return ( mad( distance1, ( val01 - val00 ), val00 ) );
  }
  const float val10 = get_pixel_2d( basei, in, size );
  const float valx0 = mad( distance0, ( val10 - val00 ), val00 );

  basei.y = basei.y + 1;
  if ( basei.y > end_index.y ) // interpolate across "x"
  {
    return valx0;
  }
  const float val11 = get_pixel_2d( basei, in, size );
  basei.x = basei.x - 1;
  const float val01 = get_pixel_2d( basei, in, size );
  const float valx1 = mad( distance0, ( val11 - val01 ), val01 );

  return ( mad( distance1, ( valx1 - valx0 ), valx0 ) );
}
#endif // DIM_2

//------------------------------------------------------------------------------
#ifdef DIM_3
float evaluate_at_continuous_index_3d(
  const float3 cindex,
  __global const INPIXELTYPE *in,
  const uint3 size,
  const int3 start_index,
  const int3 end_index )
{
  long3 basei = (long3)( (long)( floor( cindex.x ) ), (long)( floor( cindex.y ) ), (long)( floor( cindex.z ) ) );

  if ( basei.x < start_index.x )
  {
    basei.x = start_index.x;
  }
  const float distance0 = cindex.x - (float)( basei.x );

  if ( basei.y < start_index.y )
  {
    basei.y = start_index.y;
  }
  const float distance1 = cindex.y - (float)( basei.y );

  if ( basei.z < start_index.z )
  {
    basei.z = start_index.z;
  }
  const float distance2 = cindex.z - (float)( basei.z );

  if ( distance0 <= 0. && distance1 <= 0. && distance2 <= 0. )
  {
    return ( ( get_pixel_3d( basei, in, size ) ) );
  }

  const float val000 = get_pixel_3d( basei, in, size );

  if ( distance2 <= 0. )
  {
    if ( distance1 <= 0. ) // interpolate across "x"
    {
      basei.x = basei.x + 1;
      if ( basei.x > end_index.x )
      {
        return val000;
      }
      const float val100 = get_pixel_3d( basei, in, size );
      return ( mad( distance0, ( val100 - val000 ), val000 ) );
    }
    else if ( distance0 <= 0. ) // interpolate across "y"
    {
      basei.y = basei.y + 1;
      if ( basei.y > end_index.y )
      {
        return val000;
      }
      const float val010 = get_pixel_3d( basei, in, size );
      return ( mad( distance1, ( val010 - val000 ), val000 ) );
    }
    else  // interpolate across "xy"
    {
      basei.x = basei.x + 1;
      if ( basei.x > end_index.x ) // interpolate across "y"
      {
        basei.x = basei.x - 1;
        basei.y = basei.y + 1;
        if ( basei.y > end_index.y )
        {
          return val000;
        }
        const float val010 = get_pixel_3d( basei, in, size );
        return ( mad( distance1, ( val010 - val000 ), val000 ) );
      }
      const float val100 = get_pixel_3d( basei, in, size );
      const float valx00 = mad( distance0, ( val100 - val000 ), val000 );
      basei.y = basei.y + 1;
      if ( basei.y > end_index.y ) // interpolate across "x"
      {
        return valx00;
      }
      const float val110 = get_pixel_3d( basei, in, size );
      basei.x = basei.x - 1;
      const float val010 = get_pixel_3d( basei, in, size );
      const float valx10 = mad( distance0, ( val110 - val010 ), val010 );
      return ( mad( distance1, ( valx10 - valx00 ), valx00 ) );
    }
  }
  else
  {
    if ( distance1 <= 0. )
    {
      if ( distance0 <= 0. ) // interpolate across "z"
      {
        basei.z = basei.z + 1;
        if ( basei.z > end_index.z )
        {
          return val000;
        }
        const float val001 = get_pixel_3d( basei, in, size );
        return ( mad( distance2, ( val001 - val000 ), val000 ) );
      }
      else // interpolate across "xz"
      {
        basei.x = basei.x + 1;
        if ( basei.x > end_index.x ) // interpolate across "z"
        {
          basei.x = basei.x - 1;
          basei.z = basei.z + 1;
          if ( basei.z > end_index.z )
          {
            return val000;
          }
          const float val001 = get_pixel_3d( basei, in, size );
          return ( mad( distance2, ( val001 - val000 ), val000 ) );
        }
        const float val100 = get_pixel_3d( basei, in, size );
        const float valx00 = mad( distance0, ( val100 - val000 ), val000 );
        basei.z = basei.z + 1;
        if ( basei.z > end_index.z ) // interpolate across "x"
        {
          return valx00;
        }
        const float val101 = get_pixel_3d( basei, in, size );
        basei.x = basei.x - 1;
        const float val001 = get_pixel_3d( basei, in, size );
        const float valx01 = mad( distance0, ( val101 - val001 ), val001 );
        return ( mad( distance2, ( valx01 - valx00 ), valx00 ) );
      }
    }
    else if ( distance0 <= 0. ) // interpolate across "yz"
    {
      basei.y = basei.y + 1;
      if ( basei.y > end_index.y ) // interpolate across "z"
      {
        basei.y = basei.y - 1;
        basei.z = basei.z + 1;
        if ( basei.z > end_index.z )
        {
          return ( ( val000 ) );
        }
        const float val001 = get_pixel_3d( basei, in, size );
        return ( mad( distance2, ( val001 - val000 ), val000 ) );
      }
      const float val010 = get_pixel_3d( basei, in, size );
      const float val0x0 = mad( distance1, ( val010 - val000 ), val000 );

      basei.z = basei.z + 1;
      if ( basei.z > end_index.z ) // interpolate across "y"
      {
        return val0x0;
      }
      const float val011 = get_pixel_3d( basei, in, size );
      basei.y = basei.y - 1;
      const float val001 = get_pixel_3d( basei, in, size );
      const float val0x1 = mad( distance1, ( val011 - val001 ), val001 );
      return ( mad( distance2, ( val0x1 - val0x0 ), val0x0 ) );
    }
    else // interpolate across "xyz"
    {
      basei.x = basei.x + 1;
      if ( basei.x > end_index.x ) // interpolate across "yz"
      {
        basei.x = basei.x - 1;
        basei.y = basei.y + 1;
        if ( basei.y > end_index.y )  // interpolate across "z"
        {
          basei.y = basei.y - 1;
          basei.z = basei.z + 1;
          if ( basei.z > end_index.z )
          {
            return val000;
          }
          const float val001 = get_pixel_3d( basei, in, size );
          return ( mad( distance2, ( val001 - val000 ), val000 ) );
        }
        const float val010 = get_pixel_3d( basei, in, size );
        const float val0x0 = mad( distance1, ( val010 - val000 ), val000 );
        basei.z = basei.z + 1;
        if ( basei.z > end_index.z ) // interpolate across "y"
        {
          return ( ( val0x0 ) );
        }
        const float val011 = get_pixel_3d( basei, in, size );
        basei.y = basei.y - 1;
        const float val001 = get_pixel_3d( basei, in, size );
        const float val0x1 = mad( distance1, ( val011 - val001 ), val001 );
        return ( mad( distance2, ( val0x1 - val0x0 ), val0x0 ) );
      }
      const float val100 = get_pixel_3d( basei, in, size );
      const float valx00 = mad( distance0, ( val100 - val000 ), val000 );
      basei.y = basei.y + 1;
      if ( basei.y > end_index.y ) // interpolate across "xz"
      {
        basei.y = basei.y - 1;
        basei.z = basei.z + 1;
        if ( basei.z > end_index.z ) // interpolate across "x"
        {
          return ( ( valx00 ) );
        }
        const float val101 = get_pixel_3d( basei, in, size );
        basei.x = basei.x - 1;
        const float val001 = get_pixel_3d( basei, in, size );
        const float valx01 = mad( distance0, ( val101 - val001 ), val001 );
        return ( mad( distance2, ( valx01 - valx00 ), valx00 ) );
      }
      const float val110 = get_pixel_3d( basei, in, size );
      basei.x = basei.x - 1;
      const float val010 = get_pixel_3d( basei, in, size );
      const float valx10 = mad( distance0, ( val110 - val010 ), val010 );
      const float valxx0 = mad( distance1, ( valx10 - valx00 ), valx00 );
      basei.z = basei.z + 1;
      if ( basei.z > end_index.z ) // interpolate across "xy"
      {
        return valxx0;
      }
      const float val011 = get_pixel_3d( basei, in, size );
      basei.x = basei.x + 1;
      const float val111 = get_pixel_3d( basei, in, size );
      basei.y = basei.y - 1;
      const float val101 = get_pixel_3d( basei, in, size );
      basei.x = basei.x - 1;
      const float val001 = get_pixel_3d( basei, in, size );
      const float valx01 = mad( distance0, ( val101 - val001 ), val001 );
      const float valx11 = mad( distance0, ( val111 - val011 ), val011 );
      const float valxx1 = mad( distance1, ( valx11 - valx01 ), valx01 );

      return ( mad( distance2, ( valxx1 - valxx0 ), valxx0 ) );
    }
  }
}
#endif // DIM_3

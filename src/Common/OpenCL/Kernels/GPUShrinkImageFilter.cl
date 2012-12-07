/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
//
// \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
// Department of Radiology, Leiden, The Netherlands
//
// This implementation was taken from elastix (http://elastix.isi.uu.nl/).
//
// \note This work was funded by the Netherlands Organisation for
// Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
//

//------------------------------------------------------------------------------
// Apple OpenCL 1.0 support function
bool is_valid_3d( const uint3 index, const uint3 size )
{
  /* NOTE: More than three-level nested conditional statements (e.g.,
  if A && B && C..) invalidates command queue during kernel
  execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
  GT). Therefore, we flattened conditional statements. */
  if ( index.x >= size.x ) { return false; }
  if ( index.y >= size.y ) { return false; }
  if ( index.z >= size.z ) { return false; }
  return true;
}

//------------------------------------------------------------------------------
OUTPIXELTYPE Functor( const INPIXELTYPE in )
{
  OUTPIXELTYPE out = (OUTPIXELTYPE)in;

  return out;
}

//------------------------------------------------------------------------------
#ifdef DIM_1
__kernel void ShrinkImageFilter( __global const INPIXELTYPE *in,
  __global OUTPIXELTYPE *out,
  uint in_image_size, uint out_image_size,
  uint offset, uint shrinkfactors )
{
  uint index = get_global_id( 0 );

  if ( index < out_image_size )
  {
    uint input_index = mad24( index, shrinkfactors, offset );
    uint out_gidx = index;
    uint in_gidx = input_index;
    out[out_gidx] = Functor( in[in_gidx] );
  }
}

#endif

//------------------------------------------------------------------------------
#ifdef DIM_2
__kernel void ShrinkImageFilter( __global const INPIXELTYPE *in,
  __global OUTPIXELTYPE *out,
  uint2 image_size_in, uint2 image_size_out,
  uint2 offset, uint2 shrinkfactors )
{
  uint2 index_out = (uint2)( get_global_id( 0 ), get_global_id( 1 ) );

  if ( index_out.x < image_size_out.x && index_out.y < image_size_out.y )
  {
    uint2 index_in = mad24( index_out, shrinkfactors, offset );
    uint  gidx_in = mad24( image_size_in.x, index_in.y, index_in.x );
    uint  gidx_out = mad24( image_size_out.x, index_out.y, index_out.x );
    out[gidx_out] = Functor( in[gidx_in] );
  }
}

#endif

//------------------------------------------------------------------------------
#ifdef DIM_3
__kernel void ShrinkImageFilter( __global const INPIXELTYPE *in,
  __global OUTPIXELTYPE *out,
  uint3 image_size_in, uint3 image_size_out,
  uint3 offset, uint3 shrinkfactors )
{
  uint3 index_out = (uint3)( get_global_id( 0 ), get_global_id( 1 ), get_global_id( 2 ) );

  if ( is_valid_3d( index_out, image_size_out ) )
  {
    uint3 index_in = mad24( index_out, shrinkfactors, offset );
    uint  gidx_in = mad24( image_size_in.x, mad24( index_in.z, image_size_in.y, index_in.y ), index_in.x );
    uint  gidx_out = mad24( image_size_out.x, mad24( index_out.z, image_size_out.y, index_out.y ), index_out.x );
    out[gidx_out] = Functor( in[gidx_in] );
  }
}

#endif

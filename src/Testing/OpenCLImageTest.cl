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

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

/* Copy input 2D image to output 2D image */
__kernel void Image2DCopy( __read_only image2d_t input, __write_only image2d_t output )
{
  int2  coord = (int2)( get_global_id( 0 ), get_global_id( 1 ) );
  uint4 temp = read_imageui( input, imageSampler, coord );

  write_imageui( output, coord, temp );
}


/* Copy input 3D image to 2D image */
__kernel void Image3DCopy( __read_only image3d_t input, __write_only image2d_t output )
{
  int2 coord = (int2)( get_global_id( 0 ), get_global_id( 1 ) );

  // Read first slice into lower half
  uint4 temp0 = read_imageui( input, imageSampler, (int4)( coord, 0, 0 ) );

  // Read second slice into upper half
  uint4 temp1 = read_imageui( input, imageSampler, (int4)( (int2)( get_global_id( 0 ), get_global_id( 1 ) - get_global_size( 1 ) / 2 ), 1, 0 ) );

  write_imageui( output, coord, temp0 + temp1 );
}

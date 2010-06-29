/*--------------------------------------------------------------------------*\
Copyright (c) 2008-2010, Danny Ruijters. All rights reserved.
http://www.dannyruijters.nl/cubicinterpolation/
This file is part of CUDA Cubic B-Spline Interpolation (CI).

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
*  Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
*  Neither the name of the copyright holders nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are
those of the authors and should not be interpreted as representing official
policies, either expressed or implied.
\*--------------------------------------------------------------------------*/

#ifndef _CUBIC3D_KERNEL_H_
#define _CUBIC3D_KERNEL_H_

#include "internal/bspline_kernel.cu"

//! Trilinearly interpolated texture lookup, using unnormalized coordinates.
//! This function merely serves as a reference for the tricubic versions.
//! @param tex  3D texture
//! @param coord  unnormalized 3D texture coordinate
template<class T, enum cudaTextureReadMode mode>
__device__ float linearTex3D( texture<T, 3, mode> tex, float3 coord )
{
  return tex3D( tex, coord.x, coord.y, coord.z );
}

//! Tricubic interpolated texture lookup, using unnormalized coordinates.
//! Straight forward implementation, using 64 nearest neighbour lookups.
//! @param tex  3D texture
//! @param coord  unnormalized 3D texture coordinate
template<class T, enum cudaTextureReadMode mode>
__device__ float cubicTex3DSimple( texture<T, 3, mode> tex, float3 coord )
{
  // transform the coordinate from [0,extent] to [-0.5, extent-0.5]
  const float3 coord_grid = coord - 0.5f;
  float3 index = floor(coord_grid);
  const float3 fraction = coord_grid - index;
  index = index + 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]

  float result = 0.0f;
  for ( float z = -1; z < 2.5f; z++ )  //range [-1, 2]
  {
    float bsplineZ = bspline( z - fraction.z );
    float w = index.z + z;
    for ( float y = -1; y < 2.5f; y++ )
    {
      float bsplineYZ = bspline( y - fraction.y ) * bsplineZ;
      float v = index.y + y;
      for ( float x = -1; x < 2.5f; x++ )
      {
        float bsplineXYZ = bspline( x - fraction.x ) * bsplineYZ;
        float u = index.x + x;
        result += bsplineXYZ * tex3D( tex, u, v, w );
      }
    }
  }
  return result;
}

//! Tricubic interpolated texture lookup, using unnormalized coordinates.
//! Fast implementation, using 8 trilinear lookups.
//! @param tex  3D texture
//! @param coord  unnormalized 3D texture coordinate
#define WEIGHTS bspline_weights
#define CUBICTEX3D cubicTex3D
#include "internal/cubicTex3D_kernel.cu"
#undef CUBICTEX3D
#undef WEIGHTS

// Fast tricubic interpolated 1st order derivative texture lookup in x-, y-
// and z-direction, using unnormalized coordinates.
__device__ void bspline_weights_1st_derivative_x( float3 fraction,
  float3& w0, float3& w1, float3& w2, float3& w3 )
{
  float t0, t1, t2, t3;
  bspline_weights_1st_derivative( fraction.x, t0, t1, t2, t3 );
  w0.x = t0; w1.x = t1; w2.x = t2; w3.x = t3;
  bspline_weights( fraction.y, t0, t1, t2, t3 );
  w0.y = t0; w1.y = t1; w2.y = t2; w3.y = t3;
  bspline_weights( fraction.z, t0, t1, t2, t3 );
  w0.z = t0; w1.z = t1; w2.z = t2; w3.z = t3;
}

__device__ void bspline_weights_1st_derivative_y( float3 fraction,
  float3& w0, float3& w1, float3& w2, float3& w3 )
{
  float t0, t1, t2, t3;
  bspline_weights( fraction.x, t0, t1, t2, t3 );
  w0.x = t0; w1.x = t1; w2.x = t2; w3.x = t3;
  bspline_weights_1st_derivative( fraction.y, t0, t1, t2, t3 );
  w0.y = t0; w1.y = t1; w2.y = t2; w3.y = t3;
  bspline_weights( fraction.z, t0, t1, t2, t3 );
  w0.z = t0; w1.z = t1; w2.z = t2; w3.z = t3;
}

__device__ void bspline_weights_1st_derivative_z( float3 fraction,
  float3& w0, float3& w1, float3& w2, float3& w3 )
{
  float t0, t1, t2, t3;
  bspline_weights( fraction.x, t0, t1, t2, t3 );
  w0.x = t0; w1.x = t1; w2.x = t2; w3.x = t3;
  bspline_weights( fraction.y, t0, t1, t2, t3 );
  w0.y = t0; w1.y = t1; w2.y = t2; w3.y = t3;
  bspline_weights_1st_derivative( fraction.z, t0, t1, t2, t3 );
  w0.z = t0; w1.z = t1; w2.z = t2; w3.z = t3;
}

#define WEIGHTS bspline_weights_1st_derivative_x
#define CUBICTEX3D cubicTex3D_1st_derivative_x
#include "internal/cubicTex3D_kernel.cu"
#undef CUBICTEX3D
#undef WEIGHTS

#define WEIGHTS bspline_weights_1st_derivative_y
#define CUBICTEX3D cubicTex3D_1st_derivative_y
#include "internal/cubicTex3D_kernel.cu"
#undef CUBICTEX3D
#undef WEIGHTS

#define WEIGHTS bspline_weights_1st_derivative_z
#define CUBICTEX3D cubicTex3D_1st_derivative_z
#include "internal/cubicTex3D_kernel.cu"
#undef CUBICTEX3D
#undef WEIGHTS

#endif // _CUBIC3D_KERNEL_H_

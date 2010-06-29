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

//! Tricubic interpolated texture lookup, using unnormalized coordinates.
//! Fast implementation, using 8 trilinear lookups.
//! @param tex  3D texture
//! @param coord  unnormalized 3D texture coordinate
template<class floatN, class T, enum cudaTextureReadMode mode>
__device__ floatN CUBICTEX3D(texture<T, 3, mode> tex, float3 coord)
{
  // shift the coordinate from [0,extent] to [-0.5, extent-0.5]
  const float3 coord_grid = coord - 0.5f;
  const float3 index = floor(coord_grid);
  const float3 fraction = coord_grid - index;
  float3 w0, w1, w2, w3;
  WEIGHTS(fraction, w0, w1, w2, w3);

  const float3 g0 = w0 + w1;
  const float3 g1 = w2 + w3;
  const float3 h0 = (w1 / g0) - 0.5f + index;  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
  const float3 h1 = (w3 / g1) + 1.5f + index;  //h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]

  // fetch the eight linear interpolations
  // weighting and fetching is interleaved for performance and stability reasons
  floatN tex000 = tex3D(tex, h0.x, h0.y, h0.z);
  floatN tex100 = tex3D(tex, h1.x, h0.y, h0.z);
  tex000 = g0.x * tex000 + g1.x * tex100;  //weigh along the x-direction
  floatN tex010 = tex3D(tex, h0.x, h1.y, h0.z);
  floatN tex110 = tex3D(tex, h1.x, h1.y, h0.z);
  tex010 = g0.x * tex010 + g1.x * tex110;  //weigh along the x-direction
  tex000 = g0.y * tex000 + g1.y * tex010;  //weigh along the y-direction
  floatN tex001 = tex3D(tex, h0.x, h0.y, h1.z);
  floatN tex101 = tex3D(tex, h1.x, h0.y, h1.z);
  tex001 = g0.x * tex001 + g1.x * tex101;  //weigh along the x-direction
  floatN tex011 = tex3D(tex, h0.x, h1.y, h1.z);
  floatN tex111 = tex3D(tex, h1.x, h1.y, h1.z);
  tex011 = g0.x * tex011 + g1.x * tex111;  //weigh along the x-direction
  tex001 = g0.y * tex001 + g1.y * tex011;  //weigh along the y-direction

  return (g0.z * tex000 + g1.z * tex001);  //weigh along the z-direction
}


// Specializations

// These specializations fill in the floatN and T class types and therefore
// allow the cubicTex3D function to be called without any template arguments,
// thus with any <> brackets.

// 1-dimensional voxels
template<enum cudaTextureReadMode mode> __device__ float CUBICTEX3D(texture<float, 3, mode> tex, float3 coord) {return CUBICTEX3D<float, float, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float CUBICTEX3D(texture<uchar, 3, mode> tex, float3 coord) {return CUBICTEX3D<float, uchar, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float CUBICTEX3D(texture<char, 3, mode> tex, float3 coord) {return CUBICTEX3D<float, char, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float CUBICTEX3D(texture<ushort, 3, mode> tex, float3 coord) {return CUBICTEX3D<float, ushort, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float CUBICTEX3D(texture<short, 3, mode> tex, float3 coord) {return CUBICTEX3D<float, short, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float CUBICTEX3D(texture<uint, 3, mode> tex, float3 coord) {return CUBICTEX3D<float, uint, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float CUBICTEX3D(texture<int, 3, mode> tex, float3 coord) {return CUBICTEX3D<float, int, mode>(tex, coord);}
// 2-dimensional voxels
template<enum cudaTextureReadMode mode> __device__ float2 CUBICTEX3D(texture<float2, 3, mode> tex, float3 coord) {return CUBICTEX3D<float2, float2, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float2 CUBICTEX3D(texture<uchar2, 3, mode> tex, float3 coord) {return CUBICTEX3D<float2, uchar2, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float2 CUBICTEX3D(texture<char2, 3, mode> tex, float3 coord) {return CUBICTEX3D<float2, char2, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float2 CUBICTEX3D(texture<ushort2, 3, mode> tex, float3 coord) {return CUBICTEX3D<float2, ushort2, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float2 CUBICTEX3D(texture<short2, 3, mode> tex, float3 coord) {return CUBICTEX3D<float2, short2, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float2 CUBICTEX3D(texture<uint2, 3, mode> tex, float3 coord) {return CUBICTEX3D<float2, uint2, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float2 CUBICTEX3D(texture<int2, 3, mode> tex, float3 coord) {return CUBICTEX3D<float2, int2, mode>(tex, coord);}
// 3-dimensional voxels
template<enum cudaTextureReadMode mode> __device__ float3 CUBICTEX3D(texture<float3, 3, mode> tex, float3 coord) {return CUBICTEX3D<float3, float3, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float3 CUBICTEX3D(texture<uchar3, 3, mode> tex, float3 coord) {return CUBICTEX3D<float3, uchar3, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float3 CUBICTEX3D(texture<char3, 3, mode> tex, float3 coord) {return CUBICTEX3D<float3, char3, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float3 CUBICTEX3D(texture<ushort3, 3, mode> tex, float3 coord) {return CUBICTEX3D<float3, ushort3, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float3 CUBICTEX3D(texture<short3, 3, mode> tex, float3 coord) {return CUBICTEX3D<float3, short3, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float3 CUBICTEX3D(texture<uint3, 3, mode> tex, float3 coord) {return CUBICTEX3D<float3, uint3, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float3 CUBICTEX3D(texture<int3, 3, mode> tex, float3 coord) {return CUBICTEX3D<float3, int3, mode>(tex, coord);}
// 4-dimensional voxels
template<enum cudaTextureReadMode mode> __device__ float4 CUBICTEX3D(texture<float4, 3, mode> tex, float3 coord) {return CUBICTEX3D<float4, float4, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float4 CUBICTEX3D(texture<uchar4, 3, mode> tex, float3 coord) {return CUBICTEX3D<float4, uchar4, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float4 CUBICTEX3D(texture<char4, 3, mode> tex, float3 coord) {return CUBICTEX3D<float4, char4, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float4 CUBICTEX3D(texture<ushort4, 3, mode> tex, float3 coord) {return CUBICTEX3D<float4, ushort4, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float4 CUBICTEX3D(texture<short4, 3, mode> tex, float3 coord) {return CUBICTEX3D<float4, short4, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float4 CUBICTEX3D(texture<uint4, 3, mode> tex, float3 coord) {return CUBICTEX3D<float4, uint4, mode>(tex, coord);}
template<enum cudaTextureReadMode mode> __device__ float4 CUBICTEX3D(texture<int4, 3, mode> tex, float3 coord) {return CUBICTEX3D<float4, int4, mode>(tex, coord);}

// Function definitions with (float x, float y, float z) instead of (float3 coord)
// 1-dimensional voxels
template<enum cudaTextureReadMode mode> __device__ float CUBICTEX3D(texture<float, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float, float, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float CUBICTEX3D(texture<uchar, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float, uchar, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float CUBICTEX3D(texture<char, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float, char, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float CUBICTEX3D(texture<ushort, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float, ushort, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float CUBICTEX3D(texture<short, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float, short, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float CUBICTEX3D(texture<uint, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float, uint, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float CUBICTEX3D(texture<int, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float, int, mode>(tex, make_float3(x,y,z));}
// 2-dimensional voxels
template<enum cudaTextureReadMode mode> __device__ float2 CUBICTEX3D(texture<float2, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float2, float2, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float2 CUBICTEX3D(texture<uchar2, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float2, uchar2, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float2 CUBICTEX3D(texture<char2, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float2, char2, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float2 CUBICTEX3D(texture<ushort2, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float2, ushort2, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float2 CUBICTEX3D(texture<short2, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float2, short2, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float2 CUBICTEX3D(texture<uint2, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float2, uint2, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float2 CUBICTEX3D(texture<int2, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float2, int2, mode>(tex, make_float3(x,y,z));}
// 3-dimensional voxels
template<enum cudaTextureReadMode mode> __device__ float3 CUBICTEX3D(texture<float3, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float3, float3, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float3 CUBICTEX3D(texture<uchar3, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float3, uchar3, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float3 CUBICTEX3D(texture<char3, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float3, char3, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float3 CUBICTEX3D(texture<ushort3, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float3, ushort3, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float3 CUBICTEX3D(texture<short3, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float3, short3, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float3 CUBICTEX3D(texture<uint3, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float3, uint3, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float3 CUBICTEX3D(texture<int3, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float3, int3, mode>(tex, make_float3(x,y,z));}
// 4-dimensional voxels
template<enum cudaTextureReadMode mode> __device__ float4 CUBICTEX3D(texture<float4, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float4, float4, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float4 CUBICTEX3D(texture<uchar4, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float4, uchar4, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float4 CUBICTEX3D(texture<char4, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float4, char4, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float4 CUBICTEX3D(texture<ushort4, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float4, ushort4, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float4 CUBICTEX3D(texture<short4, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float4, short4, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float4 CUBICTEX3D(texture<uint4, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float4, uint4, mode>(tex, make_float3(x,y,z));}
template<enum cudaTextureReadMode mode> __device__ float4 CUBICTEX3D(texture<int4, 3, mode> tex, float x, float y, float z) {return CUBICTEX3D<float4, int4, mode>(tex, make_float3(x,y,z));}


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

#ifndef _3D_CUBIC_BSPLINE_PREFILTER_H_
#define _3D_CUBIC_BSPLINE_PREFILTER_H_

#include <stdio.h>
#include "internal/cubicPrefilter_kernel.cu"
#include "../cudaInlineFunctions.h"

//--------------------------------------------------------------------------
// Global CUDA procedures
//--------------------------------------------------------------------------
template<class floatN>
__global__ void SamplesToCoefficients3DX(
  floatN* volume, // in-place processing
  uint width,     // width of the volume
  uint height,    // height of the volume
  uint depth )    // depth of the volume
{
  // process lines in x-direction
  const uint y = blockIdx.x * blockDim.x + threadIdx.x;
  const uint z = blockIdx.y * blockDim.y + threadIdx.y;
  const uint startIdx = (z * height + y) * width;

  ConvertToInterpolationCoefficients( volume + startIdx, width, 1 );
}

template<class floatN>
__global__ void SamplesToCoefficients3DY(
  floatN* volume, // in-place processing
  uint width,     // width of the volume
  uint height,    // height of the volume
  uint depth )    // depth of the volume
{
  // process lines in y-direction
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint z = blockIdx.y * blockDim.y + threadIdx.y;
  const uint startIdx = z * height * width + x;

  ConvertToInterpolationCoefficients( volume + startIdx, height, width );
}

template<class floatN>
__global__ void SamplesToCoefficients3DZ(
  floatN* volume, // in-place processing
  uint width,     // width of the volume
  uint height,    // height of the volume
  uint depth )    // depth of the volume
{
  // process lines in z-direction
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  const uint startIdx = y * width + x;
  const uint slice = height * width;

  ConvertToInterpolationCoefficients( volume + startIdx, depth, slice );
}

//--------------------------------------------------------------------------
// Exported functions
//--------------------------------------------------------------------------

//! Convert the voxel values into cubic b-spline coefficients
//! @param volume  pointer to the voxel volume in GPU (device) memory
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
template<class floatN>
extern void CubicBSplinePrefilter3D( floatN* volume,
  size_t width_, size_t height_, size_t depth_ )
{
  /** Supress x86_64 warnings... **/
  uint width  = (uint)width_;
  uint height = (uint)height_;
  uint depth  = (uint)depth_;

  // Try to determine the optimal block dimensions
  uint dimX = min(min(PowTwoDivider(width), PowTwoDivider(height)), 64);
  uint dimY = min(min(PowTwoDivider(depth), PowTwoDivider(height)), 512/dimX);
  dim3 dimBlock(dimX, dimY);

  // Replace the voxel values by the B-spline coefficients
  dim3 dimGridX( height / dimBlock.x, depth / dimBlock.y );
  SamplesToCoefficients3DX<floatN><<<dimGridX, dimBlock>>>( volume, width, height, depth );
  cuda::cudaCheckMsg( "SamplesToCoefficients3DX kernel failed" );

  dim3 dimGridY( width / dimBlock.x, depth / dimBlock.y );
  SamplesToCoefficients3DY<floatN><<<dimGridY, dimBlock>>>( volume, width, height, depth );
  cuda::cudaCheckMsg( "SamplesToCoefficients3DY kernel failed" );

  dim3 dimGridZ( width / dimBlock.x, height / dimBlock.y );
  SamplesToCoefficients3DZ<floatN><<<dimGridZ, dimBlock>>>( volume, width, height, depth );
  cuda::cudaCheckMsg( "SamplesToCoefficients3DZ kernel failed" );
}

#endif  //_3D_CUBIC_BSPLINE_PREFILTER_H_

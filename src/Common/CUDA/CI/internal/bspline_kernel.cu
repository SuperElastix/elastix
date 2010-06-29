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

#ifndef _CUDA_BSPLINE_H_
#define _CUDA_BSPLINE_H_

#include "cutil_math_bugfixes.h"
#include "math_func.cu"

// Cubic B-spline function
// The 3rd order Maximal Order and Minimum Support function, that it is maximally differentiable.
inline __host__ __device__ float bspline( float t )
{
  t = fabs( t );
  const float a = 2.0f - t;

  if ( t < 1.0f )       return 2.0f / 3.0f - 0.5f * t * t * a;
  else if ( t < 2.0f )  return a * a * a / 6.0f;
  else return 0.0f;
}

// The first order derivative of the cubic B-spline
inline __host__ __device__ float bspline_1st_derivative( float t )
{
  if      (-2.0f < t && t <= -1.0f ) return  0.5f * t * t + 2.0f * t + 2.0f;
  else if (-1.0f < t && t <=  0.0f ) return -1.5f * t * t - 2.0f * t;
  else if ( 0.0f < t && t <=  1.0f ) return  1.5f * t * t - 2.0f * t;
  else if ( 1.0f < t && t <   2.0f ) return -0.5f * t * t + 2.0f * t - 2.0f;
  else return 0.0f;
}

// The second order derivative of the cubic B-spline
inline __host__ __device__ float bspline_2nd_derivative( float t )
{
  t = fabs( t );

  if      ( t < 1.0f ) return 3.0f * t - 2.0f;
  else if ( t < 2.0f ) return 2.0f - t;
  else return 0.0f;
}

// Inline calculation of the bspline convolution weights, without conditional statements
template<class T> inline __device__ void bspline_weights(
  T fraction, T& w0, T& w1, T& w2, T& w3 )
{
  const T one_frac = 1.0f - fraction;
  const T squared = fraction * fraction;
  const T one_sqd = one_frac * one_frac;

  w0 = 1.0f / 6.0f * one_sqd * one_frac;
  w1 = 2.0f / 3.0f - 0.5f * squared * ( 2.0f - fraction );
  w2 = 2.0f / 3.0f - 0.5f * one_sqd * ( 2.0f - one_frac );
  w3 = 1.0f / 6.0f * squared * fraction;
}

// Inline calculation of the first order derivative bspline convolution weights, without conditional statements
template<class T> inline __device__ void bspline_weights_1st_derivative(
  T fraction, T& w0, T& w1, T& w2, T& w3 )
{
  const T squared = fraction * fraction;

  w0 = -0.5f * squared + fraction - 0.5f;
  w1 =  1.5f * squared - 2.0f * fraction;
  w2 = -1.5f * squared + fraction + 0.5f;
  w3 =  0.5f * squared;
}

// Inline calculation of the second order derivative bspline convolution weights, without conditional statements
template<class T> inline __device__ void bspline_weights_2nd_derivative(
  T fraction, T& w0, T& w1, T& w2, T& w3 )
{
  w0 =  1.0f - fraction;
  w1 =  3.0f * fraction - 2.0f;
  w2 = -3.0f * fraction + 1.0f;
  w3 =  fraction;
}

#endif // _CUDA_BSPLINE_H_

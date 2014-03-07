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

#ifndef _CUBIC_BSPLINE_PREFILTER_KERNEL_H_
#define _CUBIC_BSPLINE_PREFILTER_KERNEL_H_

#include "math_func.cu"

// The code below is based on the work of Philippe Thevenaz.
// See <http://bigwww.epfl.ch/thevenaz/interpolation/>

#define Pole (sqrt(3.0f)-2.0f)  //pole for cubic b-spline

//--------------------------------------------------------------------------
// Local GPU device procedures
//--------------------------------------------------------------------------
template<class floatN>
__device__ floatN InitialCausalCoefficient(
  floatN* c,        // coefficients
  uint DataLength,  // number of coefficients
  int step )
{
  const uint Horizon = UMIN( 28, DataLength );

  // this initialization corresponds to mirror boundaries
  // accelerated loop
  float zn = Pole;
  floatN Sum = *c;
  for ( uint n = 1; n < Horizon; n++ )
  {
    c += step;
    Sum += zn * *c;
    zn *= Pole;
  }
  return Sum;
}

template<class floatN>
__device__ floatN InitialAntiCausalCoefficient(
  floatN* c,        // last coefficient
  uint DataLength,  // number of samples or coefficients
  int step )
{
  // this initialization corresponds to mirror boundaries
  return((Pole / (Pole * Pole - 1.0f)) * (Pole * c[-step] + *c));
}

template<class floatN>
__device__ void ConvertToInterpolationCoefficients(
  floatN* coeffs,   // input samples --> output coefficients
  uint DataLength,  // number of samples or coefficients
  int step )
{
  // compute the overall gain
  const float Lambda = (1.0f - Pole) * (1.0f - 1.0f / Pole);

  // causal initialization
  floatN* c = coeffs;
  floatN previous_c;  //cache the previously calculated c rather than look it up again (faster!)
  *c = previous_c = Lambda * InitialCausalCoefficient( c, DataLength, step );
  // causal recursion
  for ( uint n = 1; n < DataLength; n++ )
  {
    c += step;
    *c = previous_c = Lambda * *c + Pole * previous_c;
  }
  // anticausal initialization
  *c = previous_c = InitialAntiCausalCoefficient( c, DataLength, step );
  // anticausal recursion
  for ( int n = DataLength - 2; 0 <= n; n-- )
  {
    c -= step;
    *c = previous_c = Pole * (previous_c - *c);
  }
}

#endif // _CUBIC_BSPLINE_PREFILTER_KERNEL_H_

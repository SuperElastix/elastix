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
// OpenCL implementation of itk::BSplineDecompositionImageFilter

//------------------------------------------------------------------------------
void set_initial_causal_coefficient(BUFFPIXELTYPE *scratch,
                                    const float z, const ulong data_length)
{
  float       sum;
  float       zn, z2n, iz;
  const float tolerance = 1e-10f;

  // this initialization corresponds to mirror boundaries
  uint horizon = data_length;

  zn = z;
  if(tolerance > 0.0f)
  {
    horizon = (uint)( ceil( log(tolerance) / log( fabs(z) ) ) );
  }
  if(horizon < data_length)
  {
    // accelerated loop
    sum = scratch[0]; // verify this
    for(uint n = 1; n < horizon; ++n)
    {
      //sum += zn * scratch[n];
      sum = mad(zn, scratch[n], sum);
      zn *= z;
    }
    scratch[0] = sum;
  }
  else
  {
    // full loop
    iz = 1.0f / z;
    z2n = pow( z, (float)(data_length - 1L) );
    sum = mad(z2n, scratch[data_length - 1L], scratch[0]);
    z2n *= z2n * iz;
    for(uint n = 1; n <= (data_length - 2); ++n)
    {
      //sum += (zn + z2n) * scratch[n];
      sum = mad(zn + z2n, scratch[n], sum);
      zn *= z;
      z2n *= iz;
    }
    scratch[0] = sum / (1.0f - pown(zn, 2));
  }
}

//------------------------------------------------------------------------------
// OpenCL implementation of
// BSplineDecompositionImageFilter::SetInitialAntiCausalCoefficient()
void set_initial_anticausal_coefficient(BUFFPIXELTYPE *scratch,
                                        const float z, const ulong data_length)
{
  // this initialization corresponds to mirror boundaries
  // See Unser, 1999, Box 2 for explanation
  // Also see erratum at http://bigwww.epfl.ch/publications/unser9902.html
  scratch[data_length - 1] =
    ( z / (pown(z, 2) - 1.0f) )
    * ( mad(z, scratch[data_length - 2], scratch[data_length - 1]) );
}

//------------------------------------------------------------------------------
// OpenCL implementation of
// BSplineDecompositionImageFilter::DataToCoefficients1D()
bool data_to_coefficients_1d(BUFFPIXELTYPE *scratch,
                             const uint3 image_size,
                             const float2 in_spline_poles,
                             const int number_of_poles,
                             const uint direction)
{
  // Define data_length
  uint data_length = 0;
  if(direction == 0)
  {
    data_length = image_size.x;
  }
  else if(direction == 1)
  {
    data_length = image_size.y;
  }
  else if(direction == 2)
  {
    data_length = image_size.z;
  }

  if(data_length == 1) //Required by mirror boundaries
  {
    return false;
  }

  // compute overall gain
  float c0 = 1.0f;
  if(number_of_poles == 1)
  {
    float t1 = 1.0f - in_spline_poles.x;
    float t2 = 1.0f - (1.0f / in_spline_poles.x);
    c0 = t1 * t2;
  }
  else if (number_of_poles == 2)
  {
    float t1 = 1.0f - in_spline_poles.x;
    float t2 = 1.0f - (1.0f / in_spline_poles.x);
    float t3 = 1.0f - in_spline_poles.y;
    float t4 = 1.0f - (1.0f / in_spline_poles.y);
    c0 = t1 * t2 * t3 * t4;
  }

  // apply the gain
  for(uint n = 0; n < data_length; ++n)
  {
    scratch[n] *= c0;
  }

  // define spline_poles[2], it is better for loops
  float spline_poles[2];
  spline_poles[0] = in_spline_poles.x;
  spline_poles[1] = in_spline_poles.y;

  // loop over all poles
  for(uint k = 0; k < number_of_poles; ++k)
  {
    // causal initialization
    set_initial_causal_coefficient(scratch, spline_poles[k], data_length);
    // causal recursion
    for(uint n = 1; n < data_length; ++n)
    {
      //scratch[n] += spline_poles[k] * scratch[n - 1];
      scratch[n] = mad(spline_poles[k], scratch[n - 1], scratch[n]);
    }
    // anticausal initialization
    set_initial_anticausal_coefficient(scratch, spline_poles[k], data_length);
    // anticausal recursion
    for(int n = data_length - 2; 0 <= n; n--)
    {
      scratch[n] = spline_poles[k] * (scratch[n + 1] - scratch[n]);
    }
  }
  return true;
}

//------------------------------------------------------------------------------
// Get global memory offset
uint get_image_offset(const uint gix,
                      const uint giy,
                      const uint giz,
                      const uint width, const uint height)
{
  uint gidx = mad24(width, mad24(giz, height, giy), gix);
  return gidx;
}

//------------------------------------------------------------------------------
#ifdef DIM_1
__kernel void BSplineDecompositionImageFilter(__global const INPIXELTYPE *in,
                                              __global OUTPIXELTYPE *out,
                                              uint2 image_size,
                                              float2 spline_poles,
                                              uint number_of_poles,
                                              uint direction)
{
  uint gi = get_global_id(0);
  // Define length
  uint length = 0;

  if(direction == 0)
  {
    length = image_size.x;
  }
  else if(direction == 1)
  {
    length = image_size.y;
  }

  if(gi < length)
  {
    // Local scratch buffer
    BUFFPIXELTYPE scratch[BUFFSIZE];

    // Copy coefficients to scratch
    uint id = 0;
    uint lidx = 0;
    for(uint i = 0; i < length; ++i)
    {
      if(image_size.y != 0)
      {
        if(direction == 0)
        {
          lidx = get_image_offset(i, 0, gi, image_size.x, 1);
        }
        else if(direction == 1)
        {
          lidx = get_image_offset(gi, 0, i, image_size.x, 1);
        }
      }
      else
      {
        lidx = i;
      }
      scratch[id++] = (BUFFPIXELTYPE)(out[lidx]);
    }

    // Perform 1D BSpline calculations
    data_to_coefficients_1d(scratch, (uint3)(image_size.x, image_size.y, 0),
                            spline_poles, number_of_poles, direction);

    // Copy scratch back to coefficients
    id = 0;
    lidx = 0;
    for(uint i = 0; i < length; ++i)
    {
      if(image_size.y != 0)
      {
        if(direction == 0)
        {
          lidx = get_image_offset(i, 0, gi, image_size.x, 1);
        }
        else if(direction == 1)
        {
          lidx = get_image_offset(gi, 0, i, image_size.x, 1);
        }
      }
      else
      {
        lidx = i;
      }
      out[lidx] = (OUTPIXELTYPE)(scratch[id++]);
    }
  }
}

#endif

//------------------------------------------------------------------------------
#ifdef DIM_2
__kernel void BSplineDecompositionImageFilter(__global const INPIXELTYPE *in,
                                              __global OUTPIXELTYPE *out,
                                              uint3 image_size,
                                              float2 spline_poles,
                                              int number_of_poles,
                                              uint direction)
{
  uint2 index = (uint2)( get_global_id(0), get_global_id(1) );

  // 0 (direction x) : y/z
  // 1 (direction y) : x/z
  // 2 (direction z) : x/y
  uint3 length;

  if(direction == 0)
  {
    length.x = image_size.y;
    length.y = image_size.z;
    length.z = image_size.x; // looping over
  }
  else if(direction == 1)
  {
    length.x = image_size.x;
    length.y = image_size.z;
    length.z = image_size.y; // looping over
  }
  else if(direction == 2)
  {
    length.x = image_size.x;
    length.y = image_size.y;
    length.z = image_size.z; // looping over
  }

  if(index.x < length.x && index.y < length.y)
  {
    // Local scratch buffer
    BUFFPIXELTYPE scratch[BUFFSIZE];

    // Copy coefficients to scratch
    uint id = 0;
    uint lidx = 0;

    // Having if() statement inside loop will cost performance penalty.
    // Therefore we are moving for() loop inside the if () statement.
    // The code become less compact, but faster.
    if(direction == 0)
    {
      for(uint i = 0; i < length.z; ++i)
      {
        lidx = get_image_offset(i, index.x, index.y, image_size.x, image_size.y);
        scratch[id++] = (BUFFPIXELTYPE)(out[lidx]);
      }
    }
    else if(direction == 1)
    {
      for(uint i = 0; i < length.z; ++i)
      {
        lidx = get_image_offset(index.x, i, index.y, image_size.x, image_size.y);
        scratch[id++] = (BUFFPIXELTYPE)(out[lidx]);
      }
    }
    else if(direction == 2)
    {
      for(uint i = 0; i < length.z; ++i)
      {
        lidx = get_image_offset(index.x, index.y, i, image_size.x, image_size.y);
        scratch[id++] = (BUFFPIXELTYPE)(out[lidx]);
      }
    }

    // Perform 1D BSpline calculations
    data_to_coefficients_1d(scratch, image_size,
                            spline_poles, number_of_poles, direction);

    // Copy scratch back to coefficients
    id = 0;

    // Having if() statement inside loop will cost performance penalty.
    // Therefore we are moving for() loop inside the if () statement.
    // The code become less compact, but faster.
    if(direction == 0)
    {
      for(uint i = 0; i < length.z; ++i)
      {
        lidx = get_image_offset(i, index.x, index.y, image_size.x, image_size.y);
        out[lidx] = (OUTPIXELTYPE)(scratch[id++]);
      }
    }
    else if(direction == 1)
    {
      for(uint i = 0; i < length.z; ++i)
      {
        lidx = get_image_offset(index.x, i, index.y, image_size.x, image_size.y);
        out[lidx] = (OUTPIXELTYPE)(scratch[id++]);
      }
    }
    else if(direction == 2)
    {
      for(uint i = 0; i < length.z; ++i)
      {
        lidx = get_image_offset(index.x, index.y, i, image_size.x, image_size.y);
        out[lidx] = (OUTPIXELTYPE)(scratch[id++]);
      }
    }
  }
}

#endif

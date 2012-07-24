/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

//------------------------------------------------------------------------------
// OpenCL implementation of
// BSplineDecompositionImageFilter::SetInitialCausalCoefficient()
void set_initial_causal_coefficient(BUFFPIXELTYPE* scratch,
                                    const float z, const ulong data_length)
{
  float sum;
  float zn, z2n, iz;
  const float tolerance = 1e-10;

  // this initialization corresponds to mirror boundaries
  uint horizon = data_length;
  zn = z;
  if(tolerance > 0.0)
  {
    horizon = (uint)(ceil(log(tolerance)/log(fabs(z))));
  }
  if(horizon < data_length)
  {
    // accelerated loop
    sum = scratch[0]; // verify this
    for(uint n = 1; n < horizon; n++)
    {
      sum += zn * scratch[n];
      zn *= z;
    }
    scratch[0] = sum;
  }
  else
  {
    // full loop
    iz = 1.0 / z;
    z2n = pow( z, (float)( data_length - 1L ) );
    sum = scratch[0] + z2n * scratch[data_length - 1L];
    z2n *= z2n * iz;
    for(uint n = 1; n <= ( data_length - 2 ); n++)
    {
      sum += ( zn + z2n ) * scratch[n];
      zn *= z;
      z2n *= iz;
    }
    scratch[0] = sum / ( 1.0 - zn * zn );
  }
}

//------------------------------------------------------------------------------
// OpenCL implementation of
// BSplineDecompositionImageFilter::SetInitialAntiCausalCoefficient()
void set_initial_anticausal_coefficient(BUFFPIXELTYPE* scratch,
                                        const float z, const ulong data_length)
{
  // this initialization corresponds to mirror boundaries
  // See Unser, 1999, Box 2 for explanation
  // Also see erratum at http://bigwww.epfl.ch/publications/unser9902.html
  scratch[data_length - 1] =
    ( z / ( z * z - 1.0 ) )
    * ( z * scratch[data_length - 2] + scratch[data_length - 1] );
}

//------------------------------------------------------------------------------
// OpenCL implementation of BSplineDecompositionImageFilter::DataToCoefficients1D()
bool data_to_coefficients_1d(BUFFPIXELTYPE* scratch,
                             const uint3 image_size,
                             const float2 in_spline_poles,
                             const int number_of_poles,
                             const uint direction)

{
  float spline_poles[2];
  spline_poles[0] = in_spline_poles.x;
  spline_poles[1] = in_spline_poles.y;

  // Define data_length
  ulong data_length = 0;
  if(direction == 0)
  {
    data_length = image_size.x;
  }
  else if (direction == 1)
  {
    data_length = image_size.y;
  }
  else if (direction == 2)
  {
    data_length = image_size.z;
  }

  float c0 = 1.0;
  if(data_length == 1) //Required by mirror boundaries
  {
    return false;
  }

  // Compute overall gain
  for(int k = 0; k < number_of_poles; k++)
  {
    // Note for cubic splines lambda = 6
    c0 = c0 * ( 1.0 - spline_poles[k] ) * ( 1.0 - 1.0 / spline_poles[k] );
  }

  // apply the gain
  for(uint n = 0; n < data_length; n++)
  {
    scratch[n] *= c0;
  }

  // loop over all poles
  for(int k = 0; k < number_of_poles; k++)
  {
    // causal initialization
    set_initial_causal_coefficient(scratch, spline_poles[k], data_length);
    // causal recursion
    for(unsigned int n = 1; n < data_length; n++)
    {
      scratch[n] += spline_poles[k] * scratch[n - 1];
    }
    // anticausal initialization
    set_initial_anticausal_coefficient(scratch, spline_poles[k], data_length);
    // anticausal recursion
    for(int n = data_length - 2; 0 <= n; n--)
    {
      scratch[n] = spline_poles[k] * ( scratch[n + 1] - scratch[n] );
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
  uint gidx = width*(giz*height + giy) + gix;
  return gidx;
}

#ifdef DIM_1
//------------------------------------------------------------------------------
__kernel void BSplineDecompositionImageFilter(__global const INPIXELTYPE* in,
                                              __global OUTPIXELTYPE* out,
                                              uint2 image_size,
                                              float2 spline_poles,
                                              uint number_of_poles,
                                              uint direction)
{
  uint gi = get_global_id(0);
  // Define length
  uint length = 0;
  if(direction == 0)
    length = image_size.x;
  else if(direction == 1)
    length = image_size.y;

  if(gi < length)
  {
    // Local scratch buffer
    BUFFPIXELTYPE scratch[BUFFSIZE];

    // Copy coefficients to scratch
    uint id = 0;
    uint lidx = 0;
    for(uint i = 0; i < length; i++)
    {
      if(image_size.y != 0)
      {
        if(direction == 0)
          lidx = get_image_offset(i, 0, gi, image_size.x, 1);
        else if(direction == 1)
          lidx = get_image_offset(gi, 0, i, image_size.x, 1);
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
    for(uint i = 0; i < length; i++)
    {
      if(image_size.y != 0)
      {
        if(direction == 0)
          lidx = get_image_offset(i, 0, gi, image_size.x, 1);
        else if(direction == 1)
          lidx = get_image_offset(gi, 0, i, image_size.x, 1);
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

#ifdef DIM_2
//------------------------------------------------------------------------------
__kernel void BSplineDecompositionImageFilter(__global const INPIXELTYPE* in,
                                              __global OUTPIXELTYPE* out,
                                              uint3 image_size,
                                              float2 spline_poles,
                                              int number_of_poles,
                                              uint direction)
{
  uint2 index = (uint2)(get_global_id(0), get_global_id(1));

  // 0 (direction x) : y/z
  // 1 (direction y) : x/z
  // 2 (direction z) : x/y
  uint length[3];
  if(direction == 0)
  {
    length[0] = image_size.y;
    length[1] = image_size.z;
    length[2] = image_size.x; // looping over
  }
  else if(direction == 1)
  {
    length[0] = image_size.x;
    length[1] = image_size.z;
    length[2] = image_size.y; // looping over
  }
  else if(direction == 2)
  {
    length[0] = image_size.x;
    length[1] = image_size.y;
    length[2] = image_size.z; // looping over
  }

  if(index.x < length[0] && index.y < length[1])
  {
    // Local scratch buffer
    BUFFPIXELTYPE scratch[BUFFSIZE];

    // Copy coefficients to scratch
    uint id = 0;
    uint lidx = 0;
    for(int i = 0; i < length[2]; i++)
    {
      if(direction == 0)
        lidx = get_image_offset(i, index.x, index.y, image_size.x, image_size.y);
      else if(direction == 1)
        lidx = get_image_offset(index.x, i, index.y, image_size.x, image_size.y);
      else if(direction == 2)
        lidx = get_image_offset(index.x, index.y, i, image_size.x, image_size.y);

      scratch[id++] = (BUFFPIXELTYPE)(out[lidx]);
    }

    // Perform 1D BSpline calculations
    data_to_coefficients_1d(scratch, image_size,
      spline_poles, number_of_poles, direction);

    // Copy scratch back to coefficients
    id = 0;
    lidx = 0;
    for(uint i = 0; i < length[2]; i++)
    {
      if(direction == 0)
        lidx = get_image_offset(i, index.x, index.y, image_size.x, image_size.y);
      else if(direction == 1)
        lidx = get_image_offset(index.x, i, index.y, image_size.x, image_size.y);
      else if(direction == 2)
        lidx = get_image_offset(index.x, index.y, i, image_size.x, image_size.y);

      out[lidx] = (OUTPIXELTYPE)(scratch[id++]);
    }
  }
}
#endif

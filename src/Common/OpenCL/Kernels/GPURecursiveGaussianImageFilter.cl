/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

// Exact copy of FilterDataArray from RecursiveSeparableImageFilter
void FilterDataArray(BUFFPIXELTYPE* outs,
                     const BUFFPIXELTYPE* data,
                     BUFFPIXELTYPE* scratch,
                     unsigned int ln,
                     float m_N0, float m_N1, float m_N2, float m_N3,
                     float m_D1, float m_D2, float m_D3, float m_D4,
                     float m_M1, float m_M2, float m_M3, float m_M4,
                     float m_BN1, float m_BN2, float m_BN3, float m_BN4,
                     float m_BM1, float m_BM2, float m_BM3, float m_BM4)

{
  /**
  * Causal direction pass
  */
  // this value is assumed to exist from the border to infinity.
  const float outV1 = data[0];

  /**
  * Initialize borders
  */
  scratch[0] = (outV1   * m_N0 +   outV1 * m_N1 + outV1   * m_N2 + outV1 * m_N3);
  scratch[1] = (data[1] * m_N0 +   outV1 * m_N1 + outV1   * m_N2 + outV1 * m_N3);
  scratch[2] = (data[2] * m_N0 + data[1] * m_N1 + outV1   * m_N2 + outV1 * m_N3);
  scratch[3] = (data[3] * m_N0 + data[2] * m_N1 + data[1] * m_N2 + outV1 * m_N3);

  // note that the outV1 value is multiplied by the Boundary coefficients m_BNi
  scratch[0] -= (outV1      * m_BN1 + outV1      * m_BN2 + outV1      * m_BN3 + outV1 * m_BN4);
  scratch[1] -= (scratch[0] * m_D1  + outV1      * m_BN2 + outV1      * m_BN3 + outV1 * m_BN4);
  scratch[2] -= (scratch[1] * m_D1  + scratch[0] * m_D2  + outV1      * m_BN3 + outV1 * m_BN4);
  scratch[3] -= (scratch[2] * m_D1  + scratch[1] * m_D2  + scratch[0] * m_D3  + outV1 * m_BN4);

  /**
  * Recursively filter the rest
  */
  for ( unsigned int i = 4; i < ln; i++ )
  {
    scratch[i]  = (data[i]        * m_N0 + data[i - 1]    * m_N1 + data[i - 2]    * m_N2 + data[i - 3]    * m_N3);
    scratch[i] -= (scratch[i - 1] * m_D1 + scratch[i - 2] * m_D2 + scratch[i - 3] * m_D3 + scratch[i - 4] * m_D4);
  }

  /**
  * Store the causal result
  */
  for ( unsigned int i = 0; i < ln; i++ )
  {
    outs[i] = scratch[i];
  }

  /**
  * AntiCausal direction pass
  */
  // this value is assumed to exist from the border to infinity.
  const float outV2 = data[ln - 1];

  /**
  * Initialize borders
  */
  scratch[ln - 1] = (outV2        * m_M1 + outV2        * m_M2 + outV2        * m_M3 + outV2 * m_M4);
  scratch[ln - 2] = (data[ln - 1] * m_M1 + outV2        * m_M2 + outV2        * m_M3 + outV2 * m_M4);
  scratch[ln - 3] = (data[ln - 2] * m_M1 + data[ln - 1] * m_M2 + outV2        * m_M3 + outV2 * m_M4);
  scratch[ln - 4] = (data[ln - 3] * m_M1 + data[ln - 2] * m_M2 + data[ln - 1] * m_M3 + outV2 * m_M4);

  // note that the outV2value is multiplied by the Boundary coefficients m_BMi
  scratch[ln - 1] -= (outV2           * m_BM1 + outV2           * m_BM2 + outV2           * m_BM3 + outV2 * m_BM4);
  scratch[ln - 2] -= (scratch[ln - 1] * m_D1  + outV2           * m_BM2 + outV2           * m_BM3 + outV2 * m_BM4);
  scratch[ln - 3] -= (scratch[ln - 2] * m_D1  + scratch[ln - 1] * m_D2  + outV2           * m_BM3 + outV2 * m_BM4);
  scratch[ln - 4] -= (scratch[ln - 3] * m_D1  + scratch[ln - 2] * m_D2  + scratch[ln - 1] * m_D3  + outV2 * m_BM4);

  /**
  * Recursively filter the rest
  */
  for ( unsigned int i = ln - 4; i > 0; i-- )
  {
    scratch[i - 1] = (data[i]     * m_M1 + data[i + 1]    * m_M2 + data[i + 2]    * m_M3 + data[i + 3]    * m_M4);
    scratch[i - 1] -= (scratch[i] * m_D1 + scratch[i + 1] * m_D2 + scratch[i + 2] * m_D3 + scratch[i + 3] * m_D4);
  }

  /**
  * Roll the antiCausal part into the output
  */
  for ( unsigned int i = 0; i < ln; i++ )
  {
    outs[i] += scratch[i];
  }
}

// Get global memory offset
unsigned int GetImageOffset(const int gix,
                            const int giy,
                            const int giz,
                            const int width, const int height)
{
  unsigned int gidx = width*(giz*height + giy) + gix;
  return gidx;
}

#ifdef DIM_1
#define DIM 1
__kernel void RecursiveGaussianImageFilter(__global const INPIXELTYPE* in, __global OUTPIXELTYPE* out,
                                           unsigned int ln, int direction,
                                           float m_N0, float m_N1, float m_N2, float m_N3,
                                           float m_D1, float m_D2, float m_D3, float m_D4,
                                           float m_M1, float m_M2, float m_M3, float m_M4,
                                           float m_BN1, float m_BN2, float m_BN3, float m_BN4,
                                           float m_BM1, float m_BM2, float m_BM3, float m_BM4,
                                           int width, int height)
{
  int gi = get_global_id(0);

  // Define length
  int length = 0;
  if(direction == 0)
    length = width;
  else if(direction == 1)
    length = height;

  if(gi < length)
  {
    // Local buffers
    BUFFPIXELTYPE inps[BUFFSIZE];
    BUFFPIXELTYPE outs[BUFFSIZE];
    BUFFPIXELTYPE scratch[BUFFSIZE];

    // Fill local input buffer
    unsigned int id = 0;
    unsigned int lidx = 0;
    for (int i = 0; i < length; i++)
    {
      if(height != 0)
      {
        if(direction == 0)
          lidx = GetImageOffset(i, 0, gi, width, 1);
        else if(direction == 1)
          lidx = GetImageOffset(gi, 0, i, width, 1);
      }
      else
      {
        lidx = i;
      }

      inps[id++] = (BUFFPIXELTYPE)(in[lidx]);
    }

    /** Apply the recursive Filter to an array of data. This method is called
    * for each line of the volume. Parameter "scratch" is a scratch
    * area used for internal computations that is the same size as the
    * parameters "outs" and "data".*/
    FilterDataArray(outs, inps, scratch, ln,
      m_N0, m_N1, m_N2, m_N3,
      m_D1, m_D2, m_D3, m_D4,
      m_M1, m_M2, m_M3, m_M4,
      m_BN1, m_BN2, m_BN3, m_BN4,
      m_BM1, m_BM2, m_BM3, m_BM4);

    // Copy to output
    id = 0;
    lidx = 0;
    for (int i = 0; i < length; i++)
    {
      if(height != 0)
      {
        if(direction == 0)
          lidx = GetImageOffset(i, 0, gi, width, 1);
        else if(direction == 1)
          lidx = GetImageOffset(gi, 0, i, width, 1);
      }
      else
      {
        lidx = i;
      }

      out[lidx] = (OUTPIXELTYPE)(outs[id++]);
    }
  }
}
#endif

#ifdef DIM_2
#define DIM 2
__kernel void RecursiveGaussianImageFilter(__global const INPIXELTYPE* in, __global OUTPIXELTYPE* out,
                                           unsigned int ln, int direction,
                                           float m_N0, float m_N1, float m_N2, float m_N3,
                                           float m_D1, float m_D2, float m_D3, float m_D4,
                                           float m_M1, float m_M2, float m_M3, float m_M4,
                                           float m_BN1, float m_BN2, float m_BN3, float m_BN4,
                                           float m_BM1, float m_BM2, float m_BM3, float m_BM4,
                                           int width, int height, int depth)
{
  int gi1 = get_global_id(0);
  int gi2 = get_global_id(1);

  // 0 (direction x) : y/z
  // 1 (direction y) : x/z
  // 2 (direction z) : x/y
  int length[3];
  if(direction == 0)
  {
    length[0] = height;
    length[1] = depth;
    length[2] = width;  // looping over
  }
  else if(direction == 1)
  {
    length[0] = width;
    length[1] = depth;
    length[2] = height; // looping over
  }
  else if(direction == 2)
  {
    length[0] = width;
    length[1] = height;
    length[2] = depth;  // looping over
  }

  if(gi1 < length[0] && gi2 < length[1])
  {
    // Local buffers
    BUFFPIXELTYPE inps[BUFFSIZE];
    BUFFPIXELTYPE outs[BUFFSIZE];
    BUFFPIXELTYPE scratch[BUFFSIZE];

    // Fill local input buffer
    unsigned int id = 0;
    unsigned int lidx = 0;
    for (int i = 0; i < length[2]; i++)
    {
      if(direction == 0)
        lidx = GetImageOffset(i, gi1, gi2, width, height);
      else if(direction == 1)
        lidx = GetImageOffset(gi1, i, gi2, width, height);
      else if(direction == 2)
        lidx = GetImageOffset(gi1, gi2, i, width, height);

      inps[id++] = (BUFFPIXELTYPE)(in[lidx]);
    }

    /** Apply the recursive Filter to an array of data. This method is called
    * for each line of the volume. Parameter "scratch" is a scratch
    * area used for internal computations that is the same size as the
    * parameters "outs" and "data".*/
    FilterDataArray(outs, inps, scratch, ln,
      m_N0, m_N1, m_N2, m_N3,
      m_D1, m_D2, m_D3, m_D4,
      m_M1, m_M2, m_M3, m_M4,
      m_BN1, m_BN2, m_BN3, m_BN4,
      m_BM1, m_BM2, m_BM3, m_BM4);

    // Copy to output
    id = 0;
    lidx = 0;
    for (int i = 0; i < length[2]; i++)
    {
      if(direction == 0)
        lidx = GetImageOffset(i, gi1, gi2, width, height);
      else if(direction == 1)
        lidx = GetImageOffset(gi1, i, gi2, width, height);
      else if(direction == 2)
        lidx = GetImageOffset(gi1, gi2, i, width, height);

      out[lidx] = (OUTPIXELTYPE)(outs[id++]);
    }
  }
}
#endif

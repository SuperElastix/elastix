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

#ifndef itkOpenCLOstreamSupport_h
#define itkOpenCLOstreamSupport_h

#include "itkOpenCL.h"

#include <ostream>
#include <iomanip>

//------------------------------------------------------------------------------
// uchar
namespace itk
{
template <typename ucharT, typename traits>
inline std::basic_ostream<ucharT, traits> &
operator<<(std::basic_ostream<ucharT, traits> & strm, const cl_uchar & _v)
{
  strm << "(uchar)(" << _v << ")";
  return strm;
}


template <typename ucharT, typename traits>
inline std::basic_ostream<ucharT, traits> &
operator<<(std::basic_ostream<ucharT, traits> & strm, const cl_uchar2 & _v)
{
  strm << "(uchar2)(";
  for (unsigned int i = 0; i < 2; ++i)
  {
    strm << _v.s[i];
    if (i != 1)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename ucharT, typename traits>
inline std::basic_ostream<ucharT, traits> &
operator<<(std::basic_ostream<ucharT, traits> & strm, const cl_uchar4 & _v)
{
  strm << "(uchar4)(";
  for (unsigned int i = 0; i < 4; ++i)
  {
    strm << _v.s[i];
    if (i != 3)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename ucharT, typename traits>
inline std::basic_ostream<ucharT, traits> &
operator<<(std::basic_ostream<ucharT, traits> & strm, const cl_uchar8 & _v)
{
  strm << "(uchar8)(";
  for (unsigned int i = 0; i < 8; ++i)
  {
    strm << _v.s[i];
    if (i != 7)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename ucharT, typename traits>
inline std::basic_ostream<ucharT, traits> &
operator<<(std::basic_ostream<ucharT, traits> & strm, const cl_uchar16 & _v)
{
  strm << "(uchar16)(";
  for (unsigned int i = 0; i < 16; ++i)
  {
    strm << _v.s[i];
    if (i != 15)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


//------------------------------------------------------------------------------
// char
template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const cl_char & _v)
{
  strm << "(char)(" << _v << ")";
  return strm;
}


template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const cl_char2 & _v)
{
  strm << "(char2)(";
  for (unsigned int i = 0; i < 2; ++i)
  {
    strm << _v.s[i];
    if (i != 1)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const cl_char4 & _v)
{
  strm << "(char4)(";
  for (unsigned int i = 0; i < 4; ++i)
  {
    strm << _v.s[i];
    if (i != 3)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const cl_char8 & _v)
{
  strm << "(char8)(";
  for (unsigned int i = 0; i < 8; ++i)
  {
    strm << _v.s[i];
    if (i != 7)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const cl_char16 & _v)
{
  strm << "(char16)(";
  for (unsigned int i = 0; i < 16; ++i)
  {
    strm << _v.s[i];
    if (i != 15)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


//------------------------------------------------------------------------------
// ushort
template <typename ushortT, typename traits>
inline std::basic_ostream<ushortT, traits> &
operator<<(std::basic_ostream<ushortT, traits> & strm, const cl_ushort & _v)
{
  strm << "(ushort)(" << _v << ")";
  return strm;
}


template <typename ushortT, typename traits>
inline std::basic_ostream<ushortT, traits> &
operator<<(std::basic_ostream<ushortT, traits> & strm, const cl_ushort2 & _v)
{
  strm << "(ushort2)(";
  for (unsigned int i = 0; i < 2; ++i)
  {
    strm << _v.s[i];
    if (i != 1)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename ushortT, typename traits>
inline std::basic_ostream<ushortT, traits> &
operator<<(std::basic_ostream<ushortT, traits> & strm, const cl_ushort4 & _v)
{
  strm << "(ushort4)(";
  for (unsigned int i = 0; i < 4; ++i)
  {
    strm << _v.s[i];
    if (i != 3)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename ushortT, typename traits>
inline std::basic_ostream<ushortT, traits> &
operator<<(std::basic_ostream<ushortT, traits> & strm, const cl_ushort8 & _v)
{
  strm << "(ushort8)(";
  for (unsigned int i = 0; i < 8; ++i)
  {
    strm << _v.s[i];
    if (i != 7)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename ushortT, typename traits>
inline std::basic_ostream<ushortT, traits> &
operator<<(std::basic_ostream<ushortT, traits> & strm, const cl_ushort16 & _v)
{
  strm << "(ushort16)(";
  for (unsigned int i = 0; i < 16; ++i)
  {
    strm << _v.s[i];
    if (i != 15)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


//------------------------------------------------------------------------------
// short
template <typename shortT, typename traits>
inline std::basic_ostream<shortT, traits> &
operator<<(std::basic_ostream<shortT, traits> & strm, const cl_short & _v)
{
  strm << "(short)(" << _v << ")";
  return strm;
}


template <typename shortT, typename traits>
inline std::basic_ostream<shortT, traits> &
operator<<(std::basic_ostream<shortT, traits> & strm, const cl_short2 & _v)
{
  strm << "(short2)(";
  for (unsigned int i = 0; i < 2; ++i)
  {
    strm << _v.s[i];
    if (i != 1)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename shortT, typename traits>
inline std::basic_ostream<shortT, traits> &
operator<<(std::basic_ostream<shortT, traits> & strm, const cl_short4 & _v)
{
  strm << "(short4)(";
  for (unsigned int i = 0; i < 4; ++i)
  {
    strm << _v.s[i];
    if (i != 3)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename shortT, typename traits>
inline std::basic_ostream<shortT, traits> &
operator<<(std::basic_ostream<shortT, traits> & strm, const cl_short8 & _v)
{
  strm << "(short8)(";
  for (unsigned int i = 0; i < 8; ++i)
  {
    strm << _v.s[i];
    if (i != 7)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename shortT, typename traits>
inline std::basic_ostream<shortT, traits> &
operator<<(std::basic_ostream<shortT, traits> & strm, const cl_short16 & _v)
{
  strm << "(short16)(";
  for (unsigned int i = 0; i < 16; ++i)
  {
    strm << _v.s[i];
    if (i != 15)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


//------------------------------------------------------------------------------
// uint
template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const cl_uint & _v)
{
  strm << "(uint)(" << _v << ")";
  return strm;
}


template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const cl_uint2 & _v)
{
  strm << "(uint2)(";
  for (unsigned int i = 0; i < 2; ++i)
  {
    strm << _v.s[i];
    if (i != 1)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const cl_uint4 & _v)
{
  strm << "(uint4)(";
  for (unsigned int i = 0; i < 4; ++i)
  {
    strm << _v.s[i];
    if (i != 3)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const cl_uint8 & _v)
{
  strm << "(uint8)(";
  for (unsigned int i = 0; i < 8; ++i)
  {
    strm << _v.s[i];
    if (i != 7)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const cl_uint16 & _v)
{
  strm << "(uint16)(";
  for (unsigned int i = 0; i < 16; ++i)
  {
    strm << _v.s[i];
    if (i != 15)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


//------------------------------------------------------------------------------
// int
template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const cl_int & _v)
{
  strm << "(int)(" << _v << ")";
  return strm;
}


template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const cl_int2 & _v)
{
  strm << "(int2)(";
  for (unsigned int i = 0; i < 2; ++i)
  {
    strm << _v.s[i];
    if (i != 1)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const cl_int4 & _v)
{
  strm << "(int4)(";
  for (unsigned int i = 0; i < 4; ++i)
  {
    strm << _v.s[i];
    if (i != 3)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const cl_int8 & _v)
{
  strm << "(int8)(";
  for (unsigned int i = 0; i < 8; ++i)
  {
    strm << _v.s[i];
    if (i != 7)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const cl_int16 & _v)
{
  strm << "(int16)(";
  for (unsigned int i = 0; i < 16; ++i)
  {
    strm << _v.s[i];
    if (i != 15)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


//------------------------------------------------------------------------------
// ulong
template <typename ulongT, typename traits>
inline std::basic_ostream<ulongT, traits> &
operator<<(std::basic_ostream<ulongT, traits> & strm, const cl_ulong & _v)
{
  strm << "(ulong)(" << _v << ")";
  return strm;
}


template <typename ulongT, typename traits>
inline std::basic_ostream<ulongT, traits> &
operator<<(std::basic_ostream<ulongT, traits> & strm, const cl_ulong2 & _v)
{
  strm << "(ulong2)(";
  for (unsigned int i = 0; i < 2; ++i)
  {
    strm << _v.s[i];
    if (i != 1)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename ulongT, typename traits>
inline std::basic_ostream<ulongT, traits> &
operator<<(std::basic_ostream<ulongT, traits> & strm, const cl_ulong4 & _v)
{
  strm << "(ulong4)(";
  for (unsigned int i = 0; i < 4; ++i)
  {
    strm << _v.s[i];
    if (i != 3)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename ulongT, typename traits>
inline std::basic_ostream<ulongT, traits> &
operator<<(std::basic_ostream<ulongT, traits> & strm, const cl_ulong8 & _v)
{
  strm << "(ulong8)(";
  for (unsigned int i = 0; i < 8; ++i)
  {
    strm << _v.s[i];
    if (i != 7)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename ulongT, typename traits>
inline std::basic_ostream<ulongT, traits> &
operator<<(std::basic_ostream<ulongT, traits> & strm, const cl_ulong16 & _v)
{
  strm << "(ulong16)(";
  for (unsigned int i = 0; i < 16; ++i)
  {
    strm << _v.s[i];
    if (i != 15)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


//------------------------------------------------------------------------------
// long
template <typename longT, typename traits>
inline std::basic_ostream<longT, traits> &
operator<<(std::basic_ostream<longT, traits> & strm, const cl_long & _v)
{
  strm << "(long)(" << _v << ")";
  return strm;
}


template <typename longT, typename traits>
inline std::basic_ostream<longT, traits> &
operator<<(std::basic_ostream<longT, traits> & strm, const cl_long2 & _v)
{
  strm << "(long2)(";
  for (unsigned int i = 0; i < 2; ++i)
  {
    strm << _v.s[i];
    if (i != 1)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename longT, typename traits>
inline std::basic_ostream<longT, traits> &
operator<<(std::basic_ostream<longT, traits> & strm, const cl_long4 & _v)
{
  strm << "(long4)(";
  for (unsigned int i = 0; i < 4; ++i)
  {
    strm << _v.s[i];
    if (i != 3)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename longT, typename traits>
inline std::basic_ostream<longT, traits> &
operator<<(std::basic_ostream<longT, traits> & strm, const cl_long8 & _v)
{
  strm << "(long8)(";
  for (unsigned int i = 0; i < 8; ++i)
  {
    strm << _v.s[i];
    if (i != 7)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename longT, typename traits>
inline std::basic_ostream<longT, traits> &
operator<<(std::basic_ostream<longT, traits> & strm, const cl_long16 & _v)
{
  strm << "(long16)(";
  for (unsigned int i = 0; i < 16; ++i)
  {
    strm << _v.s[i];
    if (i != 15)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


//------------------------------------------------------------------------------
// float
template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const cl_float & _v)
{
  strm << "(float)(" << _v << ")";
  return strm;
}


template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const cl_float2 & _v)
{
  strm << "(float2)(";
  for (unsigned int i = 0; i < 2; ++i)
  {
    strm << std::fixed << std::setprecision(8) << _v.s[i];
    if (i != 1)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const cl_float4 & _v)
{
  strm << "(float4)(";
  for (unsigned int i = 0; i < 4; ++i)
  {
    strm << std::fixed << std::setprecision(8) << _v.s[i];
    if (i != 3)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const cl_float8 & _v)
{
  strm << "(float8)(";
  for (unsigned int i = 0; i < 8; ++i)
  {
    strm << std::fixed << std::setprecision(8) << _v.s[i];
    if (i != 7)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const cl_float16 & _v)
{
  strm << "(float16)(";
  for (unsigned int i = 0; i < 16; ++i)
  {
    strm << std::fixed << std::setprecision(8) << _v.s[i];
    if (i != 15)
    {
      strm << ", ";
    }
  }
  strm << ")";
  return strm;
}


} // end namespace itk

#endif // /* itkOpenCLOstreamSupport_h */

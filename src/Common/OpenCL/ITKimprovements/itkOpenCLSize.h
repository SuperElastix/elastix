/*=========================================================================
*
*  Copyright Insight Software Consortium
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*         http://www.apache.org/licenses/LICENSE-2.0.txt
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
*=========================================================================*/
#ifndef __itkOpenCLWorkSize_h
#define __itkOpenCLWorkSize_h

#include "itkOpenCL.h"
#include "itkSize.h"

#include <string>

namespace itk
{
/** \class OpenCLSize
 * \brief OpenCLSize
 *
 * \ingroup ITKGPUCommon
 */
//class OpenCLDevice;

class ITKOpenCL_EXPORT OpenCLSize
{
public:
  typedef Size< 1 > SizeType1D;
  typedef Size< 2 > SizeType2D;
  typedef Size< 3 > SizeType3D;

  /** Null compatibility */
  struct Null {};
  static const Null null;
  OpenCLSize(const Null &):
    m_Dim(0)
  { m_Sizes[0] = 0; m_Sizes[1] = 0; m_Sizes[2] = 0; }

  OpenCLSize & operator=(const Null &) { *this = OpenCLSize(null); return *this; }
  bool IsNull() const { return m_Dim == 0; }

  /** Constructs a default size consisting of a single dimension
   * with width set to 1. */
  OpenCLSize():
    m_Dim(1)
  { m_Sizes[0] = 1; m_Sizes[1] = 1; m_Sizes[2] = 1; }

  /** Constructs a single-dimensional size with width set to size.
   * The height and depth will be set to 1. */
  OpenCLSize(const std::size_t size):
    m_Dim(1)
  { m_Sizes[0] = size; m_Sizes[1] = 1; m_Sizes[2] = 1; }

  /** Constructs a two-dimensional size of width, height.
   * The depth will be set to 1. */
  OpenCLSize(const std::size_t width, const std::size_t height):
    m_Dim(2)
  { m_Sizes[0] = width; m_Sizes[1] = height; m_Sizes[2] = 1; }

  /** Constructs a three-dimensional size of width, height and depth. */
  OpenCLSize(const std::size_t width, const std::size_t height, const std::size_t depth):
    m_Dim(3)
  { m_Sizes[0] = width; m_Sizes[1] = height; m_Sizes[2] = depth; }

  /** Constructs a single-dimensional size from one dimention itk::Size.
   * The height and depth will be set to 1.
   * \sa itk::Size */
  OpenCLSize(const SizeType1D & size):
    m_Dim(1)
  { m_Sizes[0] = size[0]; m_Sizes[1] = 1; m_Sizes[2] = 1; }

  /** Constructs a two-dimensional size from two-dimention itk::Size.
   * The depth will be set to 1.
   * \sa itk::Size */
  OpenCLSize(const SizeType2D & size):
    m_Dim(2)
  { m_Sizes[0] = size[0]; m_Sizes[1] = size[1]; m_Sizes[2] = 1; }

  /** Constructs a three-dimensional size from three-dimention itk::Size.
   * \sa itk::Size */
  OpenCLSize(const SizeType3D & size):
    m_Dim(3)
  { m_Sizes[0] = size[0]; m_Sizes[1] = size[1]; m_Sizes[2] = size[2]; }

  /** Returns the dimension for this size, 1, 2, or 3. */
  cl_uint GetDimension() const { return m_Dim; }

  /** Returns the width of this size. */
  std::size_t GetWidth() const { return m_Sizes[0]; }

  /** Returns the height of this size. */
  std::size_t GetHeight() const { return m_Sizes[1]; }

  /** Returns the depth of this size. */
  std::size_t GetDepth() const { return m_Sizes[2]; }

  /** Returns a const pointer to the size array within this object. */
  const std::size_t * GetSizes() const { return m_Sizes; }

  /** Returns the best-fit local work size that evenly divides this work
   * size and fits within the maximums defined by maxWorkItemSize
   * and maxItemsPerGroup.
   * This function is typically used to convert an arbitrary global
   * work size on a QCLKernel into a compatible local work size. */
  OpenCLSize toLocalWorkSize
    (const OpenCLSize & maxWorkItemSize, const std::size_t maxItemsPerGroup) const;

  /** Returns the best-fit local work size that evenly divides this
   * work size and fits within the maximum work group size of \a device.
   * This function is typically used to convert an arbitrary global
   * work size on a QCLKernel into a compatible local work size. */
  //OpenCLSize toLocalWorkSize(const OpenCLDevice & device) const;

  /** Returns the result of rounding this work size up to a multiple of size. */
  OpenCLSize roundTo(const OpenCLSize & size) const;

private:
  cl_uint m_Dim;
  std::size_t m_Sizes[3];
};

/** Operator ==
 * Returns true if lhs and rhs are equal, otherwise returns false. */
bool ITKOpenCL_EXPORT operator==(const OpenCLSize & lhs, const OpenCLSize & rhs);

/** Operator !=
 * Returns true if lhs and rhs are different, otherwise returns false. */
bool ITKOpenCL_EXPORT operator!=(const OpenCLSize & lhs, const OpenCLSize & rhs);

/** Stream out operator for OpenCLSize */
template< class charT, class traits >
inline
std::basic_ostream< charT, traits > & operator<<(std::basic_ostream< charT, traits > & strm,
                                                 const OpenCLSize & size)
{
  const cl_uint dim = size.GetDimension();

  if ( dim == 0 )
    {
    strm << "OpenCLSize(null)";
    }
  else if ( dim == 1 )
    {
    strm << "OpenCLSize(" << size.GetWidth() << ')';
    }
  else if ( dim == 2 )
    {
    strm << "OpenCLSize(" << size.GetWidth() << ", " << size.GetHeight() << ')';
    }
  else
    {
    strm << "OpenCLSize(" << size.GetWidth() << ", " << size.GetHeight() << ", " << size.GetDepth() << ')';
    }
  return strm;
}
} // end namespace itk

#endif /* __itkOpenCLWorkSize_h */

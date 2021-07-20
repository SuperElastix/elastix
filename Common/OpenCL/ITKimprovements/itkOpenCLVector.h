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
#ifndef itkOpenCLVector_h
#define itkOpenCLVector_h

#include "itkOpenCLVectorBase.h"
#include "itkOpenCLBuffer.h"
#include "itkMacro.h"

namespace itk
{
/** \class OpenCLVector
 * \brief The OpenCLVector class represents a templated OpenCL buffer object.
 *
 * OpenCLVector is a convenience template class that wraps an
 * OpenCL buffer object to make it appear as a host-accessible array
 * of elements of type T.
 *
 * Whenever the host CPU calls operator[](), the array's contents
 * are copied into host-accessible memory for direct access. When the
 * host sets the vector on a OpenCLKernel as an argument, the data is
 * copied back to the OpenCL compute device (e.g., the GPU).
 *
 * The type T is restricted to primitive and movable types that do
 * not require explicit construction, destruction, or operator=().
 * T can have constructors, but they will not be called.  This is
 * because the data will be copied to the OpenCL device in a different
 * address space, and the OpenCL device will not know how to construct,
 * destruct, or copy the data in a C++-compatible manner.
 *
 * Elements of a buffer object can be a scalar data type (such as an int, float),
 * vector data type, or a user-defined structure, not std::string.
 *
 * \ingroup OpenCL
 */
template <typename T>
class ITK_TEMPLATE_EXPORT OpenCLVector : public OpenCLVectorBase
{
public:
  /** Standard class typedefs. */
  typedef OpenCLVector Self;

  /** Creates a null OpenCL vector.
   * \sa IsNull() */
  OpenCLVector();

  /** Creates a copy of the \a other vector reference. The vector's
   * contents are not duplicated, modifications to this vector
   * will also affect \a other.
   * \sa operator=() */
  OpenCLVector(const OpenCLVector<T> & other);

  /** Destroys this vector reference. If this is the last reference
   * to the underlying data, the vector will be unmapped and deallocated. */
  ~OpenCLVector();

  /** Assigns the \a other vector reference to this object.
   * The vector's contents are not duplicated, modifications to
   * this vector will also affect \a other. */
  OpenCLVector<T> &
  operator=(const OpenCLVector<T> & other);

  /** Returns true if this vector is null, false otherwise. */
  bool
  IsNull() const;

  /** Releases the contents of this OpenCL vector. If not explicitly
   * released, the contents will be implicitly released when the
   * vector is destroyed. */
  void
  Release();

  /** Returns true if this OpenCL vector is empty, false otherwise. */
  bool
  IsEmpty() const
  {
    return this->m_Size == 0;
  }

  /** Returns the number of elements of type T in this OpenCL vector. */
  std::size_t
  GetSize() const
  {
    return this->m_Size;
  }

  /** Returns a reference to the element at \a index in this OpenCL vector.
   * The vector will be copied to host memory if necessary. */
  T & operator[](const std::size_t index);

  /** Returns a const reference to the element at \a index in this
   * OpenCL vector. The vector will be copied to host memory
   * if necessary. */
  const T & operator[](const std::size_t index) const;

  /** Reads the \a count elements starting \a offset in this vector
   * into \a data.
   * \sa Write() */
  void
  Read(T * data, const std::size_t count, const std::size_t offset = 0);

  /** Writes the \a count elements from \a data to \a offset in this vector.
   * \sa Read() */
  void
  Write(const T * data, const std::size_t count, const std::size_t offset = 0);

  /** Writes the contents of \a data to \a offset in this vector. */
  void
  Write(const Vector<T> & data, const std::size_t offset = 0);

  /** Returns the OpenCL context that was used to create this vector. */
  OpenCLContext *
  GetContext() const;

  /** Returns the OpenCL buffer handle for this vector. */
  OpenCLBuffer
  GetBuffer() const;

private:
  /** Constructs an OpenCL vector object of \a size
   * and associates it with \a context. */
  OpenCLVector(OpenCLContext * context, const OpenCLMemoryObject::Access access, const std::size_t size);

  /** friends from OpenCL core */
  friend class OpenCLContext;
};

//------------------------------------------------------------------------------
/** Stream out operator for OpenCLVector */
template <typename charT, typename traits, typename dataType>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const OpenCLVector<dataType> & vector)
{
  if (vector.IsNull())
  {
    strm << "OpenCLVector(null)";
    return strm;
  }

  const char indent = ' ';

  strm << "OpenCLVector" << std::endl << indent << "Size: " << vector.GetSize() << std::endl;

  strm << std::endl;

  return strm;
}


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkOpenCLVector.hxx"
#endif

#endif /* itkOpenCLVector_h */

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
#ifndef itkOpenCLVectorBase_h
#define itkOpenCLVectorBase_h

#include "itkOpenCLMemoryObject.h"

namespace itk
{
/** \class OpenCLVectorBase
 * \brief The base class for the OpenCLVector implementation.
 *
 * \ingroup OpenCL
 * \sa OpenCLVector
 */

// Forward declaration
class OpenCLContext;
class OpenCLKernel;
class OpenCLVectorBasePimpl;

class ITKOpenCL_EXPORT OpenCLVectorBase
{
protected:
  /** Creates a vector with \a elemSize. */
  OpenCLVectorBase(const std::size_t elemSize);

  /** Creates a copy of the \a other vector reference with \a elemSize.
   * The vector's contents are not duplicated, modifications to this vector
   * will also affect \a other. */
  OpenCLVectorBase(const std::size_t elemSize, const OpenCLVectorBase & other);

  /** Destroys this vector reference. If this is the last reference
   * to the underlying data, the vector will be unmapped and deallocated. */
  ~OpenCLVectorBase();

  /** Assign vector and take ownership. */
  void
  Assign(const OpenCLVectorBase & other);

  /** A vector object is created using the following method. */
  void
  Create(OpenCLContext * context, const OpenCLMemoryObject::Access access, const std::size_t size);

  /** Decrements the memobj reference count. */
  void
  Release();

  /** Enqueues a command to map a region of the buffer object given by buffer
   * into the host address.
   * \sa Unmap() */
  void
  Map();

  /** Enqueues a command to unmap a previously mapped region of a memory object.
   * \sa Map() */
  void
  Unmap() const;

  /** The following functions enqueue commands to read from a buffer object to host memory.
   * \sa Write() */
  void
  Read(void * data, const std::size_t size, const std::size_t offset = 0);

  /** The following functions enqueue commands to write from a buffer object to host memory.
   * \sa Read() */
  void
  Write(const void * data, const std::size_t size, const std::size_t offset = 0);

  /** Returns the native OpenCL identifier for this memory object. */
  cl_mem
  GetMemoryId() const;

  /** Returns the OpenCL context that created this memory object. */
  OpenCLContext *
  GetContext() const;

  /** Unmaps and returns the native OpenCL identifier for this memory object. */
  cl_mem
  GetKernelArgument() const;

protected:
  OpenCLVectorBasePimpl * d_ptr;
  std::size_t             m_ElementSize;
  std::size_t             m_Size;
  mutable void *          m_Mapped;

  /** friends from OpenCL core */
  friend class OpenCLKernel;
};

} // end namespace itk

#endif /* itkOpenCLVectorBase_h */

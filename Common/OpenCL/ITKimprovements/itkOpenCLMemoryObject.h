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
#ifndef itkOpenCLMemoryObject_h
#define itkOpenCLMemoryObject_h

#include "itkOpenCLEventList.h"
#include "itkOpenCLSize.h"
#include "itkPoint.h"

namespace itk
{
/** \class OpenCLMemoryObject
 * \brief The OpenCLMemoryObject class represents all common memory objects
 * such as buffers and image objects.
 *
 * \ingroup OpenCL
 * \sa OpenCLContext
 */

// Forward declaration
class OpenCLContext;

class ITKOpenCL_EXPORT OpenCLMemoryObject
{
protected:
  /** Constructs a null OpenCL memory object and associates it with \a context. */
  OpenCLMemoryObject(OpenCLContext * context = 0)
    : m_Context(context)
    , m_Id(0)
  {}

  /** Constructs an OpenCL memory object from the native identifier \a id,
   * and associates it with \a context. This class takes over ownership
   * of \a id and will release it in the destructor. */
  OpenCLMemoryObject(OpenCLContext * context, const cl_mem id)
    : m_Context(context)
    , m_Id(id)
  {}

  /** Destructor for OpenCL memory object. After the memory object reference count
   * becomes zero and commands queued for execution on a command-queue(s) that
   * use memory object have finished, the memory object is deleted.
   * If memory object is a buffer object, memory object cannot be deleted until
   * all sub-buffer objects associated with memory object are deleted. */
  ~OpenCLMemoryObject();

public:
  /** Standard class typedefs. */
  using Self = OpenCLMemoryObject;
  using RectangleType = Size<4>;
  using PointType = Point<std::size_t, 2>;
  using SizeType = Size<2>;

  /** \enum OpenCLMemoryObject::Access
   * This enum defines the access mode to the OpenCL memory objects.
   * \value ReadWrite This flag specifies that the memory object will be read
   *  and written by a kernel. This is the default.
   *
   * \value WriteOnly This flags specifies that the memory object will be written
   *   but not read by a kernel.
   * \note Reading from a buffer or image object created with WriteOnly inside
   *   a kernel is undefined.
   * \note ReadWrite and WriteOnly are mutually exclusive.
   *
   * \value ReadOnly This flag specifies that the memory object is a read-only
   * memory object when used inside a kernel.
   * \note Writing to a buffer or image object created with ReadOnly inside a
   * kernel is undefined.
   * \note ReadWrite or WriteOnly and ReadOnly are mutually exclusive.
   */
  enum Access
  {
    ReadWrite = 0x0001,
    WriteOnly = 0x0002,
    ReadOnly = 0x0004
  };

  /** Returns true if this OpenCL memory object is null, false otherwise. */
  bool
  IsNull() const
  {
    return this->m_Id == 0;
  }

  /** Returns the native OpenCL identifier for this memory object. */
  cl_mem
  GetMemoryId() const
  {
    return this->m_Id;
  }

  /** Returns the OpenCL context that created this memory object. */
  OpenCLContext *
  GetContext() const
  {
    return this->m_Context;
  }

  /** Returns the memory object type used to create this object. */
  cl_mem_object_type
  GetMemoryType() const;

  /** Returns the access flags that were used to create this memory object. */
  cl_mem_flags
  GetFlags() const;

  /** Returns actual size of the data store associated with memory object in bytes. */
  std::size_t
  GetSize() const;

  /** Return the host pointer argument value specified when memory object is created.
   * Otherwise a NULL value is returned. */
  void *
  GetHostPointer() const;

  /** Returns map count. The map count returned should be considered immediately stale.
   * It is unsuitable for general use in applications. This feature is provided
   * for debugging. */
  cl_uint
  GetMapCount() const;

  /** Returns memory object reference count. The reference count returned should
   * be considered immediately stale. It is unsuitable for general use in
   * applications. This feature is provided for identifying memory leaks. */
  cl_uint
  GetReferenceCount() const;

  /** Returns the access mode that was used to create this memory object. */
  OpenCLMemoryObject::Access
  GetAccess() const;

  /** Requests a command to unmap a previously mapped region at \a ptr of a memory object.
   * This function will wait until the request has finished if the \a wait is true.
   * The request is executed on the active command queue for context.
   * \sa UnmapAsync(), OpenCLBuffer::Map() */
  void
  Unmap(void * ptr, const bool wait = false);

  /** Requests a command to unmap a previously mapped region at \a ptr of a memory object.
   * The request will be started after all events in \a event_list are finished.
   * Returns an event object that can be used to wait for the request to finish.
   * The request is executed on the active command queue for context().
   * \sa Unmap(), OpenCLBuffer::MapAsync() */
  OpenCLEvent
  UnmapAsync(void * ptr, const OpenCLEventList & event_list = OpenCLEventList());

  /** Registers a user callback function with a memory object.
   * Each call to \c{clSetMemObjectDestructorCallback} registers the specified
   * user callback function on a callback stack associated with memobj.
   * The registered user callback functions are called in the reverse order in
   * which they were registered. The user callback functions are called and then
   * the memory objects resources are freed and the memory object is deleted.
   * This provides a mechanism for the application (and libraries) using memobj
   * to be notified when the memory referenced by host_ptr, specified when the
   * memory object is created and used as the storage bits for the memory
   * object, can be reused or freed. */
  cl_int
  SetDestructorCallback(void(CL_CALLBACK * pfn_notify)(cl_mem, void *), void * user_data = nullptr);

protected:
  /** Helper function to pass cl_mem \a id. */
  void
  SetId(OpenCLContext * context, const cl_mem id);

  /** Helper function to get cl_map_flags from access. */
  cl_map_flags
  GetMapFlags(const OpenCLMemoryObject::Access access);

private:
  OpenCLContext * m_Context;
  cl_mem          m_Id;

  OpenCLMemoryObject(const Self & other) = delete;
  const Self &
  operator=(const Self &) = delete;
};

/** Operator ==
 * Returns true if \a lhs OpenCL memory object is the same as \a rhs, false otherwise.
 * \sa operator!= */
bool ITKOpenCL_EXPORT
     operator==(const OpenCLMemoryObject & lhs, const OpenCLMemoryObject & rhs);

/** Operator !=
 * Returns true if \a lhs OpenCL memory object identifier is not the same as \a rhs, false otherwise.
 * \sa operator== */
bool ITKOpenCL_EXPORT
     operator!=(const OpenCLMemoryObject & lhs, const OpenCLMemoryObject & rhs);

/** Stream out operator for OpenCLMemoryObject */
template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const OpenCLMemoryObject & memoryObject)
{
  if (memoryObject.IsNull())
  {
    strm << "OpenCLMemoryObject(null)";
    return strm;
  }

  const char indent = ' ';

  strm << "OpenCLMemoryObject\n"
       << indent << "Id: " << memoryObject.GetMemoryId() << '\n'
       << indent << "Context: " << memoryObject.GetContext() << '\n'
       << indent << "Memory type: " << memoryObject.GetMemoryType() << '\n'
       << indent << "Flags: " << memoryObject.GetFlags() << '\n'
       << indent << "Size: " << memoryObject.GetSize() << '\n'
       << indent << "Map count: " << memoryObject.GetMapCount() << '\n'
       << indent << "Reference count: " << memoryObject.GetReferenceCount() << '\n'
       << indent << "Host pointer: " << memoryObject.GetHostPointer() << '\n'
       << indent << "Access: ";

  switch (memoryObject.GetAccess())
  {
    case OpenCLMemoryObject::ReadWrite:
      strm << "Read Write";
      break;
    case OpenCLMemoryObject::WriteOnly:
      strm << "Write Only";
      break;
    case OpenCLMemoryObject::ReadOnly:
      strm << "Read Only";
      break;
    default:
      strm << "Unknown";
      break;
  }

  strm << std::endl;

  return strm;
}


} // end namespace itk

#endif /* itkOpenCLMemoryObject_h */

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
#ifndef itkOpenCLBuffer_h
#define itkOpenCLBuffer_h

#include "itkOpenCLMemoryObject.h"
#include "itkOpenCLEvent.h"

namespace itk
{
/** \class OpenCLBuffer
 * \brief The OpenCLBuffer class represents an OpenCL buffer object.
 *
 * \ingroup OpenCL
 * \sa OpenCLMemoryObject, OpenCLVector
 */

// Forward declaration
class OpenCLImage;

class ITKOpenCL_EXPORT OpenCLBuffer : public OpenCLMemoryObject
{
public:
  /** Standard class typedefs. */
  using Self = OpenCLBuffer;

  /** Constructs a null OpenCL buffer object. */
  OpenCLBuffer() = default;

  /** Constructs an OpenCL buffer object that is initialized with the
   * native OpenCL identifier \a id, and associates it with \a context.
   * This class will take over ownership of \a id and will release
   * it in the destructor. */
  OpenCLBuffer(OpenCLContext * context, const cl_mem id);

  /** Constructs a copy of \a other. */
  OpenCLBuffer(const OpenCLBuffer & other);

  /** Assigns \a other to this object. */
  OpenCLBuffer &
  operator=(const OpenCLBuffer & other);

  /** Reads \a size bytes starting at \a offset into the supplied \a data to
   * the host memory. Returns true if the read was successful, false otherwise.
   * This method does not return until the buffer data has been read into memory
   * pointed to \a data.
   * \sa ReadAsync(), Write(), WriteAsync() */
  bool
  Read(void * data, const std::size_t size, const std::size_t offset = 0);

  /** Asynchronous version of the Read() method.
   * This function will queue the request and return immediately. Returns an
   * OpenCLEvent object that can be used to wait for the request to finish.
   * The request will not start until all of the events in \a event_list
   * have been signaled as completed.
   * \sa Read(), Write(), WriteAsync(), OpenCLEvent::IsComplete() */
  OpenCLEvent
  ReadAsync(void *                  data,
            const std::size_t       size,
            const OpenCLEventList & event_list = OpenCLEventList(),
            const std::size_t       offset = 0);

  /** Writes \a size bytes starting at \a offset from the supplied \a data to
   * the device memory. Returns true if the write was successful, false otherwise.
   * This function will block until the request finishes.
   * This method does not return until the buffer data has been written
   * into memory pointed to \a data.
   * \sa WriteAsync(), Read(), ReadAsync() */
  bool
  Write(const void * data, const std::size_t size, const std::size_t offset = 0);

  /** Asynchronous version of the Write() method.
   * This function will queue the request and return immediately. Returns an
   * OpenCLEvent object that can be used to wait for the request to finish.
   * The request will not start until all of the events in \a event_list
   * have been signaled as completed.
   * \sa Write(), Read(), ReadAsync(), OpenCLEvent::IsComplete() */
  OpenCLEvent
  WriteAsync(const void *            data,
             const std::size_t       size,
             const OpenCLEventList & event_list = OpenCLEventList(),
             const std::size_t       offset = 0);

  /** Reads the bytes defined by \a rect and \a bufferBytesPerLine
   * from this buffer into the supplied \a data array, with a line
   * pitch of \a hostBytesPerLine. Returns true if the read
   * was successful, false otherwise.
   * This function will block until the request finishes.
   * The request is executed on the active command queue for context().
   * \sa ReadRectAsync(), WriteRect() */
  bool
  ReadRect(void *                data,
           const RectangleType & rect,
           const std::size_t     bufferBytesPerLine,
           const std::size_t     hostBytesPerLine);

  /** Asynchronous version of the ReadRect() method.
   * This function will queue the request and return immediately. Returns an
   * OpenCLEvent object that can be used to wait for the request to finish.
   * The request will not start until all of the events in \a event_list
   * have been signaled as completed.
   * \sa ReadRect(), WriteRectAsync(), OpenCLEvent::IsComplete() */
  OpenCLEvent
  ReadRectAsync(void *                  data,
                const RectangleType &   rect,
                const std::size_t       bufferBytesPerLine,
                const std::size_t       hostBytesPerLine,
                const OpenCLEventList & event_list = OpenCLEventList());

  /** Reads the bytes in the 3D region defined by \a origin, \a size,
   * \a bufferBytesPerLine, and \a bufferBytesPerSlice from this buffer
   * into the supplied \a data array, with a line pitch of
   * \a hostBytesPerLine, and a slice pitch of \a hostBytesPerSlice.
   * Returns true if the read was successful, false otherwise.
   * This function will block until the request finishes.
   * The request is executed on the active command queue for context().
   * \sa ReadRectAsync(), WriteRect() */
  bool
  ReadRect(void *            data,
           const std::size_t origin[3],
           const std::size_t size[3],
           const std::size_t bufferBytesPerLine,
           const std::size_t bufferBytesPerSlice,
           const std::size_t hostBytesPerLine,
           const std::size_t hostBytesPerSlice);

  /** Asynchronous version of the ReadRect() method.
   * This function will queue the request and return immediately. Returns an
   * OpenCLEvent object that can be used to wait for the request to finish.
   * The request will not start until all of the events in \a event_list
   * have been signaled as completed.
   * \sa ReadRect(), WriteRectAsync(), OpenCLEvent::IsComplete() */
  OpenCLEvent
  ReadRectAsync(void *                  data,
                const std::size_t       origin[3],
                const std::size_t       size[3],
                const std::size_t       bufferBytesPerLine,
                const std::size_t       bufferBytesPerSlice,
                const std::size_t       hostBytesPerLine,
                const std::size_t       hostBytesPerSlice,
                const OpenCLEventList & event_list = OpenCLEventList());

  /** Writes the bytes at \a data, with a line pitch of \a hostBytesPerLine
   * to the region of this buffer defined by \a rect and \a bufferBytesPerLine.
   * Returns true if the write was successful, false otherwise.
   * This function will block until the request finishes.
   * The request is executed on the active command queue for context().
   * \sa WriteRectAsync(), ReadRect() */
  bool
  WriteRect(const void *          data,
            const RectangleType & rect,
            const std::size_t     bufferBytesPerLine,
            const std::size_t     hostBytesPerLine);

  /** Asynchronous version of the WriteRect() method.
   * This function will queue the request and return immediately. Returns an
   * OpenCLEvent object that can be used to wait for the request to finish.
   * The request will not start until all of the events in \a event_list
   * have been signaled as completed.
   * \sa WriteRect(), ReadRectAsync(), OpenCLEvent::IsComplete() */
  OpenCLEvent
  WriteRectAsync(const void *            data,
                 const RectangleType &   rect,
                 const std::size_t       bufferBytesPerLine,
                 const std::size_t       hostBytesPerLine,
                 const OpenCLEventList & event_list = OpenCLEventList());

  /** Writes the bytes at \a data, with a line pitch of \a hostBytesPerLine,
   * and a slice pitch of \a hostBytesPerSlice, to the 3D region defined
   * by \a origin, \a size, \a bufferBytesPerLine, and \a bufferBytesPerSlice
   * in this buffer. Returns true if the write was successful, false otherwise.
   * This function will block until the request finishes.
   * The request is executed on the active command queue for context().
   * \sa WriteRectAsync(), ReadRect() */
  bool
  WriteRect(const void *      data,
            const std::size_t origin[3],
            const std::size_t size[3],
            const std::size_t bufferBytesPerLine,
            const std::size_t bufferBytesPerSlice,
            const std::size_t hostBytesPerLine,
            const std::size_t hostBytesPerSlice);

  /** Asynchronous version of the WriteRect() method.
   * This function will queue the request and return immediately. Returns an
   * OpenCLEvent object that can be used to wait for the request to finish.
   * The request will not start until all of the events in \a event_list
   * have been signaled as completed.
   * \sa WriteRect(), ReadRectAsync(), OpenCLEvent::IsComplete() */
  OpenCLEvent
  WriteRectAsync(const void *            data,
                 const std::size_t       origin[3],
                 const std::size_t       size[3],
                 const std::size_t       bufferBytesPerLine,
                 const std::size_t       bufferBytesPerSlice,
                 const std::size_t       hostBytesPerLine,
                 const std::size_t       hostBytesPerSlice,
                 const OpenCLEventList & event_list = OpenCLEventList());

  /** Copies the \a size bytes at \a offset in this buffer
   * be copied to \a dst_offset in the buffer \a dest. Returns true
   * if the copy was successful, false otherwise.
   * This function will block until the request finishes.
   * The request is executed on the active command queue for context().
   * \sa CopyToAsync() */
  bool
  CopyToBuffer(const OpenCLBuffer & dest,
               const std::size_t    size,
               const std::size_t    dst_offset = 0,
               const std::size_t    offset = 0);

  /** Asynchronous version of the CopyToBuffer() method.
   * This function will queue the request and return immediately. Returns an
   * OpenCLEvent object that can be used to wait for the request to finish.
   * The request will not start until all of the events in \a event_list
   * have been signaled as completed.
   * \sa CopyTo(), OpenCLEvent::IsComplete() */
  OpenCLEvent
  CopyToBufferAsync(const OpenCLBuffer &    dest,
                    const std::size_t       size,
                    const OpenCLEventList & event_list = OpenCLEventList(),
                    const std::size_t       dst_offset = 0,
                    const std::size_t       offset = 0);

  /** Copies the contents of this buffer, starting at \a origin and range \a region
   * in \a dest. Returns true if the copy was successful, false otherwise.
   * This function will block until the request finishes.
   * The request is executed on the active command queue for context().
   * \sa CopyToAsync() */
  bool
  CopyToImage(const OpenCLImage & dest,
              const OpenCLSize &  origin,
              const OpenCLSize &  region,
              const std::size_t   src_offset = 0);

  /** Asynchronous version of the CopyToImage() method.
   * This function will queue the request and return immediately. Returns an
   * OpenCLEvent object that can be used to wait for the request to finish.
   * The request will not start until all of the events in \a event_list
   * have been signaled as completed.
   * \sa CopyTo(), OpenCLEvent::IsComplete() */
  OpenCLEvent
  CopyToImageAsync(const OpenCLImage &     dest,
                   const OpenCLSize &      origin,
                   const OpenCLSize &      region,
                   const OpenCLEventList & event_list = OpenCLEventList(),
                   const std::size_t       src_offset = 0);

  /** Copies the contents of \a rect within this buffer to \a dest,
   * starting at \a destPoint. The source and destination line pitch
   * values are given by \a bufferBytesPerLine and \a destBytesPerLine
   * respectively. Returns true if the copy was successful, false otherwise.
   * This function will block until the request finishes.
   * The request is executed on the active command queue for context().
   * \sa CopyToRectAsync() */
  bool
  CopyToRect(const OpenCLBuffer &  dest,
             const RectangleType & rect,
             const PointType &     destPoint,
             const std::size_t     bufferBytesPerLine,
             std::size_t           destBytesPerLine);

  /** Asynchronous version of the CopyToRect() method.
   * This function will queue the request and return immediately. Returns an
   * OpenCLEvent object that can be used to wait for the request to finish.
   * The request will not start until all of the events in \a event_list
   * have been signaled as completed.
   * \sa CopyToRect(), OpenCLEvent::IsComplete() */
  OpenCLEvent
  CopyToRectAsync(const OpenCLBuffer &    dest,
                  const RectangleType &   rect,
                  const PointType &       destPoint,
                  const std::size_t       bufferBytesPerLine,
                  const std::size_t       destBytesPerLine,
                  const OpenCLEventList & event_list = OpenCLEventList());

  /** Copies the 3D rectangle defined by \a origin and \a size within
   * this buffer to \a destOrigin within \a dest. The source and destination
   * pitch values are given by \a bufferBytesPerLine, \a bufferBytesPerSlice,
   * \a destBytesPerLine, and \a destBytesPerSlice. Returns true if
   * the copy was successful, false otherwise.
   * This function will block until the request finishes.
   * The request is executed on the active command queue for context().
   * \sa CopyToRectAsync() */
  bool
  CopyToRect(const OpenCLBuffer & dest,
             const std::size_t    origin[3],
             const std::size_t    size[3],
             const std::size_t    destOrigin[3],
             const std::size_t    bufferBytesPerLine,
             const std::size_t    bufferBytesPerSlice,
             const std::size_t    destBytesPerLine,
             const std::size_t    destBytesPerSlice);

  /** Asynchronous version of the CopyToRect() method.
   * This function will queue the request and return immediately. Returns an
   * OpenCLEvent object that can be used to wait for the request to finish.
   * The request will not start until all of the events in \a event_list
   * have been signaled as completed.
   * \sa CopyToRectAsync(), OpenCLEvent::IsComplete() */
  OpenCLEvent
  CopyToRectAsync(const OpenCLBuffer &    dest,
                  const std::size_t       origin[3],
                  const std::size_t       size[3],
                  const std::size_t       destOrigin[3],
                  const std::size_t       bufferBytesPerLine,
                  const std::size_t       bufferBytesPerSlice,
                  const std::size_t       destBytesPerLine,
                  const std::size_t       destBytesPerSlice,
                  const OpenCLEventList & event_list = OpenCLEventList());

  /** Maps the \a size bytes starting at \a offset in this buffer
   * into host memory for the specified \a access mode. Returns a
   * pointer to the mapped region.
   * This function will block until the request finishes.
   * The request is executed on the active command queue for context().
   * \sa MapAsync(), Unmap() */
  void *
  Map(const OpenCLMemoryObject::Access access, const std::size_t size, const std::size_t offset = 0);

  /** Asynchronous version of the Map() method.
   * This function will queue the request and return immediately. Returns an
   * OpenCLEvent object that can be used to wait for the request to finish.
   * The request will not start until all of the events in \a event_list
   * have been signaled as completed.
   * \sa Map(), UnmapAsync(), OpenCLEvent::IsComplete() */
  OpenCLEvent
  MapAsync(void **                          ptr,
           const OpenCLMemoryObject::Access access,
           const std::size_t                size,
           const OpenCLEventList &          event_list = OpenCLEventList(),
           const std::size_t                offset = 0);

  /** \overload
   * Maps the entire contents of this buffer into host memory for
   * the specified \a access mode. Returns a pointer to the mapped region. */
  void *
  Map(const OpenCLMemoryObject::Access access);

  /** Creates a new buffer that refers to the \a size bytes,
   * starting at \a offset within this buffer. The data in
   * the new buffer will be accessed according to \a access.
   * this function will return a null buffer.
   * \sa GetParentBuffer(), GetOffset() */
  OpenCLBuffer
  CreateSubBuffer(const OpenCLMemoryObject::Access access, const std::size_t size, const std::size_t offset = 0);

  /** Returns the parent of this buffer if it is a sub-buffer; null otherwise.
   * \sa CreateSubBuffer(), GetOffset()  */
  OpenCLBuffer
  GetParentBuffer() const;

  /** Returns the offset of this buffer within its parentBuffer()
   * if it is a sub-buffer; zero otherwise.
   * \sa CreateSubBuffer(), GetParentBuffer() */
  std::size_t
  GetOffset() const;
};

} // end namespace itk

#endif /* itkOpenCLBuffer_h */

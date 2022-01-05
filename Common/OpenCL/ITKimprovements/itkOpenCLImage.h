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
#ifndef itkOpenCLImage_h
#define itkOpenCLImage_h

#include "itkOpenCLMemoryObject.h"
#include "itkOpenCLImageFormat.h"
#include "itkOpenCLEvent.h"
#include "itkOpenCLSize.h"

namespace itk
{
/** \class OpenCLImage
 * \brief The OpenCLImage class represents an image object is used to store a
 * one, two or three dimensional texture, frame-buffer or image.
 *
 * \ingroup OpenCL
 */

// Forward declaration
class OpenCLBuffer;

class ITKOpenCL_EXPORT OpenCLImage : public OpenCLMemoryObject
{
public:
  /** Standard class typedefs. */
  using Self = OpenCLImage;
  using Superclass = OpenCLMemoryObject;

  /** Constructs a null OpenCL image object. */
  OpenCLImage() = default;

  /** Constructs a OpenCL image object that is initialized with the
   * native OpenCL identifier \a id, and associates it with \a context.
   * This class will take over ownership of \a id and will release
   * it in the destructor. */
  OpenCLImage(OpenCLContext * context, const cl_mem id)
    : OpenCLMemoryObject(context, id)
  {}

  /** Constructs a copy of \a other. */
  OpenCLImage(const OpenCLImage & other);

  /** Assigns \a other to this object. */
  OpenCLImage &
  operator=(const OpenCLImage & other);

  /** Return image format descriptor specified when image is created. */
  OpenCLImageFormat
  GetFormat() const;

  /** Return size of each element of the image memory object given by image.
   * An element is made up of n channels. The value of n is given in
   * cl_image_format descriptor. */
  std::size_t
  GetElementSizeInBytes() const;

  /** Return size in bytes of a row of elements of the image object given by image. */
  std::size_t
  GetRowSizeInBytes() const;

  /** Return calculated slice pitch in bytes of a 2D slice for the 3D image object
   * or size of each image in a 1D or 2D image array given by image.
   * \note For a 1D image, 1D image buffer and 2D image object return 0. */
  std::size_t
  GetSliceSizeInBytes() const;

  /** Returns the dimension for this image, 1, 2, or 3. */
  std::size_t
  GetDimension() const;

  /** Return width of image in pixels.
   *\sa GetHeight(), GetDepth() */
  std::size_t
  GetWidth() const;

  /** Return height of image in pixels.
   * \note For a 1D image, 1D image buffer and 1D image array object, height = 0.
   * \sa GetWidth(), GetDepth() */
  std::size_t
  GetHeight() const;

  /** Return depth of the image in pixels.
   * \note For a 1D image, 1D image buffer, 2D image or 1D and
   * 2D image array object, depth = 0
   * \sa GetWidth(), GetHeight() */
  std::size_t
  GetDepth() const;

  /** Reads from an image or image array object to host memory,
   * starting at \a origin and range \a region into \a data.
   * Returns true if the read was successful, false otherwise.
   * This method does not return until the buffer data has been read into memory
   * pointed to \a data.
   * \sa ReadAsync(), Write(), WriteAsync() */
  bool
  Read(void *             data,
       const OpenCLSize & origin,
       const OpenCLSize & region,
       const std::size_t  rowPitch = 0,
       const std::size_t  slicePitch = 0);

  /** Asynchronous version of the Read() method.
   * This function will queue the request and return immediately. Returns an
   * OpenCLEvent object that can be used to wait for the request to finish.
   * The request will not start until all of the events in \a event_list
   * have been signaled as completed.
   * \sa Read(), Write(), WriteAsync() */
  OpenCLEvent
  ReadAsync(void *                  data,
            const OpenCLSize &      origin,
            const OpenCLSize &      region,
            const OpenCLEventList & event_list = OpenCLEventList(),
            const std::size_t       rowPitch = 0,
            const std::size_t       slicePitch = 0);

  /** Write an image or image array object from host memory,
   * starting at \a origin and range \a region into \a data.
   * Returns true if the read was successful, false otherwise.
   * This method does not return until the buffer data has been written
   * into memory pointed to \a data.
   * \sa Read(), ReadAsync(), WriteAsync() */
  bool
  Write(const void *       data,
        const OpenCLSize & origin,
        const OpenCLSize & region,
        const std::size_t  rowPitch = 0,
        const std::size_t  slicePitch = 0);

  /** Asynchronous version of the Write() method.
   * This function will queue the request and return immediately. Returns an
   * OpenCLEvent object that can be used to wait for the request to finish.
   * The request will not start until all of the events in \a event_list
   * have been signaled as completed.
   * \sa Read(), ReadAsync(), Write() */
  OpenCLEvent
  WriteAsync(const void *            data,
             const OpenCLSize &      origin,
             const OpenCLSize &      region,
             const OpenCLEventList & event_list = OpenCLEventList(),
             const std::size_t       rowPitch = 0,
             const std::size_t       slicePitch = 0);

  /** Map a region of an image object starting at \a origin and range \a region
   * into the host address space for the specified \a access mode and returns
   * a pointer to this mapped region.
   * This method does not return until the specified region in image is mapped
   * into the host address space and the application can access the contents
   * of the mapped region.
   * \sa Read(), ReadAsync(), Write(), WriteAsync(), MapAsync() */
  void *
  Map(const OpenCLMemoryObject::Access access,
      const OpenCLSize &               origin,
      const OpenCLSize &               region,
      std::size_t *                    rowPitch = 0,
      std::size_t *                    slicePitch = 0);

  /** Asynchronous version of the Map() method.
   * This function will queue the request and return immediately. Returns an
   * OpenCLEvent object that can be used to wait for the request to finish.
   * The request will not start until all of the events in \a event_list
   * have been signaled as completed.
   * \sa Read(), ReadAsync(), Write(), WriteAsync(), Map() */
  OpenCLEvent
  MapAsync(void **                          data,
           const OpenCLMemoryObject::Access access,
           const OpenCLSize &               origin,
           const OpenCLSize &               region,
           const OpenCLEventList &          event_list = OpenCLEventList(),
           std::size_t *                    rowPitch = 0,
           std::size_t *                    slicePitch = 0);

  /** Copies a region of an image object starting at \a origin and range \a region
   * to \a destOrigin in \a dest. Returns true if the copy was successful,
   * false otherwise. This function will block until the request finishes.
   * The request is executed on the active command queue for context().
   * \sa CopyAsync() */
  bool
  Copy(const OpenCLImage & dest, const OpenCLSize & origin, const OpenCLSize & region, const OpenCLSize & destOrigin);

  /** Asynchronous version of the Copy() method.
   * This function will queue the request and return immediately. Returns an
   * OpenCLEvent object that can be used to wait for the request to finish.
   * The request will not start until all of the events in \a event_list
   * have been signaled as completed.
   * \sa Copy() */
  OpenCLEvent
  CopyAsync(const OpenCLImage &     dest,
            const OpenCLSize &      origin,
            const OpenCLSize &      region,
            const OpenCLSize &      destOrigin,
            const OpenCLEventList & event_list = OpenCLEventList());

  /** Copies a region of an image object starting at \a origin and range \a region
   * to \a destOrigin in \a dest. Returns true if the copy was successful,
   * false otherwise. This function will block until the request finishes.
   * The request is executed on the active command queue for context().
   * \sa CopyAsync() */
  bool
  Copy(const OpenCLBuffer & dest,
       const OpenCLSize &   origin,
       const OpenCLSize &   region,
       const std::size_t    dst_offset = 0);

  /** Asynchronous version of the Copy() method.
   * This function will queue the request and return immediately. Returns an
   * OpenCLEvent object that can be used to wait for the request to finish.
   * The request will not start until all of the events in \a event_list
   * have been signaled as completed.
   * \sa Copy() */
  OpenCLEvent
  CopyAsync(const OpenCLBuffer &    dest,
            const OpenCLSize &      origin,
            const OpenCLSize &      region,
            const OpenCLEventList & event_list = OpenCLEventList(),
            const std::size_t       dst_offset = 0);

#ifdef CL_VERSION_1_2
  /** */
  static void
  SetImageDescription(cl_image_desc & imageDescription, const OpenCLImageFormat & format, const OpenCLSize & size);

#endif

protected:
  /** Get information specific to an image object created with clCreateImage */
  std::size_t
  GetImageInfo(const cl_image_info name) const;

  /** Set the image origin information */
  void
  SetOrigin(std::size_t * origin_t, const OpenCLSize & origin) const;

  /** Set the image region information */
  void
  SetRegion(std::size_t * region_t, const OpenCLSize & region) const;

  /** Set the image size information */
  void
  SetSize(std::size_t * region_t, const OpenCLSize & region, const std::size_t value) const;

  /** friends from OpenCL core */
  friend class OpenCLBuffer;
};

/** Stream out operator for OpenCLImage2D */
template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const OpenCLImage & image)
{
  if (image.IsNull())
  {
    strm << "OpenCLImage(null)";
    return strm;
  }

  const char indent = ' ';

  strm << "OpenCLImage\n"
       << indent << "Element size(bytes): " << image.GetElementSizeInBytes() << '\n'
       << indent << "Row size(bytes): " << image.GetRowSizeInBytes() << '\n'
       << indent << "Slice size(bytes): " << image.GetSliceSizeInBytes() << '\n'
       << indent << "Dimension: " << image.GetDimension() << '\n'
       << indent << "Width: " << image.GetWidth() << '\n'
       << indent << "Height: " << image.GetHeight() << '\n'
       << indent << "Depth: " << image.GetDepth() << std::endl;

  // Stream OpenCLMemoryObject
  const OpenCLMemoryObject & memObj = image;
  strm << memObj;

  return strm;
}


} // end namespace itk

#endif /* itkOpenCLImage_h */

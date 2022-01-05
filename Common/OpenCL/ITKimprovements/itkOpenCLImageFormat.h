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
#ifndef itkOpenCLImageFormat_h
#define itkOpenCLImageFormat_h

#include "itkOpenCL.h"
#include <ostream>

namespace itk
{
/** \class OpenCLImageFormat
 * \brief The OpenCLImageFormat class represents the format of a OpenCLImage.
 * \ingroup OpenCL
 */

// Forward declaration
class OpenCLContext;

class ITKOpenCL_EXPORT OpenCLImageFormat
{
public:
  /** Standard class typedefs. */
  using Self = OpenCLImageFormat;

  /** \enum OpenCLImageFormat::ImageType
   * This enum defines the image type for the OpenCL image object.
   * \value BUFFER The OpenCL buffer.
   * \value IMAGE2D The 2D image
   * \value IMAGE3D The 2D image.
   * \value IMAGE2D_ARRAY The 1D image array.
   * \value IMAGE1D The The 1D image.
   * \value IMAGE1D_ARRAY The 1D array.
   * \value IMAGE1D_BUFFER The 1D buffer. */
  enum ImageType
  {
    BUFFER = 0x10F0,
    IMAGE2D = 0x10F1,
    IMAGE3D = 0x10F2,
    IMAGE2D_ARRAY = 0x10F3,
    IMAGE1D = 0x10F4,
    IMAGE1D_ARRAY = 0x10F5,
    IMAGE1D_BUFFER = 0x10F6
  };

  /** \enum OpenCLImageFormat::ChannelOrder
   * This enum specifies the number of channels and the channel layout i.e.
   * the memory layout in which channels are stored in the image. Valid values
   * are described in the table below:
   *
   * \value R Single red channel.
   * \value A Single alpha channel.
   * \value RG Red then green channel.
   * \value RA Red then alpha channel.
   * \value RGB Red, green, and blue channels.
     \note This format can only be used if channel data type is
           CL_UNORM_SHORT_565, CL_UNORM_SHORT_555 or CL_UNORM_INT101010.
   * \value RGBA Red, green, blue, then alpha channels.
   * \value BGRA Blue, green, red, then alpha channels.
   * \value ARGB Alpha, red, green, then blue channels.
   * \value INTENSITY Single intensity channel.
   * \note This format can only be used if channel data type is
   *       CL_UNORM_INT8, CL_UNORM_INT16, CL_SNORM_INT8, CL_SNORM_INT16,
   *       CL_HALF_FLOAT, or CL_FLOAT.
   * \value LUMINANCE Single LUMINANCE channel.
   * \note  This format can only be used if channel data type is
   *        CL_UNORM_INT8, CL_UNORM_INT16, CL_SNORM_INT8, CL_SNORM_INT16,
   *        CL_HALF_FLOAT, or CL_FLOAT.
   * \value Rx Similar to CL_R except alpha = 0 at edges.
   * \value RGx Similar to CL_RG except alpha = 0 at edges.
   * \value RGBx Similar to CL_RGB except alpha = 0 at edges.
   * \note This format can only be used if channel data type is
   *       CL_UNORM_SHORT_565, CL_UNORM_SHORT_555 or CL_UNORM_INT101010.
   * \value DEPTH Allow a CL image to be created from a GL depth texture.
   * \value DEPTH_STENCIL Allow a CL image to be created from a GL depth-stencil texture. */
  enum ChannelOrder
  {
    R = 0x10B0,
    A = 0x10B1,
    RG = 0x10B2,
    RA = 0x10B3,
    RGB = 0x10B4,
    RGBA = 0x10B5,
    BGRA = 0x10B6,
    ARGB = 0x10B7,
    INTENSITY = 0x10B8,
    LUMINANCE = 0x10B9,
    Rx = 0x10BA,
    RGx = 0x10BB,
    RGBx = 0x10BC,
    DEPTH = 0x10BD,
    DEPTH_STENCIL = 0x10BE
  };

  /** \enum OpenCLImageFormat::ChannelType
   * This enum describes the size of the channel data type.
   * The number of bits per element determined by the image_channel_data_type
   * and image_channel_order must be a power of two. The list of supported
   * values is described in the table below:
   *
   * \value SNORM_INT8 Each channel component is a normalized signed 8-bit integer value.
   * \value SNORM_INT16 Each channel component is a normalized signed 16-bit integer value.
   * \value UNORM_INT8 Each channel component is a normalized unsigned 8-bit integer value.
   * \value UNORM_INT16 Each channel component is a normalized unsigned 16-bit integer value.
   * \value UNORM_SHORT_565 Represents a normalized 5-6-5 3-channel RGB image.
   * \note The channel order must be CL_RGB or CL_RGBx.
   * \value UNORM_SHORT_555 Represents a normalized x-5-5-5 4-channel xRGB image.
   * \note The channel order must be CL_RGB or CL_RGBx.
   * \value UNORM_INT_101010 Represents a normalized x-10-10-10 4-channel xRGB image.
   * \note The channel order must be CL_RGB or CL_RGBx.
   * \value SIGNED_INT8 Each channel component is an unnormalized signed 8-bit integer value.
   * \value SIGNED_INT16 Each channel component is an unnormalized signed 16-bit integer value.
   * \value SIGNED_INT32 Each channel component is an unnormalized signed 32-bit integer value.
   * \value UNSIGNED_INT8 Each channel component is an unnormalized unsigned 8-bit integer value.
   * \value UNSIGNED_INT16 Each channel component is an unnormalized unsigned 16-bit integer value.
   * \value UNSIGNED_INT32 Each channel component is an unnormalized unsigned 32-bit integer value.
   * \value HALF_FLOAT Each channel component is a 16-bit half-float value.
   * \value FLOAT Each channel component is a single precision floating-point value.
   * \value UNORM_INT24 Each channel component is stored as an unsigned normalized 24-bit value. */
  enum ChannelType
  {
    SNORM_INT8 = 0x10D0,
    SNORM_INT16 = 0x10D1,
    UNORM_INT8 = 0x10D2,
    UNORM_INT16 = 0x10D3,
    UNORM_SHORT_565 = 0x10D4,
    UNORM_SHORT_555 = 0x10D5,
    UNORM_INT_101010 = 0x10D6,
    SIGNED_INT8 = 0x10D7,
    SIGNED_INT16 = 0x10D8,
    SIGNED_INT32 = 0x10D9,
    UNSIGNED_INT8 = 0x10DA,
    UNSIGNED_INT16 = 0x10DB,
    UNSIGNED_INT32 = 0x10DC,
    HALF_FLOAT = 0x10DD,
    FLOAT = 0x10DE,
    UNORM_INT24 = 0x10DF
  };

  /** Constructs a null OpenCL image format descriptor.
   * \sa IsNull() */
  OpenCLImageFormat();

  /** Constructs an OpenCL image format descriptor from
   * \a channelOrder and \a channelType. */
  OpenCLImageFormat(const OpenCLImageFormat::ChannelOrder channelOrder,
                    const OpenCLImageFormat::ChannelType  channelType);

  /** Constructs an OpenCL image format descriptor from
   * \a imageType \a channelOrder and \a channelType. */
  OpenCLImageFormat(const OpenCLImageFormat::ImageType    imageType,
                    const OpenCLImageFormat::ChannelOrder channelOrder,
                    const OpenCLImageFormat::ChannelType  channelType);

  /** Returns true if this OpenCL image format descriptor is null. */
  bool
  IsNull() const;

  /** Returns the image type in this OpenCL image format.
   *\sa GetChannelOrder(), GetChannelType() */
  OpenCLImageFormat::ImageType
  GetImageType() const;

  /** Returns the order of channels in this OpenCL image format.
   *\sa GetChannelType(), GetImageType() */
  OpenCLImageFormat::ChannelOrder
  GetChannelOrder() const;

  /** Returns the representation type for channels in this OpenCL image format.
   * \sa GetChannelOrder(), GetImageType() */
  OpenCLImageFormat::ChannelType
  GetChannelType() const;

private:
  cl_mem_object_type m_ImageType;
  cl_image_format    m_Format;

  /** friends from OpenCL core */
  friend class OpenCLContext;
  friend class OpenCLImage;
};

/** Operator ==
 * Returns true if \a lhs OpenCL image format is the same as \a rhs, false otherwise.
 * \sa operator!= */
bool ITKOpenCL_EXPORT
     operator==(const OpenCLImageFormat & lhs, const OpenCLImageFormat & rhs);

/** Operator !=
 * Returns true if \a lhs OpenCL image format is not the same as \a rhs, false otherwise.
 * \sa operator== */
bool ITKOpenCL_EXPORT
     operator!=(const OpenCLImageFormat & lhs, const OpenCLImageFormat & rhs);

/** Stream out operator for OpenCLImageFormat */
template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const OpenCLImageFormat & format)
{
  if (format.IsNull())
  {
    strm << "OpenCLImageFormat(null)";
    return strm;
  }

  strm << "OpenCLImageFormat(";
  switch (int(format.GetImageType()))
  {
    case OpenCLImageFormat::BUFFER:
      strm << "BUFFER, ";
      break;
    case OpenCLImageFormat::IMAGE2D:
      strm << "IMAGE2D, ";
      break;
    case OpenCLImageFormat::IMAGE3D:
      strm << "IMAGE3D, ";
      break;
    case OpenCLImageFormat::IMAGE2D_ARRAY:
      strm << "IMAGE2D_ARRAY, ";
      break;
    case OpenCLImageFormat::IMAGE1D:
      strm << "IMAGE1D, ";
      break;
    case OpenCLImageFormat::IMAGE1D_ARRAY:
      strm << "IMAGE1D_ARRAY, ";
      break;
    case OpenCLImageFormat::IMAGE1D_BUFFER:
      strm << "IMAGE1D_BUFFER, ";
      break;
    default:
      strm << int(format.GetImageType()) << ", ";
      break;
  }
  switch (int(format.GetChannelOrder()))
  {
    case OpenCLImageFormat::R:
      strm << "R, ";
      break;
    case OpenCLImageFormat::A:
      strm << "A, ";
      break;
    case OpenCLImageFormat::RG:
      strm << "RG, ";
      break;
    case OpenCLImageFormat::RA:
      strm << "RA, ";
      break;
    case OpenCLImageFormat::RGB:
      strm << "RGB, ";
      break;
    case OpenCLImageFormat::RGBA:
      strm << "RGBA, ";
      break;
    case OpenCLImageFormat::BGRA:
      strm << "BGRA, ";
      break;
    case OpenCLImageFormat::ARGB:
      strm << "ARGB, ";
      break;
    case OpenCLImageFormat::INTENSITY:
      strm << "INTENSITY, ";
      break;
    case OpenCLImageFormat::LUMINANCE:
      strm << "LUMINANCE, ";
      break;
    case OpenCLImageFormat::Rx:
      strm << "Rx, ";
      break;
    case OpenCLImageFormat::RGx:
      strm << "RGx, ";
      break;
    case OpenCLImageFormat::RGBx:
      strm << "RGBx, ";
      break;
    case OpenCLImageFormat::DEPTH:
      strm << "DEPTH, ";
      break;
    case OpenCLImageFormat::DEPTH_STENCIL:
      strm << "DEPTH_STENCIL, ";
      break;
    default:
      strm << int(format.GetChannelOrder()) << ", ";
      break;
  }
  switch (int(format.GetChannelType()))
  {
    case OpenCLImageFormat::SNORM_INT8:
      strm << "SNORM_INT8";
      break;
    case OpenCLImageFormat::SNORM_INT16:
      strm << "SNORM_INT16";
      break;
    case OpenCLImageFormat::UNORM_INT8:
      strm << "UNORM_INT8";
      break;
    case OpenCLImageFormat::UNORM_INT16:
      strm << "UNORM_INT16";
      break;
    case OpenCLImageFormat::UNORM_SHORT_565:
      strm << "UNORM_SHORT_565";
      break;
    case OpenCLImageFormat::UNORM_SHORT_555:
      strm << "UNORM_SHORT_555";
      break;
    case OpenCLImageFormat::UNORM_INT_101010:
      strm << "UNORM_INT_101010";
      break;
    case OpenCLImageFormat::SIGNED_INT8:
      strm << "SIGNED_INT8";
      break;
    case OpenCLImageFormat::SIGNED_INT16:
      strm << "SIGNED_INT16";
      break;
    case OpenCLImageFormat::SIGNED_INT32:
      strm << "SIGNED_INT32";
      break;
    case OpenCLImageFormat::UNSIGNED_INT8:
      strm << "UNSIGNED_INT8";
      break;
    case OpenCLImageFormat::UNSIGNED_INT16:
      strm << "UNSIGNED_INT16";
      break;
    case OpenCLImageFormat::UNSIGNED_INT32:
      strm << "UNSIGNED_INT32";
      break;
    case OpenCLImageFormat::HALF_FLOAT:
      strm << "HALF_FLOAT";
      break;
    case OpenCLImageFormat::FLOAT:
      strm << "FLOAT";
      break;
    case OpenCLImageFormat::UNORM_INT24:
      strm << "UNORM_INT24";
      break;
    default:
      strm << int(format.GetChannelType());
      break;
  }

  strm << ')' << std::endl;
  return strm;
}


} // end namespace itk

#endif /* itkOpenCLImageFormat_h */

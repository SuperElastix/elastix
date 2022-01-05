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
#ifndef itkOpenCLSampler_h
#define itkOpenCLSampler_h

#include "itkOpenCL.h"
#include <ostream>

namespace itk
{
/** \class OpenCLSampler
 * \brief The OpenCLSampler class represents an OpenCL sampler object.
 *
 * A sampler object describes how to sample an image when the image is read in
 * the kernel. The built-in functions to read from an image in a kernel take a
 * sampler as an argument. The sampler arguments to the image read function can
 * be sampler objects created using OpenCL functions and passed as argument
 * values to the kernel or can be samplers declared inside a kernel. In this
 * section we discuss how sampler objects are created using OpenCL functions.
 *
 * The GetFilterMode() specifies the type of filter that must be applied when
 * reading an image. This can be \c{CL_FILTER_NEAREST}, or \c{CL_FILTER_LINEAR}.
 *
 * The GetAddressingMode() specifies how out-of-range image coordinates are
 * handled when reading from an image. This can be set to
 * \c{CL_ADDRESS_MIRRORED_REPEAT}, \c{CL_ADDRESS_REPEAT},
 * \c{CL_ADDRESS_CLAMP_TO_EDGE}, \c{CL_ADDRESS_CLAMP} and \c{CL_ADDRESS_NONE}.
 *
 * Samplers are created using OpenCLContext::CreateSampler(), as follows:
 *
 * \code
 * OpenCLSampler sampler = context.CreateSampler
 *     (false, OpenCLSampler::ClampToEdge, OpenCLSampler::Linear);
 * \endcode
 *
 * Samplers can also be defined as literals in the OpenCL kernel
 * source code, which avoids the need to create an explicit
 * OpenCLSampler value:
 *
 * \code
 * __constant sampler_t imageSampler =
 * CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
 * \endcode
 *
 * The main advantage of OpenCLSampler over literal sampler values
 * is that OpenCLSampler allows the pixel derivation strategy to be
 * modified at runtime.
 *
 * \ingroup OpenCL
 * \sa OpenCLContext
 */

// Forward declaration
class OpenCLContext;

class ITKOpenCL_EXPORT OpenCLSampler
{
public:
  /** Standard class typedefs. */
  using Self = OpenCLSampler;

  /** Constructs a null OpenCL sampler object. */
  OpenCLSampler()
    : m_Context(0)
    , m_Id(0)
  {}

  /** Constructs an OpenCL sampler object from the native identifier \a id.
   * This class takes over ownership of \a id and will release it in
   * the destructor. The sampler \a id will be associated with \a context. */
  OpenCLSampler(OpenCLContext * context, cl_sampler id)
    : m_Context(context)
    , m_Id(id)
  {}

  /** Constructs a copy of \a other. The \c{clRetainSampler()} function
   * will be called to update the reference count on GetSamplerId(). */
  OpenCLSampler(const OpenCLSampler & other);

  /** Releases this OpenCL sampler object by calling \c{clReleaseSampler()}. */
  ~OpenCLSampler();

  /** Assigns \a other to this OpenCL sampler object. The current samplerId()
   * will be released with \c{clReleaseSampler()}, and the new samplerId()
   * will be retained with \c{clRetainSampler()}. */
  OpenCLSampler &
  operator=(const OpenCLSampler & other);

  /** \enum OpenCLSampler::AddressingMode
   * This enum specifies how to handle out-of-range image co-ordinates
   * when reading from an image in OpenCL.
   * \value None No special handling of out-of-range co-ordinates.
   * \value ClampToEdge Out-of-range requests clamp to the edge pixel value.
   * \value Clamp Out-of-range requests clamp to the image extents.
   * \value Repeat Repeats the image in a cycle.
   * \value Mirrored Repeats the image in a cycle. */
  enum AddressingMode
  {
    None = 0x1130,          // CL_ADDRESS_NONE
    ClampToEdge = 0x1131,   // CL_ADDRESS_CLAMP_TO_EDGE
    Clamp = 0x1132,         // CL_ADDRESS_CLAMP
    Repeat = 0x1133,        // CL_ADDRESS_REPEAT
    MirroredRepeat = 0x1134 // CL_ADDRESS_MIRRORED_REPEAT
  };

  /** \enum OpenCLSampler::FilterMode
   * This enum defines the type of filter to apply when reading from
   * an image in OpenCL.
   * \value Nearest Use the color of the nearest pixel.
   * \value Linear Interpolate linearly between pixel colors to generate
   * intermediate pixel colors. */
  enum FilterMode
  {
    Nearest = 0x1140, // CL_FILTER_NEAREST
    Linear = 0x1141   // CL_FILTER_LINEAR
  };

  /** Returns true if this OpenCL sampler object is null, false otherwise.*/
  bool
  IsNull() const
  {
    return this->m_Id == 0;
  }

  /** Returns true if this sampler is using normalized co-ordinates, false otherwise.
   * \sa GetAddressingMode(), GetFilterMode() */
  bool
  GetNormalizedCoordinates() const;

  /** Returns the addressing mode for out-of-range co-ordinates
   * when reading from an image in OpenCL.
   * \sa GetNormalizedCoordinates(), GetFilterMode() */
  OpenCLSampler::AddressingMode
  GetAddressingMode() const;

  /** Returns the type of filter to apply when reading from an image in OpenCL.
   * \sa GetNormalizedCoordinates(), GetAddressingMode() */
  OpenCLSampler::FilterMode
  GetFilterMode() const;

  /** Returns the native OpenCL identifier for this sampler;
   * or 0 if the sampler is null. */
  cl_sampler
  GetSamplerId() const
  {
    return this->m_Id;
  }

  /** Returns the OpenCL context that this sampler was created for;
   * null if not yet created within a context. */
  OpenCLContext *
  GetContext() const
  {
    return this->m_Context;
  }

private:
  OpenCLContext * m_Context;
  cl_sampler      m_Id;
};

/** Operator ==
 * Returns true if \a lhs OpenCL sampler is the same as \a rhs, false otherwise.
 * \sa operator!= */
bool ITKOpenCL_EXPORT
     operator==(const OpenCLSampler & lhs, const OpenCLSampler & rhs);

/** Operator !=
 * Returns true if \a lhs OpenCL sampler is not the same as \a rhs, false otherwise.
 * \sa operator== */
bool ITKOpenCL_EXPORT
     operator!=(const OpenCLSampler & lhs, const OpenCLSampler & rhs);

/** Stream out operator for OpenCLSampler */
template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const OpenCLSampler & sampler)
{
  if (sampler.IsNull())
  {
    strm << "OpenCLSampler(null)";
    return strm;
  }

  const char indent = ' ';

  strm << "OpenCLSampler\n" << indent << "Id: " << sampler.GetSamplerId() << std::endl;

  return strm;
}


} // end namespace itk

#endif /* itkOpenCLSampler_h */

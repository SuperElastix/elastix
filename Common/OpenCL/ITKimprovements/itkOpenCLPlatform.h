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
#ifndef itkOpenCLPlatform_h
#define itkOpenCLPlatform_h

#include "itkOpenCL.h"

#include <string>
#include <list>

namespace itk
{
/** \class OpenCLPlatform
 * \brief The OpenCLPlatform represent platform  model for OpenCL.
 *
 * The platform model consists of a host connected to one or more OpenCL devices.
 * An OpenCL device is divided into one or more compute units (CUs) which are
 * further divided into one or more processing elements (PEs).
 * Computations on a device occur within the processing elements.
 *
 * The GetAllPlatforms() function can be used to obtain the list of
 * OpenCL platforms that are accessible to the host. For each
 * platform, OpenCLDevice::GetDevices() can be used to get devices with different
 * capabilities under a single platform.
 *
 * \ingroup OpenCL
 * \sa OpenCLDevice
 */
class ITKOpenCL_EXPORT OpenCLPlatform
{
public:
  /** Standard class typedefs. */
  using Self = OpenCLPlatform;

  /** Constructs a default OpenCL platform identifier. */
  OpenCLPlatform()
    : m_Id(0)
    , m_Version(0)
  {}

  /** Constructs an OpenCL platform identifier that corresponds to the native
   * OpenCL value \a id. */
  OpenCLPlatform(cl_platform_id id)
    : m_Id(id)
    , m_Version(0)
  {}

  /** \enum OpenCLPlatform::VendorType
   * This enum defines the vendor of OpenCL platform that is represented
   * by a OpenCLDevice object.
   * \value Intel The Intel platform.
   * \value NVidia The NVidia platform.
   * \value AMD The AMD platform.
   * \value IBM The IBM platform. */
  enum VendorType
  {
    Default,
    Intel,
    NVidia,
    AMD,
    IBM
  };

  /** Returns true if this OpenCL platform identifier is null. */
  bool
  IsNull() const
  {
    return this->m_Id == 0;
  }

  /** Returns the native OpenCL platform identifier for this object. */
  cl_platform_id
  GetPlatformId() const
  {
    return this->m_Id;
  }

  /** Returns the OpenCL versions supported by this platform.
   * \sa GetVersion(), OpenCLDevice::GetOpenCLVersion() */
  OpenCLVersion
  GetOpenCLVersion() const;

  /** Returns the OpenCL versions supported by this platform as string,
   * usually something like \c{OpenCL 1.2}.
   * \sa GetOpenCLVersion() */
  std::string
  GetVersion() const;

  /** Returns true if profile() is \c FULL_PROFILE, false otherwise.
   * \sa IsEmbeddedProfile() */
  bool
  IsFullProfile() const;

  /** Returns true if profile() is \c EMBEDDED_PROFILE, false otherwise.
   * \sa IsFullProfile() */
  bool
  IsEmbeddedProfile() const;

  /** Returns the profile that is implemented by this OpenCL platform,
   * usually \c FULL_PROFILE or \c EMBEDDED_PROFILE.
   * \sa IsFullProfile(), IsEmbeddedProfile() */
  std::string
  GetProfile() const;

  /** Returns the name of this OpenCL platform. */
  std::string
  GetName() const;

  /** Returns the name of the vendor of this OpenCL platform. */
  std::string
  GetVendor() const;

  /** Returns the OpenCLPlatform::VendorType of the vendor of this OpenCL platform.
   * \sa OpenCLPlatform::VendorType */
  VendorType
  GetVendorType() const;

  /** Returns the vendor extension suffix for this platform if the
   * \c{cl_khr_icd} extension is supported; an empty string otherwise. */
  std::string
  GetExtensionSuffix() const;

  /** Returns a list of the extensions supported by this OpenCL platform.
   * \sa HasExtension() */
  std::list<std::string>
  GetExtensions() const;

  /** Returns true if this platform has an extension called \a name,
   * false otherwise. This function is more efficient than checking for \a name
   * in the return value from GetExtensions(), if the caller is only
   * interested in a single extension. Use extensions() to check
   * for several extensions at once.
   * \sa GetExtensions() */
  bool
  HasExtension(const std::string & name) const;

  /** Returns a list of all OpenCL platforms that are supported by this host. */
  static std::list<OpenCLPlatform>
  GetAllPlatforms();

  /** Returns a list of all OpenCL platforms that are supported by this host. */
  static OpenCLPlatform
  GetPlatform(const OpenCLPlatform::VendorType vendor);

private:
  cl_platform_id m_Id;
  mutable int    m_Version;
};

/** Operator ==
 * Returns true if \a lhs OpenCL platform identifier is the same as \a rhs, false otherwise.
 * \sa operator!= */
bool ITKOpenCL_EXPORT
     operator==(const OpenCLPlatform & lhs, const OpenCLPlatform & rhs);

/** Operator !=
 * Returns true if \a lhs OpenCL platform identifier is not the same as \a rhs, false otherwise.
 * \sa operator== */
bool ITKOpenCL_EXPORT
     operator!=(const OpenCLPlatform & lhs, const OpenCLPlatform & rhs);

/** Stream out operator for OpenCLPlatform */
template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const OpenCLPlatform & platform)
{
  if (platform.IsNull())
  {
    strm << "OpenCLPlatform(null)";
    return strm;
  }

  const char indent = ' ';

  strm << "OpenCLPlatform\n" << indent << "Id: " << platform.GetPlatformId() << std::endl;

  strm << indent << "OpenCL version: ";
  switch (platform.GetOpenCLVersion())
  {
    case VERSION_1_0:
      strm << "1.0";
      break;
    case VERSION_1_1:
      strm << "1.1";
      break;
    case VERSION_1_2:
      strm << "1.2";
      break;
    case VERSION_2_0:
      strm << "2.0";
      break;
    case VERSION_2_1:
      strm << "2.1";
      break;
    default:
      strm << "Unknown";
      break;
  }

  strm << '\n'
       << indent << "Full profile: " << (platform.IsFullProfile() ? "On" : "Off") << '\n'
       << indent << "Embedded profile: " << (platform.IsEmbeddedProfile() ? "On" : "Off") << '\n'
       << indent << "Profile: " << platform.GetProfile() << '\n'
       << indent << "Version: " << platform.GetVersion() << '\n'
       << indent << "Name: " << platform.GetName() << '\n'
       << indent << "Vendor: " << platform.GetVendor() << '\n'
       << indent << "Extension suffix: " << platform.GetExtensionSuffix() << std::endl;

  const std::list<std::string> extensions = platform.GetExtensions();
  const std::size_t            extensionsSize = extensions.size();
  strm << indent << "Extensions(" << extensionsSize << "): ";
  if (extensions.empty())
  {
    strm << "none";
  }
  else
  {
    strm << std::endl;
    for (std::list<std::string>::const_iterator it = extensions.begin(); it != extensions.end(); ++it)
    {
      strm << indent << indent << "- " << *it << std::endl;
    }
  }

  return strm;
}


} // end namespace itk

#endif /* itkOpenCLPlatform_h */

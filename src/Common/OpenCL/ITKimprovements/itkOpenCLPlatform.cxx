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
#include "itkOpenCLPlatform.h"
#include "itkOpenCLExtension.h"
#include "itkOpenCLStringUtils.h"

#include <vector>
#include <iostream>

namespace itk
{
OpenCLVersion
OpenCLPlatform::GetOpenCLVersion() const
{
  if( !this->m_Version )
  {
    this->m_Version = opencl_version_flags( opencl_get_platform_info_string( this->m_Id, CL_PLATFORM_VERSION ) );
  }
  return OpenCLVersion( this->m_Version );
}


//------------------------------------------------------------------------------
std::string
OpenCLPlatform::GetVersion() const
{
  return opencl_get_platform_info_string( this->m_Id, CL_PLATFORM_VERSION );
}


//------------------------------------------------------------------------------
bool
OpenCLPlatform::IsFullProfile() const
{
  return opencl_is_platform( this->m_Id, CL_PLATFORM_PROFILE, "FULL_PROFILE" );
}


//------------------------------------------------------------------------------
bool
OpenCLPlatform::IsEmbeddedProfile() const
{
  return opencl_is_platform( this->m_Id, CL_PLATFORM_PROFILE, "EMBEDDED_PROFILE" );
}


//------------------------------------------------------------------------------
std::string
OpenCLPlatform::GetProfile() const
{
  return opencl_get_platform_info_string( this->m_Id, CL_PLATFORM_PROFILE );
}


//------------------------------------------------------------------------------
std::string
OpenCLPlatform::GetName() const
{
  return opencl_get_platform_info_string( this->m_Id, CL_PLATFORM_NAME );
}


//------------------------------------------------------------------------------
std::string
OpenCLPlatform::GetVendor() const
{
  return opencl_get_platform_info_string( this->m_Id, CL_PLATFORM_VENDOR );
}


//------------------------------------------------------------------------------
OpenCLPlatform::VendorType
OpenCLPlatform::GetVendorType() const
{
  const std::string vendorName = opencl_simplified( this->GetVendor() );
  if( vendorName.compare( 0, 20, "Intel(R) Corporation" ) == 0 )
  {
    return OpenCLPlatform::Intel;
  }
  else if( vendorName.compare( 0, 18, "NVIDIA Corporation" ) == 0 )
  {
    return OpenCLPlatform::NVidia;
  }
  else if( vendorName.compare( 0, 28, "Advanced Micro Devices, Inc." ) == 0 )
  {
    return OpenCLPlatform::AMD;
  }
  else if( vendorName.compare( 0, 3, "IBM" ) == 0 )
  {
    return OpenCLPlatform::IBM;
  }
  else
  {
    return OpenCLPlatform::Default;
  }
}


//------------------------------------------------------------------------------
std::string
OpenCLPlatform::GetExtensionSuffix() const
{
  return opencl_get_platform_info_string( this->m_Id, CL_PLATFORM_ICD_SUFFIX_KHR );
}


//------------------------------------------------------------------------------
std::list< std::string > OpenCLPlatform::GetExtensions() const
{
  if( this->IsNull() )
  {
    return std::list< std::string >();
  }

  const std::string extensions = opencl_simplified(
    opencl_get_platform_info_string( this->m_Id, CL_PLATFORM_EXTENSIONS ) );
  if( !extensions.empty() )
  {
    return opencl_split_string( extensions, ' ' );
  }
  else
  {
    return std::list< std::string >();
  }
}


//------------------------------------------------------------------------------
bool
OpenCLPlatform::HasExtension( const std::string & name ) const
{
  std::size_t size;

  if( !this->m_Id || clGetPlatformInfo( this->m_Id, CL_PLATFORM_EXTENSIONS,
    0, 0, &size ) != CL_SUCCESS )
  {
    return false;
  }
  std::string buffer( size, '\0' );
  clGetPlatformInfo( this->m_Id, CL_PLATFORM_EXTENSIONS, size, &buffer[ 0 ], &size );
  return opencl_has_extension( buffer, name );
}


//------------------------------------------------------------------------------
std::list< OpenCLPlatform > OpenCLPlatform::GetAllPlatforms()
{
  cl_uint size;

  if( clGetPlatformIDs( 0, 0, &size ) != CL_SUCCESS )
  {
    return std::list< OpenCLPlatform >();
  }
  std::vector< cl_platform_id > buffer( size );
  clGetPlatformIDs( size, &buffer[ 0 ], &size );
  std::list< OpenCLPlatform > platforms;
  for( std::size_t index = 0; index < buffer.size(); ++index )
  {
    platforms.push_back( OpenCLPlatform( buffer[ index ] ) );
  }
  return platforms;
}


//------------------------------------------------------------------------------
OpenCLPlatform
OpenCLPlatform::GetPlatform( const OpenCLPlatform::VendorType vendor )
{
  const std::list< OpenCLPlatform > platforms = OpenCLPlatform::GetAllPlatforms();

  if( platforms.empty() )
  {
    return OpenCLPlatform();
  }

  cl_platform_id platformID = 0;

  for( std::list< itk::OpenCLPlatform >::const_iterator platform = platforms.begin();
    platform != platforms.end(); ++platform )
  {
    const std::string vendorName = opencl_simplified( ( *platform ).GetVendor() );

    if( ( vendorName.compare( 0, 20, "Intel(R) Corporation" ) == 0 ) && ( vendor == OpenCLPlatform::Intel ) )
    {
      platformID = ( *platform ).GetPlatformId();
      break;
    }
    else if( ( vendorName.compare( 0, 18, "NVIDIA Corporation" ) == 0 ) && ( vendor == OpenCLPlatform::NVidia ) )
    {
      platformID = ( *platform ).GetPlatformId();
      break;
    }
    else if( ( vendorName.compare( 0, 28,
      "Advanced Micro Devices, Inc." ) == 0 ) && ( vendor == OpenCLPlatform::AMD ) )
    {
      platformID = ( *platform ).GetPlatformId();
      break;
    }
    else if( ( vendorName.compare( 0, 3, "IBM" ) == 0 ) && ( vendor == OpenCLPlatform::IBM ) )
    {
      platformID = ( *platform ).GetPlatformId();
      break;
    }
  }

  return OpenCLPlatform( platformID );
}


//------------------------------------------------------------------------------
//! Operator ==
bool
operator==( const OpenCLPlatform & lhs, const OpenCLPlatform & rhs )
{
  if( &rhs == &lhs )
  {
    return true;
  }
  return lhs.GetPlatformId() == rhs.GetPlatformId();
}


//------------------------------------------------------------------------------
//! Operator !=
bool
operator!=( const OpenCLPlatform & lhs, const OpenCLPlatform & rhs )
{
  return !( lhs == rhs );
}


} // namespace itk

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
#include "itkOpenCLStringUtils.h"

#include <iosfwd>
#include <sstream>
#include <cstring>

namespace itk
{
inline bool
opencl_isspace( const char c )
{
  switch( c )
  {
    case '\t':
    case '\n':
    case '\r':
    case '\f':
    case ' ':
      return true;

    default:
      return false;
  }
}


//------------------------------------------------------------------------------
std::string
opencl_simplified( const std::string & str )
{
  if( str.empty() )
  {
    return str;
  }

  const std::size_t size = str.size();
  std::string       result( size, '\0' );
  const char *      from    = &str[ 0 ];
  const char *      fromend = from + size;
  int               outc    = 0;
  char *            to      = &result[ 0 ];
  for(;; )
  {
    while( from != fromend && opencl_isspace( char(*from) ) )
    {
      from++;
    }
    while( from != fromend && !opencl_isspace( char(*from) ) )
    {
      to[ outc++ ] = *from++;
    }
    if( from != fromend )
    {
      to[ outc++ ] = ' ';
    }
    else
    {
      break;
    }
  }
  if( outc > 0 && to[ outc - 1 ] == ' ' )
  {
    outc--;
  }
  result.resize( outc );

  return result;
}


//------------------------------------------------------------------------------
bool
opencl_has_extension( const std::string & list, const std::string & name )
{
  if( list.empty() || name.empty() )
  {
    return false;
  }
  const std::size_t found = list.find( name );
  if( found != std::string::npos )
  {
    return true;
  }
  else
  {
    return false;
  }
}


//------------------------------------------------------------------------------
std::string
opencl_get_platform_info_string( const cl_platform_id id, const cl_platform_info name )
{
  std::size_t size;

  if( !id || clGetPlatformInfo( id, name, 0, 0, &size ) != CL_SUCCESS )
  {
    return std::string();
  }
  std::string buffer( size, '\0' );
  clGetPlatformInfo( id, name, size, &buffer[ 0 ], &size );
  return buffer;
}


//------------------------------------------------------------------------------
std::string
opencl_get_device_info_string( const cl_device_id id, const cl_device_info name )
{
  std::size_t size;

  if( !id || clGetDeviceInfo( id, name, 0, 0, &size ) != CL_SUCCESS )
  {
    return std::string();
  }
  std::string buffer( size, '\0' );
  clGetDeviceInfo( id, name, size, &buffer[ 0 ], &size );
  return buffer;
}


//------------------------------------------------------------------------------
bool
opencl_is_platform( cl_platform_id id, cl_platform_info name, const char * str )
{
  std::size_t len = strlen( str );
  std::size_t size;

  if( !id || clGetPlatformInfo( id, name, 0, 0, &size ) != CL_SUCCESS )
  {
    return false;
  }
  if( size <= len )
  {
    return false;
  }
  std::string buffer( size, '\0' );
  clGetPlatformInfo( id, name, size, &buffer[ 0 ], &size );
  if( strncmp( &buffer[ 0 ], str, len ) != 0 )
  {
    return false;
  }
  return buffer[ len ] == '\0';
}


//------------------------------------------------------------------------------
int
opencl_version_flags( const std::string & /*version*/ )
{
  // not implemented.
  return 1;
}


//------------------------------------------------------------------------------
std::list< std::string > opencl_split_string( const std::string & str, const char separator )
{
  std::list< std::string > strings;

  if( str.empty() )
  {
    return strings;
  }

  std::istringstream f( str );
  std::string        s;

  while( std::getline( f, s, separator ) )
  {
    strings.push_back( s );
  }
  return strings;
}


//------------------------------------------------------------------------------
unsigned int
opencl_get_device_info_uint( const cl_device_id id, const cl_device_info name )
{
  cl_uint value;

  if( !id || clGetDeviceInfo( id, name, sizeof( value ), &value, 0 )
    != CL_SUCCESS )
  {
    return 0;
  }
  else
  {
    const unsigned int result = value;
    return result;
  }
}


//------------------------------------------------------------------------------
int
opencl_get_device_info_int( const cl_device_id id, const cl_device_info name )
{
  cl_int value;

  if( !id || clGetDeviceInfo( id, name, sizeof( value ), &value, 0 )
    != CL_SUCCESS )
  {
    return 0;
  }
  else
  {
    const int result = value;
    return result;
  }
}


//------------------------------------------------------------------------------
unsigned long
opencl_get_device_info_ulong( const cl_device_id id, const cl_device_info name )
{
  cl_ulong value;

  if( !id || clGetDeviceInfo( id, name, sizeof( value ), &value, 0 )
    != CL_SUCCESS )
  {
    return 0;
  }
  else
  {
    const unsigned long result = value;
    return result;
  }
}


//------------------------------------------------------------------------------
std::size_t
opencl_get_device_info_size( const cl_device_id id, const cl_device_info name )
{
  std::size_t value;

  if( !id || clGetDeviceInfo( id, name, sizeof( value ), &value, 0 )
    != CL_SUCCESS )
  {
    return 0;
  }
  else
  {
    return value;
  }
}


//------------------------------------------------------------------------------
bool
opencl_get_device_info_bool( const cl_device_id id, const cl_device_info name )
{
  cl_bool value;

  if( !id || clGetDeviceInfo( id, name, sizeof( value ), &value, 0 )
    != CL_SUCCESS )
  {
    return false;
  }
  else
  {
    return value != 0;
  }
}


//------------------------------------------------------------------------------
bool
opencl_get_device_info_is_string( const cl_device_id id, const cl_device_info name, const char * str )
{
  std::size_t len = strlen( str );
  std::size_t size;

  if( !id || clGetDeviceInfo( id, name, 0, 0, &size ) != CL_SUCCESS )
  {
    return false;
  }
  if( size <= len )
  {
    return false;
  }
  std::string buffer( size, '\0' );
  clGetDeviceInfo( id, name, size, &buffer[ 0 ], &size );
  if( strncmp( &buffer[ 0 ], str, len ) != 0 )
  {
    return false;
  }
  return buffer[ len ] == '\0';
}


} // end of namespace itk

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
#include "itkOpenCLDevice.h"
#include "itkOpenCLExtension.h"
#include "itkOpenCLStringUtils.h"

#include <vector>
#include <set>

namespace itk
{
OpenCLVersion
OpenCLDevice::GetOpenCLVersion() const
{
  if( !this->m_Version )
  {
    this->m_Version = opencl_version_flags( opencl_get_device_info_string( this->m_Id, CL_DEVICE_VERSION ) );
  }
  return OpenCLVersion( this->m_Version );
}


//------------------------------------------------------------------------------
std::string
OpenCLDevice::GetVersion() const
{
  return opencl_get_device_info_string( this->m_Id, CL_DEVICE_VERSION );
}


//------------------------------------------------------------------------------
OpenCLDevice::DeviceType
OpenCLDevice::GetDeviceType() const
{
  cl_device_type type;

  if( this->IsNull() || clGetDeviceInfo( this->m_Id, CL_DEVICE_TYPE, sizeof( type ), &type, 0 )
    != CL_SUCCESS )
  {
    return OpenCLDevice::DeviceType( 0 );
  }
  else
  {
    return OpenCLDevice::DeviceType( type );
  }
}


//------------------------------------------------------------------------------
OpenCLPlatform
OpenCLDevice::GetPlatform() const
{
  cl_platform_id platform;

  if( this->IsNull() || clGetDeviceInfo( this->m_Id, CL_DEVICE_PLATFORM, sizeof( platform ),
    &platform, 0 )
    != CL_SUCCESS )
  {
    return OpenCLPlatform();
  }
  else
  {
    return OpenCLPlatform( platform );
  }
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetVendorId() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_VENDOR_ID );
}


//------------------------------------------------------------------------------
bool
OpenCLDevice::IsAvailable() const
{
  return opencl_get_device_info_bool( this->m_Id, CL_DEVICE_AVAILABLE );
}


//------------------------------------------------------------------------------
bool
OpenCLDevice::HasCompiler() const
{
  return opencl_get_device_info_bool( this->m_Id, CL_DEVICE_COMPILER_AVAILABLE );
}


//------------------------------------------------------------------------------
bool
OpenCLDevice::HasNativeKernels() const
{
  cl_device_exec_capabilities caps;

  if( this->IsNull() || clGetDeviceInfo( this->m_Id, CL_DEVICE_EXECUTION_CAPABILITIES,
    sizeof( caps ), &caps, 0 )
    != CL_SUCCESS )
  {
    return false;
  }
  else
  {
    return ( caps & CL_EXEC_NATIVE_KERNEL ) != 0;
  }
}


//------------------------------------------------------------------------------
bool
OpenCLDevice::HasOutOfOrderExecution() const
{
  cl_command_queue_properties props;

  if( this->IsNull() || clGetDeviceInfo( this->m_Id, CL_DEVICE_QUEUE_PROPERTIES,
    sizeof( props ), &props, 0 )
    != CL_SUCCESS )
  {
    return false;
  }
  else
  {
    return ( props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ) != 0;
  }
}


//------------------------------------------------------------------------------
bool
OpenCLDevice::HasDouble() const
{
  return this->HasExtension( "cl_khr_fp64" );
}


//------------------------------------------------------------------------------
bool
OpenCLDevice::HasHalfFloat() const
{
  return this->HasExtension( "cl_khr_fp16" );
}


//------------------------------------------------------------------------------
bool
OpenCLDevice::HasErrorCorrectingMemory() const
{
  return opencl_get_device_info_bool( this->m_Id, CL_DEVICE_ERROR_CORRECTION_SUPPORT );
}


//------------------------------------------------------------------------------
bool
OpenCLDevice::HasUnifiedMemory() const
{
  return opencl_get_device_info_bool( this->m_Id, CL_DEVICE_HOST_UNIFIED_MEMORY );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetComputeUnits() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_MAX_COMPUTE_UNITS );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetClockFrequency() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_MAX_CLOCK_FREQUENCY );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetAddressBits() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_ADDRESS_BITS );
}


//------------------------------------------------------------------------------
OpenCLDevice::Endian
OpenCLDevice::GetByteOrder() const
{
  if( opencl_get_device_info_bool( this->m_Id, CL_DEVICE_ENDIAN_LITTLE ) )
  {
    return OpenCLDevice::LittleEndian;
  }
  else
  {
    return OpenCLDevice::BigEndian;
  }
}


//------------------------------------------------------------------------------
OpenCLSize
OpenCLDevice::GetMaximumWorkItemSize() const
{
  std::size_t dims = 0;

  if( this->IsNull() || clGetDeviceInfo( this->m_Id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
    sizeof( dims ), &dims, 0 ) != CL_SUCCESS || !dims )
  {
    return OpenCLSize( 1, 1, 1 );
  }

  std::vector< std::size_t > buffer( dims );
  clGetDeviceInfo( this->m_Id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof( std::size_t ) * dims, &buffer[ 0 ], 0 );
  if( dims == 1 )
  {
    return OpenCLSize( buffer[ 0 ] );
  }
  else if( dims == 2 )
  {
    return OpenCLSize( buffer[ 0 ], buffer[ 1 ] );
  }
  else
  {
    return OpenCLSize( buffer[ 0 ], buffer[ 1 ], buffer[ 2 ] );
  }
}


//------------------------------------------------------------------------------
size_t
OpenCLDevice::GetMaximumWorkItemsPerGroup() const
{
  return opencl_get_device_info_size( this->m_Id, CL_DEVICE_MAX_WORK_GROUP_SIZE );
}


//------------------------------------------------------------------------------
bool
OpenCLDevice::HasImage2D() const
{
  return opencl_get_device_info_bool( this->m_Id, CL_DEVICE_IMAGE_SUPPORT );
}


//------------------------------------------------------------------------------
bool
OpenCLDevice::HasImage3D() const
{
  if( !opencl_get_device_info_bool( this->m_Id, CL_DEVICE_IMAGE_SUPPORT ) )
  {
    return false;
  }
  return opencl_get_device_info_size( this->m_Id, CL_DEVICE_IMAGE3D_MAX_WIDTH ) != 0
         || opencl_get_device_info_size( this->m_Id, CL_DEVICE_IMAGE3D_MAX_HEIGHT ) != 0
         || opencl_get_device_info_size( this->m_Id, CL_DEVICE_IMAGE3D_MAX_DEPTH ) != 0;
}


//------------------------------------------------------------------------------
bool
OpenCLDevice::HasWritableImage3D() const
{
  return this->HasExtension( "cl_khr_3d_image_writes" );
}


//------------------------------------------------------------------------------
OpenCLSize
OpenCLDevice::GetMaximumImage2DSize() const
{
  if( !opencl_get_device_info_bool( this->m_Id, CL_DEVICE_IMAGE_SUPPORT ) )
  {
    return OpenCLSize();
  }
  return OpenCLSize( opencl_get_device_info_size( this->m_Id, CL_DEVICE_IMAGE2D_MAX_WIDTH ),
    opencl_get_device_info_size( this->m_Id, CL_DEVICE_IMAGE2D_MAX_HEIGHT ) );
}


//------------------------------------------------------------------------------
OpenCLSize
OpenCLDevice::GetMaximumImage3DSize() const
{
  if( !opencl_get_device_info_bool( this->m_Id, CL_DEVICE_IMAGE_SUPPORT ) )
  {
    return OpenCLSize( 0, 0, 0 );
  }
  return OpenCLSize( opencl_get_device_info_size( this->m_Id, CL_DEVICE_IMAGE3D_MAX_WIDTH ),
    opencl_get_device_info_size( this->m_Id, CL_DEVICE_IMAGE3D_MAX_HEIGHT ),
    opencl_get_device_info_size( this->m_Id, CL_DEVICE_IMAGE3D_MAX_DEPTH ) );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetMaximumSamplers() const
{
  if( !opencl_get_device_info_bool( this->m_Id, CL_DEVICE_IMAGE_SUPPORT ) )
  {
    return 0;
  }
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_MAX_SAMPLERS );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetMaximumReadImages() const
{
  if( !opencl_get_device_info_bool( this->m_Id, CL_DEVICE_IMAGE_SUPPORT ) )
  {
    return 0;
  }
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_MAX_READ_IMAGE_ARGS );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetMaximumWriteImages() const
{
  if( !opencl_get_device_info_bool( this->m_Id, CL_DEVICE_IMAGE_SUPPORT ) )
  {
    return 0;
  }
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_MAX_WRITE_IMAGE_ARGS );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetPreferredCharVectorSize() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetPreferredShortVectorSize() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetPreferredIntVectorSize() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetPreferredLongVectorSize() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetPreferredFloatVectorSize() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetPreferredDoubleVectorSize() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetPreferredHalfFloatVectorSize() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetNativeCharVectorSize() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetNativeShortVectorSize() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetNativeIntVectorSize() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetNativeLongVectorSize() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetNativeFloatVectorSize() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetNativeDoubleVectorSize() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetNativeHalfFloatVectorSize() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF );
}


//------------------------------------------------------------------------------
OpenCLDevice::FloatCapability
OpenCLDevice::GetFloatCapabilities() const
{
  cl_device_fp_config config;

  if( this->IsNull() || clGetDeviceInfo( this->m_Id, CL_DEVICE_SINGLE_FP_CONFIG,
    sizeof( config ), &config, 0 )
    != CL_SUCCESS )
  {
    return NotSupported;
  }
  else
  {
    return OpenCLDevice::FloatCapability( config );
  }
}


//------------------------------------------------------------------------------
OpenCLDevice::FloatCapability
OpenCLDevice::GetDoubleCapabilities() const
{
  cl_device_fp_config config;

  if( this->IsNull() || clGetDeviceInfo( this->m_Id, CL_DEVICE_DOUBLE_FP_CONFIG,
    sizeof( config ), &config, 0 )
    != CL_SUCCESS )
  {
    return NotSupported;
  }
  else
  {
    return OpenCLDevice::FloatCapability( config );
  }
}


//------------------------------------------------------------------------------
OpenCLDevice::FloatCapability
OpenCLDevice::GetHalfFloatCapabilities() const
{
  cl_device_fp_config config;

  if( this->IsNull() || clGetDeviceInfo( this->m_Id, CL_DEVICE_HALF_FP_CONFIG,
    sizeof( config ), &config, 0 )
    != CL_SUCCESS )
  {
    return NotSupported;
  }
  else
  {
    return OpenCLDevice::FloatCapability( config );
  }
}


//------------------------------------------------------------------------------
std::size_t
OpenCLDevice::GetProfilingTimerResolution() const
{
  return opencl_get_device_info_size( this->m_Id, CL_DEVICE_PROFILING_TIMER_RESOLUTION );
}


//------------------------------------------------------------------------------
unsigned long
OpenCLDevice::GetMaximumAllocationSize() const
{
  return opencl_get_device_info_ulong( this->m_Id, CL_DEVICE_MAX_MEM_ALLOC_SIZE );
}


//------------------------------------------------------------------------------
unsigned long
OpenCLDevice::GetGlobalMemorySize() const
{
  return opencl_get_device_info_ulong( this->m_Id, CL_DEVICE_GLOBAL_MEM_SIZE );
}


//------------------------------------------------------------------------------
OpenCLDevice::CacheType
OpenCLDevice::GetGlobalMemoryCacheType() const
{
  cl_device_mem_cache_type type;

  if( this->IsNull() || clGetDeviceInfo( this->m_Id, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
    sizeof( type ), &type, 0 )
    != CL_SUCCESS )
  {
    return NoCache;
  }
  else
  {
    return OpenCLDevice::CacheType( type );
  }
}


//------------------------------------------------------------------------------
unsigned long
OpenCLDevice::GetGlobalMemoryCacheSize() const
{
  return opencl_get_device_info_ulong( this->m_Id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetGlobalMemoryCacheLineSize() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE );
}


//------------------------------------------------------------------------------
unsigned long
OpenCLDevice::GetLocalMemorySize() const
{
  return opencl_get_device_info_ulong( this->m_Id, CL_DEVICE_LOCAL_MEM_SIZE );
}


//------------------------------------------------------------------------------
bool
OpenCLDevice::IsLocalMemorySeparate() const
{
  cl_device_local_mem_type type;

  if( this->IsNull() || clGetDeviceInfo( this->m_Id, CL_DEVICE_LOCAL_MEM_TYPE,
    sizeof( type ), &type, 0 )
    != CL_SUCCESS )
  {
    return false;
  }
  else
  {
    return type == CL_LOCAL;
  }
}


//------------------------------------------------------------------------------
unsigned long
OpenCLDevice::GetMaximumConstantBufferSize() const
{
  return opencl_get_device_info_ulong( this->m_Id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetMaximumConstantArguments() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_MAX_CONSTANT_ARGS );
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetDefaultAlignment() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_MEM_BASE_ADDR_ALIGN ) / 8;
}


//------------------------------------------------------------------------------
unsigned int
OpenCLDevice::GetMinimumAlignment() const
{
  return opencl_get_device_info_uint( this->m_Id, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE );
}


//------------------------------------------------------------------------------
std::size_t
OpenCLDevice::GetMaximumParameterBytes() const
{
  return opencl_get_device_info_size( this->m_Id, CL_DEVICE_MAX_PARAMETER_SIZE );
}


//------------------------------------------------------------------------------
bool
OpenCLDevice::IsFullProfile() const
{
  return opencl_get_device_info_is_string( this->m_Id, CL_DEVICE_PROFILE, "FULL_PROFILE" );
}


//------------------------------------------------------------------------------
bool
OpenCLDevice::IsEmbeddedProfile() const
{
  return opencl_get_device_info_is_string( this->m_Id, CL_DEVICE_PROFILE, "EMBEDDED_PROFILE" );
}


//------------------------------------------------------------------------------
std::string
OpenCLDevice::GetProfile() const
{
  return opencl_get_device_info_string( this->m_Id, CL_DEVICE_PROFILE );
}


//------------------------------------------------------------------------------
std::string
OpenCLDevice::GetDriverVersion() const
{
  return opencl_get_device_info_string( this->m_Id, CL_DRIVER_VERSION );
}


//------------------------------------------------------------------------------
std::string
OpenCLDevice::GetName() const
{
  return opencl_get_device_info_string( this->m_Id, CL_DEVICE_NAME );
}


//------------------------------------------------------------------------------
std::string
OpenCLDevice::GetVendor() const
{
  return opencl_get_device_info_string( this->m_Id, CL_DEVICE_VENDOR );
}


//------------------------------------------------------------------------------
std::list< std::string > OpenCLDevice::GetExtensions() const
{
  if( this->IsNull() )
  {
    return std::list< std::string >();
  }
  const std::string extensions = opencl_simplified( opencl_get_device_info_string( this->m_Id, CL_DEVICE_EXTENSIONS ) );
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
std::string
OpenCLDevice::GetLanguageVersion() const
{
  std::string vers = opencl_get_device_info_string( this->m_Id, CL_DEVICE_OPENCL_C_VERSION );

  if( vers.empty() && !( this->GetOpenCLVersion() & VERSION_1_1 ) )
  {
    vers = "OpenCL 1.0";
  }
  return vers;
}


//------------------------------------------------------------------------------
bool
OpenCLDevice::HasExtension( const std::string & name ) const
{
  std::size_t size;

  if( this->IsNull() || clGetDeviceInfo( this->m_Id, CL_DEVICE_EXTENSIONS, 0, 0, &size ) != CL_SUCCESS )
  {
    return false;
  }
  std::string buffer( size, '\0' );
  clGetDeviceInfo( this->m_Id, CL_DEVICE_EXTENSIONS, size, &buffer[ 0 ], &size );
  return opencl_has_extension( buffer, name );
}


//------------------------------------------------------------------------------
std::list< OpenCLDevice > OpenCLDevice::GetAllDevices()
{
  const std::list< itk::OpenCLPlatform > platforms = OpenCLPlatform::GetAllPlatforms();
  std::list< OpenCLDevice >              devices;

  for( std::list< itk::OpenCLPlatform >::const_iterator platform = platforms.begin();
    platform != platforms.end(); ++platform )
  {
    cl_uint size;
    if( clGetDeviceIDs( ( *platform ).GetPlatformId(), CL_DEVICE_TYPE_ALL, 0, 0, &size ) != CL_SUCCESS )
    {
      continue;
    }
    std::vector< cl_device_id > buffer( size );
    clGetDeviceIDs( ( *platform ).GetPlatformId(), CL_DEVICE_TYPE_ALL, size, &buffer[ 0 ], &size );
    for( std::vector< cl_device_id >::iterator device = buffer.begin(); device != buffer.end(); ++device )
    {
      devices.push_back( OpenCLDevice( *device ) );
    }
  }
  return devices;
}


//------------------------------------------------------------------------------
std::list< OpenCLDevice > OpenCLDevice::GetDevices( const OpenCLDevice::DeviceType types,
  const OpenCLPlatform & platform )
{
  std::list< OpenCLDevice >   devices;
  std::list< OpenCLPlatform > platforms;

  if( platform.IsNull() )
  {
    platforms = OpenCLPlatform::GetAllPlatforms();
  }
  else
  {
    platforms.push_back( platform );
  }
  for( std::list< itk::OpenCLPlatform >::iterator platform = platforms.begin();
    platform != platforms.end(); ++platform )
  {
    cl_uint size;
    if( clGetDeviceIDs( ( *platform ).GetPlatformId(), cl_device_type( types ), 0, 0, &size ) != CL_SUCCESS )
    {
      continue;
    }
    if( !size )
    {
      continue;
    }
    std::vector< cl_device_id > buffer( size );
    clGetDeviceIDs( ( *platform ).GetPlatformId(), cl_device_type( types ),
      size, &buffer[ 0 ], &size );
    for( std::vector< cl_device_id >::iterator device = buffer.begin(); device != buffer.end(); ++device )
    {
      devices.push_back( OpenCLDevice( *device ) );
    }
    break;
  }
  return devices;
}


//------------------------------------------------------------------------------
std::list< OpenCLDevice > OpenCLDevice::GetDevices( const OpenCLDevice::DeviceType type,
  const OpenCLPlatform::VendorType vendor )
{
  const OpenCLPlatform platform = OpenCLPlatform::GetPlatform( vendor );

  if( platform.IsNull() )
  {
    return std::list< OpenCLDevice >();
  }
  else
  {
    const std::list< itk::OpenCLDevice > allDevices = itk::OpenCLDevice::GetDevices( type, platform );
    std::list< OpenCLDevice >            devices;
    for( std::list< itk::OpenCLDevice >::const_iterator dev = allDevices.begin(); dev != allDevices.end(); ++dev )
    {
      if( ( ( *dev ).GetDeviceType() & type ) != 0 )
      {
        devices.push_back( *dev );
      }
    }
    return devices;
  }
}


//------------------------------------------------------------------------------
OpenCLDevice
OpenCLDevice::GetMaximumFlopsDevice( const std::list< OpenCLDevice > & devices,
  const OpenCLDevice::DeviceType type )
{
  if( devices.empty() )
  {
    return OpenCLDevice();
  }

  // Find the device that has maximum Flops
  int          maxFlops = 0;
  cl_device_id id       = 0;
  for( std::list< OpenCLDevice >::const_iterator device = devices.begin(); device != devices.end(); ++device )
  {
    int deviceFlops = ( *device ).GetComputeUnits() * ( *device ).GetClockFrequency();
    if( deviceFlops > maxFlops && ( ( *device ).GetDeviceType() == type ) )
    {
      maxFlops = deviceFlops;
      id       = ( *device ).GetDeviceId();
    }
  }

  return OpenCLDevice( id );
}


//------------------------------------------------------------------------------
OpenCLDevice
OpenCLDevice::GetMaximumFlopsDevice( const OpenCLDevice::DeviceType type )
{
  const std::list< OpenCLDevice > devices = itk::OpenCLDevice::GetAllDevices();

  return GetMaximumFlopsDevice( devices, type );
}


//------------------------------------------------------------------------------
OpenCLDevice
OpenCLDevice::GetMaximumFlopsDeviceByVendor( const OpenCLDevice::DeviceType type,
  const OpenCLPlatform::VendorType vendor )
{
  const std::list< OpenCLDevice > devices = itk::OpenCLDevice::GetDevices( type, vendor );

  return GetMaximumFlopsDevice( devices, type );
}


//------------------------------------------------------------------------------
OpenCLDevice
OpenCLDevice::GetMaximumFlopsDeviceByPlatform( const OpenCLDevice::DeviceType types,
  const OpenCLPlatform & platform )
{
  const std::list< OpenCLDevice > devices = itk::OpenCLDevice::GetDevices( types, platform );

  return GetMaximumFlopsDevice( devices, types );
}


//------------------------------------------------------------------------------
std::list< OpenCLDevice > OpenCLDevice::GetMaximumFlopsDevices( const OpenCLDevice::DeviceType type,
  const OpenCLPlatform & platform )
{
  const std::list< OpenCLDevice > allDevices = itk::OpenCLDevice::GetDevices( type, platform );

  if( allDevices.empty() )
  {
    return std::list< OpenCLDevice >();
  }

  // Find the device that has maximum Flops
  typedef std::pair< std::size_t, cl_device_id > DeviceType;
  typedef std::set< DeviceType >                 MaximumFlopsDevicesType;
  MaximumFlopsDevicesType maximumFlopsDevices;
  for( std::list< OpenCLDevice >::const_iterator device = allDevices.begin(); device != allDevices.end(); ++device )
  {
    int        deviceFlops = ( *device ).GetComputeUnits() * ( *device ).GetClockFrequency();
    DeviceType deviceWithFlops( deviceFlops, ( *device ).GetDeviceId() );
    maximumFlopsDevices.insert( deviceWithFlops );
  }

  // Combine result
  std::list< OpenCLDevice > devices;
  for( MaximumFlopsDevicesType::const_reverse_iterator rit = maximumFlopsDevices.rbegin();
    rit != maximumFlopsDevices.rend();
    ++rit )
  {
    OpenCLDevice device( ( *rit ).second );
    devices.push_back( device );
  }

  return devices;
}


//------------------------------------------------------------------------------
//! Operator ==
bool
operator==( const OpenCLDevice & lhs, const OpenCLDevice & rhs )
{
  if( &rhs == &lhs )
  {
    return true;
  }
  return lhs.GetDeviceId() == rhs.GetDeviceId();
}


//------------------------------------------------------------------------------
//! Operator !=
bool
operator!=( const OpenCLDevice & lhs, const OpenCLDevice & rhs )
{
  return !( lhs == rhs );
}


} // namespace itk

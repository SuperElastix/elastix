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
#include "itkOpenCLContext.h"
#include "itkOpenCLKernels.h"
#include "itkOpenCLProfilingTimeProbe.h"

#include <iostream>
#include <fstream>

#include "itksys/MD5.h"
#include "itkOpenCLMacro.h"

namespace itk
{
// static variable initialization
OpenCLContext::Pointer OpenCLContext::m_Instance = 0;

//------------------------------------------------------------------------------
// Return the single instance of the OpenCLContext
OpenCLContext::Pointer
OpenCLContext::GetInstance()
{
  if( !OpenCLContext::m_Instance )
  {
    // Try the factory first
    OpenCLContext::m_Instance = ObjectFactory< Self >::Create();
    // if the factory did not provide one, then create it here
    if( !OpenCLContext::m_Instance )
    {
      // For the windows OS, use a special output window
      OpenCLContext::m_Instance = new OpenCLContext;
      // Remove extra reference from construction.
      OpenCLContext::m_Instance->UnRegister();
    }
  }
  // Return the instance
  return OpenCLContext::m_Instance;
}


//------------------------------------------------------------------------------
void
OpenCLContext::SetInstance( OpenCLContext * instance )
{
  if( OpenCLContext::m_Instance == instance )
  {
    return;
  }
  OpenCLContext::m_Instance = instance;
}


//------------------------------------------------------------------------------
// This just calls GetInstance
OpenCLContext::Pointer
OpenCLContext::New()
{
  return GetInstance();
}


//------------------------------------------------------------------------------
class OpenCLContextPimpl
{
public:

  OpenCLContextPimpl() :
    id( 0 ),
    is_created( false ),
    last_error( CL_SUCCESS )
  {}

  ~OpenCLContextPimpl()
  {
    // Release the command queues for the context.
    command_queue         = OpenCLCommandQueue();
    default_command_queue = OpenCLCommandQueue();

    // Release the context.
    if( is_created )
    {
      clReleaseContext( id );
    }
  }


public:

  cl_context         id;
  bool               is_created;
  OpenCLCommandQueue command_queue;
  OpenCLCommandQueue default_command_queue;
  OpenCLDevice       default_device;
  cl_int             last_error;
};

//------------------------------------------------------------------------------
std::string
GetOpenCLDebugFileName( const std::string & source )
{
  // Create unique filename based on the source code
  const std::size_t sourceSize = source.size();
  itksysMD5 *       md5        = itksysMD5_New();

  itksysMD5_Initialize( md5 );
  itksysMD5_Append( md5, (unsigned char *)source.c_str(), sourceSize );
  const std::size_t DigestSize = 32u;
  char              Digest[ DigestSize ];
  itksysMD5_FinalizeHex( md5, Digest );
  const std::string hex( Digest, DigestSize );

  // construct the name
  std::string fileName( itk::OpenCLKernelsDebugDirectory );
  fileName.append( "/ocl-" );
  fileName.append( hex );
  fileName.append( ".cl" );

  // free resources
  itksysMD5_Delete( md5 );

  return fileName;
}


//------------------------------------------------------------------------------
OpenCLContext::OpenCLContext() :
  d_ptr( new OpenCLContextPimpl() )
{}

//------------------------------------------------------------------------------
// Destructor has to be in cxx, otherwise compiler will print warning messages.
OpenCLContext::~OpenCLContext()
{}

//------------------------------------------------------------------------------
bool
OpenCLContext::IsCreated() const
{
  ITK_OPENCL_D( const OpenCLContext );
  return d->is_created;
}


//------------------------------------------------------------------------------
// This callback function will be used by the OpenCL implementation to report
// information on errors during context creation as well as errors that occur
// at runtime in this context. This callback function may be called
// asynchronously by the OpenCL implementation. It is the application's
// responsibility to ensure that the callback function is thread-safe.
void CL_CALLBACK
opencl_context_notify( const char * errinfo,
  const void * /*private_info*/,
  std::size_t /*cb*/,
  void * /*user_data*/ )
{
  itkOpenCLErrorMacroGeneric(
      << "OpenCL error during context creation or runtime:"
      << std::endl << errinfo );
}


//------------------------------------------------------------------------------
bool
OpenCLContext::Create( const OpenCLDevice::DeviceType type )
{
  ITK_OPENCL_D( OpenCLContext );
  if( d->is_created )
  {
    return true;
  }

  // The "cl_khr_icd" extension says that a null platform cannot
  // be supplied to OpenCL any more, so find the first platform
  // that has devices that match "type".
  const std::list< OpenCLDevice > devices = OpenCLDevice::GetDevices( type );
  this->CreateContext( devices, d );

  // Check if OpenCL context has been created
  d->is_created = ( d->id != 0 );
  if( !d->is_created )
  {
    itkOpenCLWarningMacro( << "OpenCLContext::Create(type:" << int(type) << "):"
                           << this->GetErrorName( d->last_error ) );
  }
  else
  {
    this->SetUpProfiling();
  }

  return d->is_created;
}


//------------------------------------------------------------------------------
bool
OpenCLContext::Create( const std::list< OpenCLDevice > & devices )
{
  ITK_OPENCL_D( OpenCLContext );
  if( d->is_created )
  {
    return true;
  }
  if( devices.empty() )
  {
    this->ReportError( CL_INVALID_VALUE, __FILE__, __LINE__, ITK_LOCATION );
    return false;
  }
  std::vector< cl_device_id > devs;
  for( std::list< OpenCLDevice >::const_iterator dev = devices.begin();
    dev != devices.end(); ++dev )
  {
    devs.push_back( ( *dev ).GetDeviceId() );
  }

  cl_platform_id        platform = devices.front().GetPlatform().GetPlatformId();
  cl_context_properties props[]  = {
    CL_CONTEXT_PLATFORM,
    intptr_t( platform ),
    0
  };
  d->id = clCreateContext( props, devs.size(), &devs[ 0 ],
    opencl_context_notify, 0, &( d->last_error ) );
  d->is_created = ( d->id != 0 );
  if( !d->is_created )
  {
    itkOpenCLWarningMacro( << "OpenCLContext::Create:" << this->GetErrorName( d->last_error ) );
  }
  else
  {
    this->SetUpProfiling();
  }

  return d->is_created;
}


//------------------------------------------------------------------------------
bool
OpenCLContext::Create( const OpenCLContext::CreateMethod method )
{
  ITK_OPENCL_D( OpenCLContext );
  if( d->is_created )
  {
    return true;
  }

  if( method == OpenCLContext::Default )
  {
    std::list< OpenCLDevice > devices;
    const OpenCLDevice        gpuDevice
      = OpenCLDevice::GetMaximumFlopsDevice( OpenCLDevice::GPU );
    if( !gpuDevice.IsNull() )
    {
      devices.push_back( gpuDevice );
      this->CreateContext( devices, d );
    }
    else
    {
      const OpenCLDevice cpuDevice
        = OpenCLDevice::GetMaximumFlopsDevice( OpenCLDevice::CPU );
      if( !cpuDevice.IsNull() )
      {
        devices.push_back( cpuDevice );
        this->CreateContext( devices, d );
      }
      else
      {
        const OpenCLDevice acceleratorDevice
          = OpenCLDevice::GetMaximumFlopsDevice( OpenCLDevice::Accelerator );
        if( !acceleratorDevice.IsNull() )
        {
          devices.push_back( acceleratorDevice );
          this->CreateContext( devices, d );
        }
        else
        {
          itkGenericExceptionMacro( << "Unable to create OpenCLContext with method MultipleMaximumFlopsDevices." );
        }
      }
    }
  }
  else if( method == OpenCLContext::DevelopmentSingleMaximumFlopsDevice )
  {
    OpenCLDevice device;

#ifdef OPENCL_USE_INTEL
#ifdef OPENCL_USE_INTEL_GPU
    device = OpenCLDevice::GetMaximumFlopsDeviceByVendor( OpenCLDevice::GPU, OpenCLPlatform::Intel );
#elif OPENCL_USE_INTEL_CPU
    device = OpenCLDevice::GetMaximumFlopsDeviceByVendor( OpenCLDevice::CPU, OpenCLPlatform::Intel );
#else
    itkGenericExceptionMacro( << "Unknown Intel OpenCL platform." );
#endif

    // NVidia platform
#elif OPENCL_USE_NVIDIA
    device = OpenCLDevice::GetMaximumFlopsDeviceByVendor( OpenCLDevice::GPU, OpenCLPlatform::NVidia );
    // AMD platform
#elif OPENCL_USE_AMD
#ifdef OPENCL_USE_AMD_GPU
    device = OpenCLDevice::GetMaximumFlopsDeviceByVendor( OpenCLDevice::GPU, OpenCLPlatform::AMD );
#elif OPENCL_USE_AMD_CPU
    device = OpenCLDevice::GetMaximumFlopsDeviceByVendor( OpenCLDevice::CPU, OpenCLPlatform::AMD );
#else
    itkGenericExceptionMacro( << "Unknown AMD OpenCL platform." );
#endif

    // Unknown platform
#else
    itkGenericExceptionMacro( << "Not supported OpenCL platform by OpenCLContext." );
#endif

    std::list< OpenCLDevice > devices;
    devices.push_back( device );
    this->CreateContext( devices, d );
  }
  else if( method == OpenCLContext::DevelopmentMultipleMaximumFlopsDevices )
  {
    std::list< OpenCLDevice > devices;
    // Intel platform
#ifdef OPENCL_USE_INTEL
#ifdef OPENCL_USE_INTEL_GPU
    devices = OpenCLDevice::GetDevices( OpenCLDevice::GPU, OpenCLPlatform::Intel );
#elif OPENCL_USE_INTEL_CPU
    devices = OpenCLDevice::GetDevices( OpenCLDevice::CPU, OpenCLPlatform::Intel );
#else
    itkGenericExceptionMacro( << "Unknown Intel OpenCL platform." );
#endif

    // NVidia platform
#elif OPENCL_USE_NVIDIA
    devices = OpenCLDevice::GetDevices( OpenCLDevice::GPU, OpenCLPlatform::NVidia );
    // AMD platform
#elif OPENCL_USE_AMD
#ifdef OPENCL_USE_AMD_GPU
    devices = OpenCLDevice::GetDevices( OpenCLDevice::GPU, OpenCLPlatform::AMD );
#elif OPENCL_USE_AMD_CPU
    devices = OpenCLDevice::GetDevices( OpenCLDevice::CPU, OpenCLPlatform::AMD );
#else
    itkGenericExceptionMacro( << "Unknown AMD OpenCL platform." );
#endif

    // Unknown platform
#else
    itkGenericExceptionMacro( << "Not supported OpenCL platform by OpenCLContext." );
#endif
    this->CreateContext( devices, d );
  }
  else if( method == OpenCLContext::SingleMaximumFlopsDevice )
  {
    std::list< OpenCLDevice > devices;
    const OpenCLDevice        device = OpenCLDevice::GetMaximumFlopsDevice( OpenCLDevice::GPU );
    devices.push_back( device );
    this->CreateContext( devices, d );
  }
  else if( method == OpenCLContext::MultipleMaximumFlopsDevices )
  {
    std::list< OpenCLDevice > devices;
    devices = OpenCLDevice::GetMaximumFlopsDevices( OpenCLDevice::GPU );
    if( !devices.empty() )
    {
      this->CreateContext( devices, d );
    }
    else
    {
      devices = OpenCLDevice::GetMaximumFlopsDevices( OpenCLDevice::CPU );
      if( !devices.empty() )
      {
        this->CreateContext( devices, d );
      }
      else
      {
        devices = OpenCLDevice::GetMaximumFlopsDevices( OpenCLDevice::Accelerator );
        if( !devices.empty() )
        {
          this->CreateContext( devices, d );
        }
        else
        {
          itkGenericExceptionMacro( << "Unable to create OpenCLContext with method MultipleMaximumFlopsDevices." );
        }
      }
    }
  }

  // Check if OpenCL context has been created
  d->is_created = ( d->id != 0 );
  if( !d->is_created )
  {
    itkOpenCLWarningMacro( << "OpenCLContext::Create(method:" << int(method) << "):"
                           << this->GetErrorName( d->last_error ) );
  }
  else
  {
    this->SetUpProfiling();
  }

  return d->is_created;
}


//------------------------------------------------------------------------------
bool
OpenCLContext::Create( const OpenCLPlatform & platfrom,
  const OpenCLDevice::DeviceType type )
{
  ITK_OPENCL_D( OpenCLContext );
  if( d->is_created )
  {
    return true;
  }

  this->CreateContext( platfrom, type, d );

  // Check if OpenCL context has been created
  d->is_created = ( d->id != 0 );
  if( !d->is_created )
  {
    itkOpenCLWarningMacro( << "OpenCLContext::Create(platfrom id:" << platfrom.GetPlatformId() << "):"
                           << this->GetErrorName( d->last_error ) );
  }
  else
  {
    this->SetUpProfiling();
  }

  return d->is_created;
}


//------------------------------------------------------------------------------
bool
OpenCLContext::Create()
{
  ITK_OPENCL_D( OpenCLContext );
  if( d->is_created )
  {
    return true;
  }

  OpenCLPlatform platform;

#ifdef OPENCL_USE_INTEL
#ifdef OPENCL_USE_INTEL_GPU
  platform = OpenCLPlatform::GetPlatform( OpenCLPlatform::Intel );
  this->CreateContext( platform, OpenCLDevice::GPU, d );
#elif OPENCL_USE_INTEL_CPU
  platform = OpenCLPlatform::GetPlatform( OpenCLPlatform::Intel );
  this->CreateContext( platform, OpenCLDevice::CPU, d );
#else
  itkGenericExceptionMacro( << "Unknown Intel OpenCL platform." );
#endif

  // NVidia platform
#elif OPENCL_USE_NVIDIA
  platform = OpenCLPlatform::GetPlatform( OpenCLPlatform::NVidia );
  this->CreateContext( platform, OpenCLDevice::GPU, d );
  // AMD platform
#elif OPENCL_USE_AMD
#ifdef OPENCL_USE_AMD_GPU
  platform = OpenCLPlatform::GetPlatform( OpenCLPlatform::AMD );
  this->CreateContext( platform, OpenCLDevice::GPU, d );
#elif OPENCL_USE_AMD_CPU
  platform = OpenCLPlatform::GetPlatform( OpenCLPlatform::AMD );
  this->CreateContext( platform, OpenCLDevice::CPU, d );
#else
  itkGenericExceptionMacro( << "Unknown AMD OpenCL platform." );
#endif

  // Unknown platform
#else
  itkGenericExceptionMacro( << "Not supported OpenCL platform by OpenCLContext." );
#endif

  // Check if OpenCL context has been created
  d->is_created = ( d->id != 0 );
  if( !d->is_created )
  {
    itkOpenCLWarningMacro( << "OpenCLContext::Create():"
                           << this->GetErrorName( d->last_error ) );
  }
  else
  {
    this->SetUpProfiling();
  }

  return d->is_created;
}


//------------------------------------------------------------------------------
// \internal
void
OpenCLContext::CreateContext( const std::list< OpenCLDevice > & devices,
  OpenCLContextPimpl * d )
{
  if( d->is_created )
  {
    return;
  }

  if( !devices.empty() )
  {
    std::vector< cl_device_id > devs;
    for( std::list< OpenCLDevice >::const_iterator dev = devices.begin();
      dev != devices.end(); ++dev )

    {
      devs.push_back( ( *dev ).GetDeviceId() );
    }

    cl_context_properties props[] = {
      CL_CONTEXT_PLATFORM,
      cl_context_properties( devices.front().GetPlatform().GetPlatformId() ),
      0
    };
    d->id = clCreateContext
        ( props, devs.size(), &devs[ 0 ],
        opencl_context_notify, 0, &( d->last_error ) );
  }
  else
  {
    d->last_error = CL_DEVICE_NOT_FOUND;
    d->id         = 0;
  }
}


//------------------------------------------------------------------------------
void
OpenCLContext::CreateContext( const OpenCLPlatform & platform,
  const OpenCLDevice::DeviceType type, OpenCLContextPimpl * d )
{
  if( d->is_created )
  {
    return;
  }

  if( !platform.IsNull() )
  {
    cl_context_properties props[] =
    {
      CL_CONTEXT_PLATFORM,
      (cl_context_properties)( platform.GetPlatformId() ),
      0
    };
    cl_device_type device_type = cl_device_type( type );
    d->id = clCreateContextFromType
        ( props, device_type,
        opencl_context_notify, 0, &( d->last_error ) );
  }
  else
  {
    d->last_error = CL_DEVICE_NOT_FOUND;
    d->id         = 0;
  }
}


//------------------------------------------------------------------------------
void
OpenCLContext::Release()
{
  ITK_OPENCL_D( OpenCLContext );
  if( d->is_created )
  {
    d->command_queue         = OpenCLCommandQueue();
    d->default_command_queue = OpenCLCommandQueue();
    clReleaseContext( d->id );
    d->id             = 0;
    d->default_device = OpenCLDevice();
    d->is_created     = false;
  }
}


//------------------------------------------------------------------------------
cl_context
OpenCLContext::GetContextId() const
{
  ITK_OPENCL_D( const OpenCLContext );
  return d->id;
}


//------------------------------------------------------------------------------
void
OpenCLContext::SetContextId( cl_context id )
{
  ITK_OPENCL_D( OpenCLContext );
  if( d->id == id || !id )
  {
    return;
  }
  this->Release();
  clRetainContext( id );
  d->id         = id;
  d->is_created = true;
}


//------------------------------------------------------------------------------
std::list< OpenCLDevice > OpenCLContext::GetDevices() const
{
  ITK_OPENCL_D( const OpenCLContext );
  std::list< OpenCLDevice > devs;
  if( d->is_created )
  {
    std::size_t size = 0;
    if( clGetContextInfo( d->id, CL_CONTEXT_DEVICES, 0, 0, &size ) == CL_SUCCESS && size > 0 )
    {
      std::vector< cl_device_id > buf( size );
      if( clGetContextInfo( d->id, CL_CONTEXT_DEVICES, size, &buf[ 0 ], 0 ) == CL_SUCCESS )
      {
        for( std::size_t index = 0; index < size; ++index )
        {
          if( buf[ index ] != 0 )
          {
            devs.push_back( OpenCLDevice( buf[ index ] ) );
          }
        }
      }
    }
  }
  return devs;
}


//------------------------------------------------------------------------------
OpenCLDevice
OpenCLContext::GetDefaultDevice() const
{
  ITK_OPENCL_D( const OpenCLContext );
  if( d->is_created )
  {
    if( !d->default_device.IsNull() )
    {
      return d->default_device;
    }
    std::size_t size = 0;
    if( clGetContextInfo( d->id, CL_CONTEXT_DEVICES, 0, 0, &size )
      == CL_SUCCESS && size > 0 )
    {
      std::vector< cl_device_id > buf( size );
      if( clGetContextInfo( d->id, CL_CONTEXT_DEVICES, size, &buf[ 0 ], 0 ) == CL_SUCCESS )
      {
        return OpenCLDevice( buf[ 0 ] );
      }
    }
  }
  return OpenCLDevice();
}


//------------------------------------------------------------------------------
cl_int
OpenCLContext::GetLastError() const
{
  ITK_OPENCL_D( const OpenCLContext );
  return d->last_error;
}


//------------------------------------------------------------------------------
void
OpenCLContext::SetLastError( const cl_int error )
{
  ITK_OPENCL_D( OpenCLContext );
  d->last_error = error;
}


//------------------------------------------------------------------------------
std::string
OpenCLContext::GetErrorName( const cl_int code )
{
  static const char * errorString[] = {
    "CL_SUCCESS",                                   // 0
    "CL_DEVICE_NOT_FOUND",                          // -1
    "CL_DEVICE_NOT_AVAILABLE",                      // -2
    "CL_COMPILER_NOT_AVAILABLE",                    // -3
    "CL_MEM_OBJECT_ALLOCATION_FAILURE",             // -4
    "CL_OUT_OF_RESOURCES",                          // -5
    "CL_OUT_OF_HOST_MEMORY",                        // -6
    "CL_PROFILING_INFO_NOT_AVAILABLE",              // -7
    "CL_MEM_COPY_OVERLAP",                          // -8
    "CL_IMAGE_FORMAT_MISMATCH",                     // -9
    "CL_IMAGE_FORMAT_NOT_SUPPORTED",                // -10
    "CL_BUILD_PROGRAM_FAILURE",                     // -11
    "CL_MAP_FAILURE",                               // -12
    "CL_MISALIGNED_SUB_BUFFER_OFFSET",              // -13
    "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST", // -14
    "CL_COMPILE_PROGRAM_FAILURE",                   // -15
    "CL_LINKER_NOT_AVAILABLE",                      // -16
    "CL_LINK_PROGRAM_FAILURE",                      // -17
    "CL_DEVICE_PARTITION_FAILED",                   // -18
    "CL_KERNEL_ARG_INFO_NOT_AVAILABLE",             // -19
    "",                                             // -20
    "",                                             // -21
    "",                                             // -22
    "",                                             // -23
    "",                                             // -24
    "",                                             // -25
    "",                                             // -26
    "",                                             // -27
    "",                                             // -28
    "",                                             // -29
    "CL_INVALID_VALUE",                             // -30
    "CL_INVALID_DEVICE_TYPE",                       // -31
    "CL_INVALID_PLATFORM",                          // -32
    "CL_INVALID_DEVICE",                            // -33
    "CL_INVALID_CONTEXT",                           // -34
    "CL_INVALID_QUEUE_PROPERTIES",                  // -35
    "CL_INVALID_COMMAND_QUEUE",                     // -36
    "CL_INVALID_HOST_PTR",                          // -37
    "CL_INVALID_MEM_OBJECT",                        // -38
    "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",           // -39
    "CL_INVALID_IMAGE_SIZE",                        // -40
    "CL_INVALID_SAMPLER",                           // -41
    "CL_INVALID_BINARY",                            // -42
    "CL_INVALID_BUILD_OPTIONS",                     // -43
    "CL_INVALID_PROGRAM",                           // -44
    "CL_INVALID_PROGRAM_EXECUTABLE",                // -45
    "CL_INVALID_KERNEL_NAME",                       // -46
    "CL_INVALID_KERNEL_DEFINITION",                 // -47
    "CL_INVALID_KERNEL",                            // -48
    "CL_INVALID_ARG_INDEX",                         // -49
    "CL_INVALID_ARG_VALUE",                         // -50
    "CL_INVALID_ARG_SIZE",                          // -51
    "CL_INVALID_KERNEL_ARGS",                       // -52
    "CL_INVALID_WORK_DIMENSION",                    // -53
    "CL_INVALID_WORK_GROUP_SIZE",                   // -54
    "CL_INVALID_WORK_ITEM_SIZE",                    // -55
    "CL_INVALID_GLOBAL_OFFSET",                     // -56
    "CL_INVALID_EVENT_WAIT_LIST",                   // -57
    "CL_INVALID_EVENT",                             // -58
    "CL_INVALID_OPERATION",                         // -59
    "CL_INVALID_GL_OBJECT",                         // -60
    "CL_INVALID_BUFFER_SIZE",                       // -61
    "CL_INVALID_MIP_LEVEL",                         // -62
    "CL_INVALID_GLOBAL_WORK_SIZE",                  // -63
    "CL_INVALID_PROPERTY",                          // -64
    "CL_INVALID_IMAGE_DESCRIPTOR",                  // -65
    "CL_INVALID_COMPILER_OPTIONS",                  // -66
    "CL_INVALID_LINKER_OPTIONS",                    // -67
    "CL_INVALID_DEVICE_PARTITION_COUNT",            // -68
  };

  if( code != CL_SUCCESS )
  {
    const int errorCount = sizeof( errorString ) / sizeof( errorString[ 0 ] );
    const int index      = -code;
    if( index >= 0 && index < errorCount )
    {
      return errorString[ index ];
    }
    else
    {
      return "Unspecified Error";
    }
  }
  else
  {
    return "No Error";
  }
}


//------------------------------------------------------------------------------
OpenCLCommandQueue
OpenCLContext::GetCommandQueue()
{
  ITK_OPENCL_D( OpenCLContext );
  if( !d->command_queue.IsNull() )
  {
    return d->command_queue;
  }
  else
  {
    return this->GetDefaultCommandQueue();
  }
}


//------------------------------------------------------------------------------
void
OpenCLContext::SetCommandQueue( const OpenCLCommandQueue & queue )
{
  ITK_OPENCL_D( OpenCLContext );
  d->command_queue = queue;
}


//------------------------------------------------------------------------------
OpenCLCommandQueue
OpenCLContext::GetDefaultCommandQueue()
{
  ITK_OPENCL_D( OpenCLContext );
  if( d->default_command_queue.IsNull() )
  {
    if( !d->is_created )
    {
      return OpenCLCommandQueue();
    }
    OpenCLDevice dev = this->GetDefaultDevice();
    if( dev.IsNull() )
    {
      return OpenCLCommandQueue();
    }
    cl_command_queue queue;
#ifdef OPENCL_PROFILING
    queue = clCreateCommandQueue( d->id, dev.GetDeviceId(), CL_QUEUE_PROFILING_ENABLE, &( d->last_error ) );
#else
    queue = clCreateCommandQueue( d->id, dev.GetDeviceId(), 0, &( d->last_error ) );
#endif

    if( !queue )
    {
      itkOpenCLWarningMacro( << "OpenCLContext::GetDefaultCommandQueue:"
                             << this->GetErrorName( d->last_error ) );
      return OpenCLCommandQueue();
    }
    d->default_command_queue = OpenCLCommandQueue( this, queue );
  }
  return d->default_command_queue;
}


//------------------------------------------------------------------------------
// Returns the active queue handle without incurring retain/release overhead.
cl_command_queue
OpenCLContext::GetActiveQueue()
{
  ITK_OPENCL_D( OpenCLContext );
  cl_command_queue queue = d->command_queue.GetQueueId();
  if( queue )
  {
    return queue;
  }
  queue = d->default_command_queue.GetQueueId();
  if( queue )
  {
    return queue;
  }
  return this->GetDefaultCommandQueue().GetQueueId();
}


//------------------------------------------------------------------------------
OpenCLCommandQueue
OpenCLContext::CreateCommandQueue( const cl_command_queue_properties properties,
  const OpenCLDevice & device )
{
  ITK_OPENCL_D( OpenCLContext );
  cl_command_queue queue;
  if( device.IsNull() )
  {
    queue = clCreateCommandQueue( d->id, this->GetDefaultDevice().GetDeviceId(),
      properties, &( d->last_error ) );
  }
  else
  {
    queue = clCreateCommandQueue( d->id, device.GetDeviceId(),
      properties, &( d->last_error ) );
  }
  this->ReportError( d->last_error, __FILE__, __LINE__, ITK_LOCATION );
  if( queue )
  {
    return OpenCLCommandQueue( this, queue );
  }
  else
  {
    return OpenCLCommandQueue();
  }
}


//------------------------------------------------------------------------------
OpenCLBuffer
OpenCLContext::CreateBufferDevice( const OpenCLMemoryObject::Access access,
  const std::size_t size )
{
  if( size == 0 )
  {
    return OpenCLBuffer();
  }

  ITK_OPENCL_D( OpenCLContext );
  cl_mem_flags flags = cl_mem_flags( access );
  cl_mem       mem   = clCreateBuffer( d->id, flags, size, 0, &( d->last_error ) );
  this->ReportError( d->last_error, __FILE__, __LINE__, ITK_LOCATION );
  if( mem )
  {
    return OpenCLBuffer( this, mem );
  }
  else
  {
    return OpenCLBuffer();
  }
}


//------------------------------------------------------------------------------
OpenCLBuffer
OpenCLContext::CreateBufferHost( void * data,
  const OpenCLMemoryObject::Access access, const std::size_t size )
{
  if( size == 0 )
  {
    return OpenCLBuffer();
  }

  ITK_OPENCL_D( OpenCLContext );
  cl_mem_flags flags = cl_mem_flags( access );
  if( data )
  {
    flags |= CL_MEM_USE_HOST_PTR;
  }
  else
  {
    flags |= CL_MEM_ALLOC_HOST_PTR;
  }
  cl_mem mem = clCreateBuffer( d->id, flags, size, data, &( d->last_error ) );
  this->ReportError( d->last_error, __FILE__, __LINE__, ITK_LOCATION );
  if( mem )
  {
    return OpenCLBuffer( this, mem );
  }
  else
  {
    return OpenCLBuffer();
  }
}


//------------------------------------------------------------------------------
OpenCLBuffer
OpenCLContext::CreateBufferCopy( const void * data,
  const OpenCLMemoryObject::Access access, const std::size_t size )
{
  if( size == 0 )
  {
    return OpenCLBuffer();
  }

  ITK_OPENCL_D( OpenCLContext );
  cl_mem_flags flags = cl_mem_flags( access );
  flags |= CL_MEM_COPY_HOST_PTR;
  cl_mem mem = clCreateBuffer
      ( d->id, flags, size, const_cast< void * >( data ), &( d->last_error ) );
  this->ReportError( d->last_error, __FILE__, __LINE__, ITK_LOCATION );
  if( mem )
  {
    return OpenCLBuffer( this, mem );
  }
  else
  {
    return OpenCLBuffer();
  }
}


//------------------------------------------------------------------------------
OpenCLImage
OpenCLContext::CreateImageDevice( const OpenCLImageFormat & format,
  const OpenCLMemoryObject::Access access,
  const OpenCLSize & size )
{
  if( size.IsZero() )
  {
    return OpenCLImage();
  }

  ITK_OPENCL_D( OpenCLContext );
  cl_mem_flags flags = cl_mem_flags( access );

#ifdef CL_VERSION_1_2
  // Define image description
  cl_image_desc imageDesc;
  OpenCLImage::SetImageDescription( imageDesc, format, size );
  cl_mem mem = clCreateImage
      ( d->id, flags, &( format.m_Format ), &imageDesc, 0, &( d->last_error ) );
#else
  cl_mem mem = NULL;
  if( size.GetDimension() == 1 )
  {
    itkGenericExceptionMacro( << "OpenCLContext::CreateImageDevice() not supported for 1D." );
  }
  else if( size.GetDimension() == 2 )
  {
    mem = clCreateImage2D
        ( d->id, flags, &( format.m_Format ), size[ 0 ], size[ 1 ],
        0, 0, &( d->last_error ) );
  }
  else if( size.GetDimension() == 3 )
  {
    clCreateImage3D
      ( d->id, flags, &( format.m_Format ), size[ 0 ], size[ 1 ], size[ 2 ],
      0, 0, 0, &( d->last_error ) );
  }
#endif

  this->ReportError( d->last_error, __FILE__, __LINE__, ITK_LOCATION );
  if( mem )
  {
    return OpenCLImage( this, mem );
  }
  else
  {
    return OpenCLImage();
  }
}


//------------------------------------------------------------------------------
OpenCLImage
OpenCLContext::CreateImageHost( const OpenCLImageFormat & format,
  void * data, const OpenCLSize & size,
  const OpenCLMemoryObject::Access access )
{
  if( size.IsZero() )
  {
    return OpenCLImage();
  }

  ITK_OPENCL_D( OpenCLContext );
  cl_mem_flags flags = cl_mem_flags( access );
  if( data )
  {
    flags |= CL_MEM_USE_HOST_PTR;
  }
  else
  {
    flags |= CL_MEM_ALLOC_HOST_PTR;
  }

#ifdef CL_VERSION_1_2
  // Define image description
  cl_image_desc imageDesc;
  OpenCLImage::SetImageDescription( imageDesc, format, size );
  cl_mem mem = clCreateImage
      ( d->id, flags, &( format.m_Format ), &imageDesc, data, &( d->last_error ) );
#else
  cl_mem mem = NULL;
  if( size.GetDimension() == 1 )
  {
    itkGenericExceptionMacro( << "OpenCLContext::CreateImageHost() not supported for 1D." );
  }
  else if( size.GetDimension() == 2 )
  {
    mem = clCreateImage2D
        ( d->id, flags, &( format.m_Format ), size[ 0 ], size[ 1 ],
        0, data, &( d->last_error ) );
  }
  else if( size.GetDimension() == 3 )
  {
    clCreateImage3D
      ( d->id, flags, &( format.m_Format ), size[ 0 ], size[ 1 ], size[ 2 ],
      0, 0, data, &( d->last_error ) );
  }
#endif

  this->ReportError( d->last_error, __FILE__, __LINE__, ITK_LOCATION );
  if( mem )
  {
    return OpenCLImage( this, mem );
  }
  else
  {
    return OpenCLImage();
  }
}


//------------------------------------------------------------------------------
OpenCLImage
OpenCLContext::CreateImageCopy( const OpenCLImageFormat & format,
  const void * data, const OpenCLSize & size,
  const OpenCLMemoryObject::Access access )
{
  if( size.IsZero() )
  {
    return OpenCLImage();
  }

  ITK_OPENCL_D( OpenCLContext );
  cl_mem_flags flags = cl_mem_flags( access ) | CL_MEM_COPY_HOST_PTR;

#ifdef CL_VERSION_1_2
  // Define image description
  cl_image_desc imageDesc;
  OpenCLImage::SetImageDescription( imageDesc, format, size );
  cl_mem mem = clCreateImage
      ( d->id, flags, &( format.m_Format ), &imageDesc,
      const_cast< void * >( data ), &( d->last_error ) );
#else
  cl_mem mem = NULL;
  if( size.GetDimension() == 1 )
  {
    itkGenericExceptionMacro( << "OpenCLContext::CreateImageCopy() not supported for 1D." );
  }
  else if( size.GetDimension() == 2 )
  {
    mem = clCreateImage2D
        ( d->id, flags, &( format.m_Format ), size[ 0 ], size[ 1 ],
        0, const_cast< void * >( data ), &( d->last_error ) );
  }
  else if( size.GetDimension() == 3 )
  {
    clCreateImage3D
      ( d->id, flags, &( format.m_Format ), size[ 0 ], size[ 1 ], size[ 2 ],
      0, 0, const_cast< void * >( data ), &( d->last_error ) );
  }
#endif

  this->ReportError( d->last_error, __FILE__, __LINE__, ITK_LOCATION );
  if( mem )
  {
    return OpenCLImage( this, mem );
  }
  else
  {
    return OpenCLImage();
  }
}


//------------------------------------------------------------------------------
OpenCLProgram
OpenCLContext::CreateProgramFromSourceCode( const std::string & sourceCode,
  const std::string & prefixSourceCode,
  const std::string & postfixSourceCode )
{
  if( sourceCode.empty() )
  {
    itkOpenCLWarningMacro( << "The source code is empty for the OpenCL program." );
    return OpenCLProgram();
  }

#ifdef OPENCL_PROFILING
  itk::OpenCLProfilingTimeProbe timer( "Creating OpenCL program using clCreateProgramWithSource" );
#endif

  std::stringstream sstream;

  // Prepends prefix source code if provided
  if( !prefixSourceCode.empty() )
  {
    sstream << prefixSourceCode << std::endl;
  }

  // Add the main source code
  sstream << sourceCode;

  // Appends postfix source code if provided
  if( !postfixSourceCode.empty() )
  {
    sstream << std::endl << postfixSourceCode;
  }

  const std::string oclSource     = sstream.str();
  const std::size_t oclSourceSize = oclSource.size();

  if( oclSourceSize == 0 )
  {
    itkOpenCLWarningMacro( << "Cannot build OpenCL brogram from empty source." );
    return OpenCLProgram();
  }

#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  // To work with the Intel SDK for OpenCL* - Debugger plug-in, the OpenCL*
  // kernel code must exist in a text file separate from the code of the host.
  // Also the full path to the file has to be provided.
  const std::string fileName = GetOpenCLDebugFileName( oclSource );
  if( prefixSourceCode != "" )
  {
    std::ofstream debugfile( fileName.c_str() );
    if( debugfile.is_open() == false )
    {
      itkOpenCLWarningMacro( << "Cannot create OpenCL debug source file: " << fileName );
      return OpenCLProgram();
    }
    debugfile << oclSource;
    debugfile.close();

    itkOpenCLWarningMacro( << "For Debugging your OpenCL kernel use:\n" << fileName );
  }

  std::cout << "Creating OpenCL program from source." << std::endl;
#endif

  // Create OpenCL program and return
#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  return this->CreateOpenCLProgram( fileName, oclSource, oclSourceSize );
#else
  return this->CreateOpenCLProgram( std::string(), oclSource, oclSourceSize );
#endif
}


//------------------------------------------------------------------------------
OpenCLProgram
OpenCLContext::CreateProgramFromSourceFile( const std::string & filename,
  const std::string & prefixSourceCode,
  const std::string & postfixSourceCode )
{
  if( filename.empty() )
  {
    itkOpenCLWarningMacro( << "The filename must be specified." );
    return OpenCLProgram();
  }

  // open the file
  std::ifstream inputFile( filename.c_str(), std::ifstream::in | std::ifstream::binary );
  if( inputFile.is_open() == false )
  {
    itkOpenCLWarningMacro( << "Cannot open OpenCL source file: " << filename );
    return OpenCLProgram();
  }

  std::stringstream sstream;

  // Prepends prefix source code if provided
  if( !prefixSourceCode.empty() )
  {
    sstream << prefixSourceCode << std::endl;
  }

  // Add the main source code
  sstream << inputFile.rdbuf();

  // Appends postfix source code if provided
  if( !postfixSourceCode.empty() )
  {
    sstream << std::endl << postfixSourceCode;
  }

  inputFile.close();

  const std::string oclSource     = sstream.str();
  const std::size_t oclSourceSize = oclSource.size();

  if( oclSourceSize == 0 )
  {
    itkOpenCLWarningMacro( << "Cannot build OpenCL source file: " << filename << " is empty." );
    return OpenCLProgram();
  }

  std::string fileName( filename );

#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  // To work with the Intel SDK for OpenCL* - Debugger plug-in, the OpenCL*
  // kernel code must exist in a text file separate from the code of the host.
  // Also the full path to the file has to be provided.
  if( prefixSourceCode != "" )
  {
    fileName = GetOpenCLDebugFileName( oclSource );
  }
  if( prefixSourceCode != "" )
  {
    std::ofstream debugfile( fileName.c_str() );
    if( debugfile.is_open() == false )
    {
      itkOpenCLWarningMacro( << "Cannot create OpenCL debug source file: " << fileName );
      return OpenCLProgram();
    }
    debugfile << oclSource;
    debugfile.close();

    itkOpenCLWarningMacro( << "For Debugging your OpenCL kernel use:\n"
                           << fileName << " , not original .cl file." );
  }

  std::cout << "Creating OpenCL program from : " << fileName << std::endl;
#endif

  // Create OpenCL program and return
  return this->CreateOpenCLProgram( fileName, oclSource, oclSourceSize );
}


//------------------------------------------------------------------------------
OpenCLProgram
OpenCLContext::CreateOpenCLProgram( const std::string & filename,
  const std::string & source,
  const std::size_t sourceSize )
{
  if( source.empty() )
  {
    itkOpenCLWarningMacro( << "The source is empty for the OpenCL program in filename: '"
                           << filename << "'" );
    return OpenCLProgram();
  }

#ifdef OPENCL_PROFILING
  itk::OpenCLProfilingTimeProbe timer( "Creating OpenCL program using clCreateProgramWithSource" );
#endif

  ITK_OPENCL_D( OpenCLContext );
  const char * code = source.c_str();

  this->OpenCLDebug( "clCreateProgramWithSource" );
  cl_program program = clCreateProgramWithSource( d->id, 1, &code, &sourceSize, &( d->last_error ) );
  this->ReportError( d->last_error, __FILE__, __LINE__, ITK_LOCATION );

  if( d->last_error != CL_SUCCESS )
  {
    itkOpenCLWarningMacro( "Cannot create OpenCL program, filename: '" << filename << "'" );
    return OpenCLProgram();
  }
  else
  {
    return OpenCLProgram( this, program, filename );
  }
}


//------------------------------------------------------------------------------
OpenCLProgram
OpenCLContext::CreateProgramFromBinaryCode( const unsigned char * binary,
  const std::size_t size )
{
  ITK_OPENCL_D( OpenCLContext );
  cl_device_id device = GetDefaultDevice().GetDeviceId();

  this->OpenCLDebug( "clCreateProgramWithBinary" );
  cl_program program = clCreateProgramWithBinary( d->id, 1, &device, &size,
    &binary, 0, &( d->last_error ) );
  this->ReportError( d->last_error, __FILE__, __LINE__, ITK_LOCATION );
  if( d->last_error != CL_SUCCESS )
  {
    itkOpenCLWarningMacro( "Cannot create OpenCL program from binary." );
    return OpenCLProgram();
  }
  else
  {
    return OpenCLProgram( this, program );
  }
}


//------------------------------------------------------------------------------
OpenCLProgram
OpenCLContext::BuildProgramFromSourceCode( const std::string & sourceCode,
  const std::string & prefixSourceCode,
  const std::string & postfixSourceCode )
{
  OpenCLProgram program = this->CreateProgramFromSourceCode( sourceCode,
    prefixSourceCode, postfixSourceCode );

  if( program.IsNull() || program.Build() )
  {
    return program;
  }
  return OpenCLProgram();
}


//------------------------------------------------------------------------------
OpenCLProgram
OpenCLContext::BuildProgramFromSourceCode( const std::list< OpenCLDevice > & devices,
  const std::string & sourceCode,
  const std::string & prefixSourceCode,
  const std::string & postfixSourceCode,
  const std::string & extraBuildOptions )
{
  OpenCLProgram program = this->CreateProgramFromSourceCode( sourceCode,
    prefixSourceCode, postfixSourceCode );

  if( program.IsNull() || program.Build( devices, extraBuildOptions ) )
  {
    return program;
  }
  return OpenCLProgram();
}


//------------------------------------------------------------------------------
OpenCLProgram
OpenCLContext::BuildProgramFromSourceFile( const std::string & filename,
  const std::string & prefixSourceCode,
  const std::string & postfixSourceCode )
{
  OpenCLProgram program = this->CreateProgramFromSourceFile( filename,
    prefixSourceCode, postfixSourceCode );

  if( program.IsNull() || program.Build() )
  {
    return program;
  }
  return OpenCLProgram();
}


//------------------------------------------------------------------------------
OpenCLProgram
OpenCLContext::BuildProgramFromSourceFile( const std::list< OpenCLDevice > & devices,
  const std::string & fileName,
  const std::string & prefixSourceCode,
  const std::string & postfixSourceCode,
  const std::string & extraBuildOptions )
{
  OpenCLProgram program = this->CreateProgramFromSourceFile( fileName,
    prefixSourceCode, postfixSourceCode );

  if( program.IsNull() || program.Build( devices, extraBuildOptions ) )
  {
    return program;
  }
  return OpenCLProgram();
}


//------------------------------------------------------------------------------
std::list< OpenCLImageFormat > open_cl_get_supported_image_formats(
  const cl_context ctx,
  const cl_mem_flags flags,
  const cl_mem_object_type image_type )
{
  cl_uint count = 0;

  std::list< OpenCLImageFormat > list;

  if( clGetSupportedImageFormats
      ( ctx, flags, image_type,
    0, 0, &count ) != CL_SUCCESS || !count )
  {
    return list;
  }
  std::vector< cl_image_format > buf( count );
  if( clGetSupportedImageFormats
      ( ctx, flags, image_type,
    count, &buf[ 0 ], 0 ) != CL_SUCCESS )
  {
    return list;
  }
  for( cl_uint index = 0; index < count; ++index )
  {
    list.push_back( OpenCLImageFormat
        ( OpenCLImageFormat::ChannelOrder( buf[ index ].image_channel_order ),
      OpenCLImageFormat::ChannelType( buf[ index ].image_channel_data_type ) ) );
  }
  return list;
}


//------------------------------------------------------------------------------
std::list< OpenCLImageFormat > OpenCLContext::GetSupportedImageFormats(
  const OpenCLImageFormat::ImageType image_type, const cl_mem_flags flags ) const
{
  ITK_OPENCL_D( const OpenCLContext );

  std::list< OpenCLImageFormat > list_image_formats;
  switch( image_type )
  {
    case OpenCLImageFormat::BUFFER:
      list_image_formats = open_cl_get_supported_image_formats(
      d->id, flags, CL_MEM_OBJECT_BUFFER );
      break;
    case OpenCLImageFormat::IMAGE2D:
      list_image_formats =  open_cl_get_supported_image_formats(
      d->id, flags, CL_MEM_OBJECT_IMAGE2D );
      break;
    case OpenCLImageFormat::IMAGE3D:
      list_image_formats =  open_cl_get_supported_image_formats(
      d->id, flags, CL_MEM_OBJECT_IMAGE3D );
      break;
    case OpenCLImageFormat::IMAGE2D_ARRAY:
      list_image_formats =  open_cl_get_supported_image_formats(
      d->id, flags, CL_MEM_OBJECT_IMAGE2D_ARRAY );
      break;
    case OpenCLImageFormat::IMAGE1D:
      list_image_formats =  open_cl_get_supported_image_formats(
      d->id, flags, CL_MEM_OBJECT_IMAGE1D );
      break;
    case OpenCLImageFormat::IMAGE1D_ARRAY:
      list_image_formats = open_cl_get_supported_image_formats(
      d->id, flags, CL_MEM_OBJECT_IMAGE1D_ARRAY );
      break;
    case OpenCLImageFormat::IMAGE1D_BUFFER:
      list_image_formats =  open_cl_get_supported_image_formats(
      d->id, flags, CL_MEM_OBJECT_IMAGE1D_BUFFER );
      break;
  }

  // Compose the final list
  std::list< OpenCLImageFormat > list;
  for( std::list< OpenCLImageFormat >::const_iterator it = list_image_formats.begin();
    it != list_image_formats.end(); ++it )
  {
    list.push_back( OpenCLImageFormat(
      OpenCLImageFormat::ImageType( image_type ),
      OpenCLImageFormat::ChannelOrder( ( *it ).GetChannelOrder() ),
      OpenCLImageFormat::ChannelType( ( *it ).GetChannelType() ) ) );
  }

  return list;
}


//------------------------------------------------------------------------------
OpenCLSampler
OpenCLContext::CreateSampler( const bool normalizedCoordinates,
  const OpenCLSampler::AddressingMode addressingMode,
  const OpenCLSampler::FilterMode filterMode )
{
  ITK_OPENCL_D( OpenCLContext );
  cl_sampler sampler = clCreateSampler
      ( d->id, normalizedCoordinates ? CL_TRUE : CL_FALSE,
      cl_addressing_mode( addressingMode ),
      cl_filter_mode( filterMode ), &( d->last_error ) );
  this->ReportError( d->last_error, __FILE__, __LINE__, ITK_LOCATION );
  if( sampler )
  {
    return OpenCLSampler( this, sampler );
  }
  else
  {
    return OpenCLSampler();
  }
}


//------------------------------------------------------------------------------
OpenCLUserEvent
OpenCLContext::CreateUserEvent()
{
  ITK_OPENCL_D( OpenCLContext );
  cl_event event = clCreateUserEvent( d->id, &( d->last_error ) );
  this->ReportError( d->last_error, __FILE__, __LINE__, ITK_LOCATION );
  return OpenCLUserEvent( event, true );
}


//------------------------------------------------------------------------------
void
OpenCLContext::Flush()
{
  clFlush( this->GetActiveQueue() );
}


//------------------------------------------------------------------------------
void
OpenCLContext::Finish()
{
  clFinish( this->GetActiveQueue() );
}


//------------------------------------------------------------------------------
cl_int
OpenCLContext::Marker( const OpenCLEventList & event_list )
{
  if( event_list.IsEmpty() )
  {
    return CL_SUCCESS;
  }

  cl_event event;

#ifdef CL_VERSION_1_2
  const cl_int error = clEnqueueMarkerWithWaitList(
    this->GetActiveQueue(), event_list.GetSize(), event_list.GetEventData(), &event );
#else
  const cl_int error = clEnqueueMarker( this->GetActiveQueue(), &event );
#endif

  this->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  if( error == CL_SUCCESS )
  {
    clWaitForEvents( 1, &event );
    clReleaseEvent( event );
  }
  return error;
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLContext::MarkerAsync( const OpenCLEventList & event_list )
{
  if( event_list.IsEmpty() )
  {
    return OpenCLEvent();
  }

  cl_event event;

#ifdef CL_VERSION_1_2
  const cl_int error = clEnqueueMarkerWithWaitList(
    this->GetActiveQueue(), event_list.GetSize(), event_list.GetEventData(), &event );
#else
  const cl_int error = clEnqueueMarker( this->GetActiveQueue(), &event );
#endif

  this->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  if( error != CL_SUCCESS )
  {
    return OpenCLEvent();
  }
  else
  {
    return OpenCLEvent( event );
  }
}


//------------------------------------------------------------------------------
cl_int
OpenCLContext::Barrier( const OpenCLEventList & event_list )
{
  if( event_list.IsEmpty() )
  {
    return CL_SUCCESS;
  }

#ifdef CL_VERSION_1_2
  cl_event     event;
  const cl_int error = clEnqueueBarrierWithWaitList(
    this->GetActiveQueue(), event_list.GetSize(), event_list.GetEventData(), &event );
#else
  const cl_int error = clEnqueueBarrier( this->GetActiveQueue() );
#endif

  this->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );

#ifdef CL_VERSION_1_2
  if( error == CL_SUCCESS )
  {
    clWaitForEvents( 1, &event );
    clReleaseEvent( event );
  }
  return error;
#else
  return CL_SUCCESS;
#endif
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLContext::BarrierAsync( const OpenCLEventList & event_list )
{
  if( event_list.IsEmpty() )
  {
    return OpenCLEvent();
  }

#ifdef CL_VERSION_1_2
  cl_event     event;
  const cl_int error = clEnqueueBarrierWithWaitList(
    this->GetActiveQueue(), event_list.GetSize(), event_list.GetEventData(), &event );
#else
  const cl_int error = clEnqueueBarrier( this->GetActiveQueue() );
#endif

  this->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );

#ifdef CL_VERSION_1_2
  if( error != CL_SUCCESS )
  {
    return OpenCLEvent();
  }
  else
  {
    return OpenCLEvent( event );
  }
#else
  return OpenCLEvent();
#endif
}


//------------------------------------------------------------------------------
cl_int
OpenCLContext::WaitForFinished( const OpenCLEventList & event_list )
{
  if( event_list.IsEmpty() )
  {
    return CL_SUCCESS;
  }
  const cl_int error = clWaitForEvents( event_list.GetSize(), event_list.GetEventData() );
  if( error != CL_SUCCESS )
  {
    itkOpenCLErrorMacroGeneric( << "OpenCLContext::WaitForFinished:"
                                << OpenCLContext::GetErrorName( error ) );
  }
  return error;
}


//------------------------------------------------------------------------------
void
OpenCLContext::SetDefaultDevice( const OpenCLDevice & device )
{
  ITK_OPENCL_D( OpenCLContext );
  d->default_device = device;
}


//------------------------------------------------------------------------------
// \internal
void
OpenCLContext::OpenCLDebug( const std::string & callname )
{
#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  std::cout << callname << "..." << std::endl;
#endif
}


//------------------------------------------------------------------------------
void
OpenCLContext::ReportError( const cl_int code, const char * fileName,
  const int lineNumber, const char * location )
{
  ITK_OPENCL_D( OpenCLContext );
  d->last_error = code;
  if( code != CL_SUCCESS )
  {
    ExceptionObject e_( fileName, lineNumber,
    this->GetErrorName( code ).c_str(), location );
    throw e_;
  }
}


//------------------------------------------------------------------------------
// \internal
void
OpenCLContext::OpenCLProfile( cl_event clEvent,
  const std::string & message,
  const bool releaseEvent )
{
#ifdef OPENCL_PROFILING
  if( !clEvent )
  {
    return;
  }

  cl_int errid;

  // Execution time
  errid = clWaitForEvents( 1, &clEvent );
  this->ReportError( errid, __FILE__, __LINE__, ITK_LOCATION );
  if( errid != CL_SUCCESS )
  {
    itkOpenCLWarningMacro( "clWaitForEvents failed" );
    return;
  }
  cl_ulong start, end;
  errid  = clGetEventProfilingInfo( clEvent, CL_PROFILING_COMMAND_END, sizeof( cl_ulong ), &end, NULL );
  errid |= clGetEventProfilingInfo( clEvent, CL_PROFILING_COMMAND_START, sizeof( cl_ulong ), &start, NULL );
  this->ReportError( errid, __FILE__, __LINE__, ITK_LOCATION );
  if( errid != CL_SUCCESS )
  {
    itkOpenCLWarningMacro( "clGetEventProfilingInfo failed" );
    return;
  }

  const double dSeconds = 1.0e-9 * (double)( end - start );
  std::cout << "GPU " << message << " execution took " << dSeconds << " seconds." << std::endl;

  // Release event if required
  if( releaseEvent )
  {
    errid = clReleaseEvent( clEvent );
    this->ReportError( errid, __FILE__, __LINE__, ITK_LOCATION );
    if( errid != CL_SUCCESS )
    {
      itkOpenCLWarningMacro( "clReleaseEvent failed" );
      return;
    }
    clEvent = 0;
  }
#endif
}


//------------------------------------------------------------------------------
// \internal
void
OpenCLContext::SetUpProfiling()
{
#ifdef OPENCL_PROFILING
  OpenCLCommandQueue queue = this->CreateCommandQueue( CL_QUEUE_PROFILING_ENABLE );

  if( !queue.IsProfilingEnabled() )
  {
    itkOpenCLWarningMacro( << "OpenCLContext attempted to create OpenCL command queue "
                           << "with CL_QUEUE_PROFILING_ENABLE, but failed." );
  }
  else
  {
    this->SetCommandQueue( queue );
  }
#endif
}


} // namespace itk

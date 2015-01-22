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
#include "itkOpenCLProgram.h"
#include "itkOpenCLContext.h"
#include "itkOpenCLProfilingTimeProbe.h"
#include "itkOpenCLMacro.h"

// begin of OpenCLProgramSupport namespace
namespace OpenCLProgramSupport
{
bool
GetOpenCLMathAndOptimizationOptions( std::string & options )
{
  bool anyOptionSet = false;

  // OpenCL Math Intrinsics Options
#ifdef OPENCL_MATH_SINGLE_PRECISION_CONSTANT
  if( options.size() != 0 )
  {
    options.append( " " );
  }
  options.append( "-cl-single-precision-constant" );
  anyOptionSet = true;
#endif

#ifdef OPENCL_MATH_DENORMS_ARE_ZERO
  if( options.size() != 0 )
  {
    options.append( " " );
  }
  options.append( "-cl-denorms-are-zero" );
  anyOptionSet = true;
#endif

#ifdef OPENCL_MATH_FP32_CORRECTLY_ROUNDED_DIVIDE_SQRT
  if( options.size() != 0 )
  {
    options.append( " " );
  }
  options.append( "-cl-fp32-correctly-rounded-divide-sqrt" );
  anyOptionSet = true;
#endif

  // OpenCL Optimization Options
#ifdef OPENCL_OPTIMIZATION_OPT_DISABLE
  if( options.size() != 0 )
  {
    options.append( " " );
  }
  options.append( "-cl-opt-disable" );
  anyOptionSet = true;
#endif

#ifndef OPENCL_OPTIMIZATION_OPT_DISABLE
#ifdef OPENCL_OPTIMIZATION_MAD_ENABLE
  if( options.size() != 0 )
  {
    options.append( " " );
  }
  options.append( "-cl-mad-enable" );
  anyOptionSet = true;
#endif

#ifdef OPENCL_OPTIMIZATION_NO_SIGNED_ZEROS
  if( options.size() != 0 )
  {
    options.append( " " );
  }
  options.append( "-cl-no-signed-zeros" );
  anyOptionSet = true;
#endif

#ifdef OPENCL_OPTIMIZATION_FAST_RELAXED_MATH
  if( options.size() != 0 )
  {
    options.append( " " );
  }
  options.append( "-cl-fast-relaxed-math" );
  anyOptionSet = true;
#endif

#ifdef OPENCL_OPTIMIZATION_UNIFORM_WORK_GROUP_SIZE
  if( options.size() != 0 )
  {
    options.append( " " );
  }
  options.append( "-cl-uniform-work-group-size" );
  anyOptionSet = true;
#endif

#ifndef OPENCL_OPTIMIZATION_FAST_RELAXED_MATH
#ifdef OPENCL_OPTIMIZATION_UNSAFE_MATH_OPTIMIZATIONS
  if( options.size() != 0 )
  {
    options.append( " " );
  }
  options.append( "-cl-unsafe-math-optimizations" );
  anyOptionSet = true;
#endif

#ifdef OPENCL_OPTIMIZATION_FINITE_MATH_ONLY
  if( options.size() != 0 )
  {
    options.append( " " );
  }
  options.append( "-cl-finite-math-only" );
  anyOptionSet = true;
#endif
#endif // #ifndef OPENCL_OPTIMIZATION_FAST_RELAXED_MATH
#endif // #ifndef OPENCL_OPTIMIZATION_CL_OPT_DISABLE

  // OpenCL Options to Request or Suppress Warnings
#ifdef OPENCL_WARNINGS_DISABLE
  if( options.size() != 0 )
  {
    options.append( " " );
  }
  options.append( "-w" );
  anyOptionSet = true;
#elif OPENCL_WARNINGS_AS_ERRORS
  if( options.size() != 0 )
  {
    options.append( " " );
  }
  options.append( "-Werror" );
  anyOptionSet = true;
#endif

  // Options Controlling the OpenCL C Version
#ifdef OPENCL_C_VERSION_1_1
  if( options.size() != 0 )
  {
    options.append( " " );
  }
  // With the -cl-std=CL1.1 option will fail to compile the program for any
  // devices with CL_DEVICE_OPENCL_C_VERSION = OpenCL C 1.0.
  options.append( "-cl-std=CL1.1" );
  anyOptionSet = true;
#elif OPENCL_C_VERSION_1_2
  if( options.size() != 0 )
  {
    options.append( " " );
  }
  // With the -cl-std=CL1.2 option will fail to compile the program for any
  // devices with CL_DEVICE_OPENCL_C_VERSION = OpenCL C 1.0 or OpenCL C 1.1.
  options.append( "-cl-std=CL1.2" );
  anyOptionSet = true;
#elif OPENCL_C_VERSION_2_0
  if( options.size() != 0 )
  {
    options.append( " " );
  }
  // With the -cl-std=CL2.0 option will fail to compile the program for any
  // devices with CL_DEVICE_OPENCL_C_VERSION = OpenCL C 1.0, OpenCL C 1.1 or OpenCL C 1.2.
  options.append( "-cl-std=CL2.0" );
  anyOptionSet = true;
#endif

  return anyOptionSet;
}


} // end of OpenCLProgramSupport namespace

namespace itk
{
OpenCLProgram::OpenCLProgram() :
  m_Context( 0 ), m_Id( 0 )
{}

//------------------------------------------------------------------------------
OpenCLProgram::OpenCLProgram( OpenCLContext * context,
  cl_program id,
  const std::string & fileName ) :
  m_Context( context ), m_Id( id ), m_FileName( fileName )
{}

//------------------------------------------------------------------------------
OpenCLProgram::OpenCLProgram( const OpenCLProgram & other ) :
  m_Context( other.m_Context ), m_Id( other.m_Id ), m_FileName( other.m_FileName )
{
  if( this->m_Id )
  {
    clRetainProgram( this->m_Id );
  }
}


//------------------------------------------------------------------------------
OpenCLProgram::~OpenCLProgram()
{
  if( this->m_Id )
  {
    clReleaseProgram( this->m_Id );
  }
}


//------------------------------------------------------------------------------
OpenCLProgram &
OpenCLProgram::operator=( const OpenCLProgram & other )
{
  this->m_Context = other.m_Context;
  if( other.m_Id )
  {
    clRetainProgram( other.m_Id );
  }
  if( this->m_Id )
  {
    clReleaseProgram( this->m_Id );
  }
  this->m_Id = other.m_Id;
  return *this;
}


//------------------------------------------------------------------------------
bool
OpenCLProgram::Build( const std::string & extraBuildOptions )
{
  return this->Build( std::list< OpenCLDevice >(), extraBuildOptions );
}


//------------------------------------------------------------------------------
bool
OpenCLProgram::Build( const std::list< OpenCLDevice > & devices,
  const std::string & extraBuildOptions )
{
  std::vector< cl_device_id > devs;

  for( std::list< OpenCLDevice >::const_iterator dev = devices.begin();
    dev != devices.end(); ++dev )
  {
    if( ( *dev ).GetDeviceId() )
    {
      devs.push_back( ( *dev ).GetDeviceId() );
    }
  }

  // Get OpenCL math and optimization options
  std::string oclOptions;
  OpenCLProgramSupport::GetOpenCLMathAndOptimizationOptions( oclOptions );

  // Append extra OpenCL options if provided
  oclOptions = !extraBuildOptions.empty() ? oclOptions + " " + extraBuildOptions : oclOptions;

#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  if( GetFileName().size() > 0 )
  {
    const std::string message = "clBuildProgram from file '" + GetFileName() + "'";
    this->GetContext()->OpenCLDebug( message );
  }
  else
  {
    this->GetContext()->OpenCLDebug( "clBuildProgram from source" );
  }
#endif

#ifdef OPENCL_PROFILING
  itk::OpenCLProfilingTimeProbe timer( "Building OpenCL program using clBuildProgram" );
#endif

  cl_int error;

#if defined( OPENCL_USE_INTEL_CPU ) && defined( _DEBUG )
  // Enable debugging mode in the Intel OpenCL runtime
  if( this->GetFileName().size() > 0 )
  {
    const std::string oclDebugOptions = "-g -s \"" + this->GetFileName() + "\"";
    oclOptions = oclOptions.empty() ? oclDebugOptions :
      oclDebugOptions + " " + oclOptions;
  }

  if( devs.size() == 0 )
  {
    error = clBuildProgram( this->m_Id, 0, 0,
      oclOptions.empty() ? 0 : &oclOptions[ 0 ], 0, 0 );
  }
  else
  {
    error = clBuildProgram( this->m_Id, devs.size(), &devs[ 0 ],
      oclOptions.empty() ? 0 : &oclOptions[ 0 ], 0, 0 );
  }
#else
  if( devs.size() == 0 )
  {
    error = clBuildProgram( this->m_Id, 0, 0,
      oclOptions.empty() ? 0 : &oclOptions[ 0 ], 0, 0 );
  }
  else
  {
    error = clBuildProgram( this->m_Id, devs.size(), &devs[ 0 ],
      oclOptions.empty() ? 0 : &oclOptions[ 0 ], 0, 0 );
  }
#endif

  this->GetContext()->SetLastError( error );

  if( error == CL_SUCCESS )
  {
    return true;
  }

  // Report error about build
  itkOpenCLErrorMacroGeneric( << "OpenCLProgram::build:" << OpenCLContext::GetErrorName( error ) );

  // Throw OpenCLCompileError exception
  OpenCLCompileError e( __FILE__, __LINE__ );
  e.SetLocation( ITK_LOCATION );
  e.SetDescription( GetLog() );
  throw e;

  return false;
}


//------------------------------------------------------------------------------
std::string
OpenCLProgram::GetLog() const
{
  // Get the list of devices for the program's GetContext.
  // Note: CL_PROGRAM_DEVICES is unreliable on some OpenCL implementations.
  OpenCLContext * ctx = this->GetContext();

  if( !ctx )
  {
    return std::string();
  }

  const std::list< OpenCLDevice > devs = ctx->GetDevices();

  // Retrieve the device GetLogs and concatenate them.
  std::string log;
  for( std::list< itk::OpenCLDevice >::const_iterator dev = devs.begin();
    dev != devs.end(); ++dev )
  {
    std::size_t size = 0;
    if( clGetProgramBuildInfo
        ( this->m_Id, ( *dev ).GetDeviceId(), CL_PROGRAM_BUILD_LOG, 0, 0, &size ) != CL_SUCCESS || !size )
    {
      continue;
    }
    std::string buffer( size, '\0' );
    if( clGetProgramBuildInfo
        ( this->m_Id, ( *dev ).GetDeviceId(), CL_PROGRAM_BUILD_LOG, size, &buffer[ 0 ], 0 ) != CL_SUCCESS || !size )
    {
      continue;
    }
    log += buffer;
  }
  return log;
}


//------------------------------------------------------------------------------
std::list< OpenCLDevice > OpenCLProgram::GetDevices() const
{
  std::list< OpenCLDevice > list;
  cl_uint                   size;

  if( clGetProgramInfo( this->m_Id, CL_PROGRAM_NUM_DEVICES,
    sizeof( size ), &size, 0 ) != CL_SUCCESS || size == 0 )
  {
    return list;
  }
  std::vector< cl_device_id > buffer( size );
  if( clGetProgramInfo( this->m_Id, CL_PROGRAM_DEVICES, size * sizeof( cl_device_id ), &buffer[ 0 ], 0 ) != CL_SUCCESS )
  {
    return list;
  }

  for( std::vector< cl_device_id >::const_iterator dev = buffer.begin();
    dev != buffer.end(); ++dev )
  {
    list.push_back( OpenCLDevice( *dev ) );
  }
  return list;
}


//------------------------------------------------------------------------------
OpenCLKernel
OpenCLProgram::CreateKernel( const std::string & name ) const
{
  cl_int    error;
  cl_kernel kernel = clCreateKernel( this->m_Id, name.c_str(), &error );

  if( kernel )
  {
    this->GetContext()->SetLastError( error );
    return OpenCLKernel( this->m_Context, kernel );
  }
  this->GetContext()->SetLastError( error );
  itkOpenCLWarningMacroGeneric( << "OpenCLProgram::CreateKernel(" << name << "):"
                                << OpenCLContext::GetErrorName( error ) );
  return OpenCLKernel();
}


//------------------------------------------------------------------------------
std::list< OpenCLKernel > OpenCLProgram::CreateKernels() const
{
  std::list< OpenCLKernel > list;
  cl_uint                   numKernels = 0;

  if( clCreateKernelsInProgram( this->m_Id, 0, 0, &numKernels ) != CL_SUCCESS )
  {
    return list;
  }
  std::vector< cl_kernel > kernels( numKernels );
  if( clCreateKernelsInProgram
      ( this->m_Id, numKernels, &kernels[ 0 ], 0 ) != CL_SUCCESS )
  {
    return list;
  }
  for( std::vector< cl_kernel >::const_iterator kernel = kernels.begin();
    kernel != kernels.end(); ++kernel )
  {
    list.push_back( OpenCLKernel( this->m_Context, *kernel ) );
  }
  return list;
}


//------------------------------------------------------------------------------
//! Operator ==
bool
operator==( const OpenCLProgram & lhs, const OpenCLProgram & rhs )
{
  if( &rhs == &lhs )
  {
    return true;
  }
  return lhs.GetProgramId() == rhs.GetProgramId();
}


//------------------------------------------------------------------------------
//! Operator !=
bool
operator!=( const OpenCLProgram & lhs, const OpenCLProgram & rhs )
{
  return !( lhs == rhs );
}


} // end namespace itk

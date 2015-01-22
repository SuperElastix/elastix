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
*  Copyright Insight Software Consortium
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*         http://www.apache.org/licenses/LICENSE-2.0.txt
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
*=========================================================================*/
#include "itkOpenCLKernelManager.h"
#include "itkOpenCLContext.h"
#include "itkOpenCLMacro.h"

namespace itk
{
OpenCLKernelManager::OpenCLKernelManager()
{
  this->m_Context = OpenCLContext::GetInstance();
}


//------------------------------------------------------------------------------
OpenCLKernelManager::~OpenCLKernelManager()
{}

//------------------------------------------------------------------------------
OpenCLKernel &
OpenCLKernelManager::GetKernel( const std::size_t kernelId )
{
  return m_Kernels[ kernelId ];
}


//------------------------------------------------------------------------------
std::size_t
OpenCLKernelManager::CreateKernel( const OpenCLProgram & program,
  const std::string & name )
{
  std::size_t createResult = 0;

  // Check the program
  if( program.IsNull() )
  {
    itkOpenCLWarningMacro(
        << "OpenCL kernel '" << name << "' has not been created." << "Provided program is null. Returned "
        << createResult );
    return createResult;
  }

  // Create the kernel
  OpenCLKernel kernel = program.CreateKernel( name );
  if( kernel.IsNull() )
  {
    itkOpenCLWarningMacro( << "Fail to create OpenCL kernel '" << name
                           << "'. Returned " << createResult );
    return createResult;
  }

  // Add kernel to container
  this->m_Kernels.push_back( kernel );

  // Add arguments list
  std::vector< KernelArgumentList > kernelArguments( kernel.GetNumberOfArguments() );
  this->m_KernelArgumentReady.push_back( kernelArguments );

  // Reset arguments and return
  this->ResetArguments( this->m_Kernels.size() - 1 );

  return this->m_Kernels.size() - 1;
}


//------------------------------------------------------------------------------
OpenCLProgram
OpenCLKernelManager::BuildProgramFromSourceCode( const std::string & sourceCode,
  const std::string & prefixSourceCode,
  const std::string & postfixSourceCode,
  const std::string & extraBuildOptions )
{
  const std::list< OpenCLDevice > devices = this->m_Context->GetDevices();
  OpenCLProgram                   program = this->m_Context->BuildProgramFromSourceCode( devices,
    sourceCode,
    prefixSourceCode,
    postfixSourceCode,
    extraBuildOptions );

  return program;
}


//------------------------------------------------------------------------------
OpenCLProgram
OpenCLKernelManager::BuildProgramFromSourceFile( const std::string & fileName,
  const std::string & prefixSourceCode,
  const std::string & postfixSourceCode,
  const std::string & extraBuildOptions )
{
  const std::list< OpenCLDevice > devices = this->m_Context->GetDevices();
  OpenCLProgram                   program = this->m_Context->BuildProgramFromSourceFile( devices,
    fileName,
    prefixSourceCode,
    postfixSourceCode,
    extraBuildOptions );

  return program;
}


//------------------------------------------------------------------------------
bool
OpenCLKernelManager::SetKernelArg( const std::size_t kernelId,
  const cl_uint argId, const std::size_t argSize,
  const void * argVal )
{
  if( kernelId >= this->m_Kernels.size() ) { return false; }

  cl_int error;
#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  std::cout << "clSetKernelArg" << "..." << std::endl;
#endif
  error = clSetKernelArg( this->GetKernel( kernelId ).GetKernelId(), argId, argSize, argVal );

  if( error != CL_SUCCESS )
  {
    itkWarningMacro( "Setting kernel argument failed with GPUKernelManager::SetKernelArg("
        << kernelId << ", " << argId << ", " << argSize << ". " << argVal << ")" );
  }

  this->m_Context->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );

  this->m_KernelArgumentReady[ kernelId ][ argId ].m_IsReady        = true;
  this->m_KernelArgumentReady[ kernelId ][ argId ].m_GPUDataManager = (GPUDataManager::Pointer)NULL;

  return true;
}


//------------------------------------------------------------------------------
bool
OpenCLKernelManager::SetKernelArgForAllKernels(
  const cl_uint argId,
  const std::size_t argSize, const void * argVal )
{
  if( this->m_Kernels.empty() )
  {
    return false;
  }

  // Set the argument to all kernels. Return true when all are true.
  bool returnValue = true;
  for( std::size_t kernelId = 0; kernelId < this->m_Kernels.size(); ++kernelId )
  {
    returnValue &= this->SetKernelArg( kernelId, argId, argSize, argVal );
  }

  return returnValue;
}


//------------------------------------------------------------------------------
bool
OpenCLKernelManager::SetKernelArgWithImage(
  const std::size_t kernelId, const cl_uint argId,
  const GPUDataManager::Pointer manager )
{
  if( kernelId >= this->m_Kernels.size() ) { return false; }

  cl_int error;
#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  std::cout << "clSetKernelArg" << "..." << std::endl;
#endif

  if( manager->GetBufferSize() > 0 )
  {
    error = clSetKernelArg( this->GetKernel( kernelId ).GetKernelId(),
      argId, sizeof( cl_mem ), manager->GetGPUBufferPointer() );
  }
  else
  {
    // Check and remove it for Intel SDK for OpenCL 2013
#if defined( OPENCL_USE_INTEL_CPU )
    // http://software.intel.com/en-us/forums/topic/281206
    itkWarningMacro( "Intel SDK for OpenCL 2012 does not support setting NULL buffers." );
    return false;
#endif
    // According OpenCL 1.1 specification clSetKernelArg arg_value could be NULL
    // object.
    cl_mem null_buffer = NULL;
    error = clSetKernelArg( this->GetKernel( kernelId ).GetKernelId(), argId, sizeof( cl_mem ), &null_buffer );
  }

  if( error != CL_SUCCESS )
  {
    itkWarningMacro( "Setting kernel argument failed with GPUKernelManager::SetKernelArgWithImage("
        << kernelId << ", " << argId << ", " << manager << ")" );
  }

  this->m_Context->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );

  this->m_KernelArgumentReady[ kernelId ][ argId ].m_IsReady        = true;
  this->m_KernelArgumentReady[ kernelId ][ argId ].m_GPUDataManager = manager;

  return true;
}


//------------------------------------------------------------------------------
// this function must be called right before GPU kernel is launched
bool
OpenCLKernelManager::CheckArgumentReady( const std::size_t kernelId )
{
  const std::size_t nArg = this->m_KernelArgumentReady[ kernelId ].size();

  for( std::size_t i = 0; i < nArg; i++ )
  {
    if( !( this->m_KernelArgumentReady[ kernelId ][ i ].m_IsReady ) ) { return false; }

    // automatic synchronization before kernel launch
    if( this->m_KernelArgumentReady[ kernelId ][ i ].m_GPUDataManager != (GPUDataManager::Pointer)NULL )
    {
      this->m_KernelArgumentReady[ kernelId ][ i ].m_GPUDataManager->SetCPUBufferDirty();
    }
  }
  return true;
}


//------------------------------------------------------------------------------
void
OpenCLKernelManager::ResetArguments( const std::size_t kernelIdx )
{
  const std::size_t nArg = this->m_KernelArgumentReady[ kernelIdx ].size();

  for( std::size_t i = 0; i < nArg; i++ )
  {
    this->m_KernelArgumentReady[ kernelIdx ][ i ].m_IsReady        = false;
    this->m_KernelArgumentReady[ kernelIdx ][ i ].m_GPUDataManager = (GPUDataManager::Pointer)NULL;
  }
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLKernelManager::LaunchKernel( const std::size_t kernelId )
{
  if( kernelId >= this->m_Kernels.size() )
  {
    return OpenCLEvent();
  }

  //if ( !CheckArgumentReady( kernelId ) )
  //{
  //  itkOpenCLErrorMacro( "GPU kernel arguments are not completely assigned for
  // the kernel: " << kernelId );
  //  return OpenCLEvent();
  //}

  OpenCLKernel & kernel = this->GetKernel( kernelId );
  if( kernel.IsNull() )
  {
    return OpenCLEvent();
  }

  return kernel.LaunchKernel();
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLKernelManager::LaunchKernel( const std::size_t kernelId,
  const OpenCLSize & global_work_size,
  const OpenCLSize & local_work_size,
  const OpenCLSize & global_work_offset )
{
  if( kernelId >= this->m_Kernels.size() )
  {
    return OpenCLEvent();
  }

  //if ( !CheckArgumentReady( kernelId ) )
  //{
  //  itkOpenCLErrorMacro( "GPU kernel arguments are not completely assigned for
  // the kernel: " << kernelId );
  //  return OpenCLEvent();
  //}

  OpenCLKernel & kernel = this->GetKernel( kernelId );
  if( kernel.IsNull() )
  {
    return OpenCLEvent();
  }

  kernel.SetGlobalWorkSize( global_work_size );
  kernel.SetLocalWorkSize( local_work_size );
  kernel.SetGlobalWorkOffset( global_work_offset );

  return kernel.LaunchKernel();
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLKernelManager::LaunchKernel( const std::size_t kernelId,
  const OpenCLEventList & event_list )
{
  if( kernelId >= this->m_Kernels.size() )
  {
    return OpenCLEvent();
  }

  //if ( !CheckArgumentReady( kernelId ) )
  //{
  //  itkOpenCLErrorMacro( "GPU kernel arguments are not completely assigned for
  // the kernel: " << kernelId );
  //  return OpenCLEvent();
  //}

  OpenCLKernel & kernel = this->GetKernel( kernelId );
  if( kernel.IsNull() )
  {
    return OpenCLEvent();
  }

  return kernel.LaunchKernel( event_list );
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLKernelManager::LaunchKernel( const std::size_t kernelId,
  const OpenCLEventList & event_list,
  const OpenCLSize & global_work_size,
  const OpenCLSize & local_work_size,
  const OpenCLSize & global_work_offset )
{
  if( kernelId >= this->m_Kernels.size() )
  {
    return OpenCLEvent();
  }

  //if ( !CheckArgumentReady( kernelId ) )
  //{
  //  itkOpenCLErrorMacro( "GPU kernel arguments are not completely assigned for
  // the kernel: " << kernelId );
  //  return OpenCLEvent();
  //}

  OpenCLKernel & kernel = this->GetKernel( kernelId );
  if( kernel.IsNull() )
  {
    return OpenCLEvent();
  }

  kernel.SetGlobalWorkSize( global_work_size );
  kernel.SetLocalWorkSize( local_work_size );
  kernel.SetGlobalWorkOffset( global_work_offset );

  return kernel.LaunchKernel( event_list );
}


//------------------------------------------------------------------------------
void
OpenCLKernelManager::SetGlobalWorkSizeForAllKernels( const OpenCLSize & size )
{
  if( this->m_Kernels.empty() )
  {
    return;
  }

  for( std::vector< OpenCLKernel >::iterator kernel = this->m_Kernels.begin();
    kernel != this->m_Kernels.end(); ++kernel )
  {
    ( *kernel ).SetGlobalWorkSize( size );
  }
}


//------------------------------------------------------------------------------
void
OpenCLKernelManager::SetLocalWorkSizeForAllKernels( const OpenCLSize & size )
{
  if( this->m_Kernels.empty() )
  {
    return;
  }

  for( std::vector< OpenCLKernel >::iterator kernel = this->m_Kernels.begin();
    kernel != this->m_Kernels.end(); ++kernel )
  {
    ( *kernel ).SetLocalWorkSize( size );
  }
}


//------------------------------------------------------------------------------
void
OpenCLKernelManager::SetGlobalWorkOffsetForAllKernels( const OpenCLSize & offset )
{
  if( this->m_Kernels.empty() )
  {
    return;
  }

  for( std::vector< OpenCLKernel >::iterator kernel = this->m_Kernels.begin();
    kernel != this->m_Kernels.end(); ++kernel )
  {
    ( *kernel ).SetGlobalWorkOffset( offset );
  }
}


} // end namespace itk

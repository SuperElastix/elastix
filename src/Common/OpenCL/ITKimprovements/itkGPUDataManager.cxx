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
#include "itkGPUDataManager.h"

namespace itk
{
// constructor
GPUDataManager::GPUDataManager()
{
  m_Context   = OpenCLContext::GetInstance();
  m_GPUBuffer = NULL;
  m_CPUBuffer = NULL;

  m_CPUBufferLock = false;
  m_GPUBufferLock = false;

  this->Initialize();
}


//------------------------------------------------------------------------------
GPUDataManager::~GPUDataManager()
{
  if( m_GPUBuffer )
  {
#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
    std::cout << "clReleaseMemObject" << "..." << std::endl;
#endif
    cl_int errid = clReleaseMemObject( m_GPUBuffer );
    m_Context->ReportError( errid, __FILE__, __LINE__, ITK_LOCATION );
  }
}


//------------------------------------------------------------------------------
void
GPUDataManager::SetBufferSize( unsigned int num )
{
  m_BufferSize = num;
}


//------------------------------------------------------------------------------
void
GPUDataManager::SetBufferFlag( cl_mem_flags flags )
{
  m_MemFlags = flags;
}


//------------------------------------------------------------------------------
void
GPUDataManager::Allocate()
{
  if( this->m_GPUBufferLock )
  {
    return;
  }

  cl_int errid;

  if( m_BufferSize > 0 )
  {
#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
    std::cout << "clCreateBuffer, "
              << this <<  "::Allocate Create GPU buffer of size "
              << m_BufferSize << " Bytes" << std::endl;
#endif
    m_GPUBuffer = clCreateBuffer( m_Context->GetContextId(),
      m_MemFlags, m_BufferSize, NULL, &errid );
    m_Context->ReportError( errid, __FILE__, __LINE__, ITK_LOCATION );
    m_IsGPUBufferDirty = true;
  }

  //this->UpdateGPUBuffer();
}


//------------------------------------------------------------------------------
void
GPUDataManager::SetCPUBufferPointer( void * ptr )
{
  m_CPUBuffer = ptr;
}


//------------------------------------------------------------------------------
void
GPUDataManager::SetCPUDirtyFlag( bool isDirty )
{
  m_IsCPUBufferDirty = isDirty;
}


//------------------------------------------------------------------------------
void
GPUDataManager::SetGPUDirtyFlag( bool isDirty )
{
  m_IsGPUBufferDirty = isDirty;
}


//------------------------------------------------------------------------------
void
GPUDataManager::SetGPUBufferDirty()
{
  this->UpdateCPUBuffer();
  m_IsGPUBufferDirty = true;
}


//------------------------------------------------------------------------------
void
GPUDataManager::SetCPUBufferDirty()
{
  this->UpdateGPUBuffer();
  m_IsCPUBufferDirty = true;
}


//------------------------------------------------------------------------------
void
GPUDataManager::UpdateCPUBuffer()
{
  if( this->m_CPUBufferLock )
  {
    return;
  }

  MutexHolderType holder( m_Mutex );

  if( m_IsCPUBufferDirty && m_GPUBuffer != NULL && m_CPUBuffer != NULL )
  {
#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
    std::cout << "clEnqueueReadBuffer, " << this
              << "::UpdateCPUBuffer GPU->CPU data copy "
              << m_GPUBuffer << "->" << m_CPUBuffer << std::endl;
#endif

    cl_int errid;
#ifdef OPENCL_PROFILING
    cl_event clEvent = NULL;
    errid = clEnqueueReadBuffer( m_Context->GetCommandQueue().GetQueueId(), m_GPUBuffer, CL_TRUE, 0,
      m_BufferSize, m_CPUBuffer, 0, NULL,
      &clEvent );
#else
    errid = clEnqueueReadBuffer( m_Context->GetCommandQueue().GetQueueId(), m_GPUBuffer, CL_TRUE, 0,
      m_BufferSize, m_CPUBuffer, 0, NULL, NULL );
#endif

    m_Context->ReportError( errid, __FILE__, __LINE__, ITK_LOCATION );
    //m_ContextManager->OpenCLProfile(clEvent, "clEnqueueReadBuffer GPU->CPU");

    m_IsCPUBufferDirty = false;
  }
}


//------------------------------------------------------------------------------
void
GPUDataManager::UpdateGPUBuffer()
{
  if( this->m_GPUBufferLock )
  {
    return;
  }

  MutexHolderType holder( m_Mutex );

  if( m_IsGPUBufferDirty && m_CPUBuffer != NULL && m_GPUBuffer != NULL )
  {
#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
    std::cout << "clEnqueueWriteBuffer, " << this << "::UpdateGPUBuffer CPU->GPU data copy "
              << m_CPUBuffer << "->" << m_GPUBuffer << std::endl;
#endif

    cl_int errid;
#ifdef OPENCL_PROFILING
    cl_event clEvent = NULL;
    errid = clEnqueueWriteBuffer(
      m_Context->GetCommandQueue().GetQueueId(), m_GPUBuffer, CL_TRUE, 0, m_BufferSize, m_CPUBuffer, 0, NULL,
      &clEvent );
#else
    errid = clEnqueueWriteBuffer(
      m_Context->GetCommandQueue().GetQueueId(), m_GPUBuffer, CL_TRUE, 0, m_BufferSize, m_CPUBuffer, 0, NULL,
      NULL );
#endif
    m_Context->ReportError( errid, __FILE__, __LINE__, ITK_LOCATION );
    //m_ContextManager->OpenCLProfile(clEvent, "clEnqueueWriteBuffer CPU->GPU");

    m_IsGPUBufferDirty = false;
  }
}


//------------------------------------------------------------------------------
cl_mem *
GPUDataManager::GetGPUBufferPointer()
{
  SetCPUBufferDirty();
  return &m_GPUBuffer;
}


//------------------------------------------------------------------------------
void *
GPUDataManager::GetCPUBufferPointer()
{
  SetGPUBufferDirty();
  return m_CPUBuffer;
}


//------------------------------------------------------------------------------
bool
GPUDataManager::Update()
{
  if( m_IsGPUBufferDirty && m_IsCPUBufferDirty )
  {
    itkExceptionMacro( "Cannot make up-to-date buffer because both CPU and GPU buffers are dirty" );
    return false;
  }

  this->UpdateGPUBuffer();
  this->UpdateCPUBuffer();

  m_IsGPUBufferDirty = m_IsCPUBufferDirty = false;

  return true;
}


//------------------------------------------------------------------------------
/**
* NOTE: each device has a command queue. Therefore, changing command queue
*       means change a compute device.
*/
//void GPUDataManager::SetCurrentCommandQueue(int queueid)
//{
//  if ( queueid >= 0 && queueid <
// (int)m_ContextManager->GetNumberOfCommandQueues() )
//    {
//    this->UpdateCPUBuffer();
//
//    // Assumption: different command queue is assigned to different device
//    m_CommandQueueId = queueid;
//
//    m_IsGPUBufferDirty = true;
//    }
//  else
//    {
//    itkWarningMacro("Not a valid command queue id");
//    }
//}

//------------------------------------------------------------------------------
//int GPUDataManager::GetCurrentCommandQueueId()
//{
//  return m_CommandQueueId;
//}

//------------------------------------------------------------------------------
void
GPUDataManager::Graft( const GPUDataManager * data )
{
  if( data )
  {
    m_BufferSize = data->m_BufferSize;
    m_Context    = data->m_Context;
    m_MemFlags   = data->m_MemFlags;

    if( m_GPUBuffer )  // Decrease reference count to GPU memory
    {
#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
      std::cout << "clReleaseMemObject" << "..." << std::endl;
#endif
      clReleaseMemObject( m_GPUBuffer );
    }
    if( data->m_GPUBuffer )  // Increase reference count to GPU memory
    {
#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
      std::cout << "clRetainMemObject" << "..." << std::endl;
#endif
      clRetainMemObject( data->m_GPUBuffer );
    }

    m_GPUBuffer = data->m_GPUBuffer;
    m_CPUBuffer = data->m_CPUBuffer;

    // m_Platform  = data->m_Platform;
    // m_Context   = data->m_Context;

    m_IsCPUBufferDirty = data->m_IsCPUBufferDirty;
    m_IsGPUBufferDirty = data->m_IsGPUBufferDirty;
  }
}


//------------------------------------------------------------------------------
void
GPUDataManager::Initialize()
{
  //if ( m_ContextManager->GetNumberOfCommandQueues() > 0 )
  //  {
  //  m_CommandQueueId = 0; // default command queue
  //  }

  if( m_GPUBuffer )  // Release GPU memory if exists
  {
#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
    std::cout << "clReleaseMemObject" << "..." << std::endl;
#endif
    cl_int errid = clReleaseMemObject( m_GPUBuffer );
    m_Context->ReportError( errid, __FILE__, __LINE__, ITK_LOCATION );
  }

  m_BufferSize       = 0;
  m_GPUBuffer        = NULL;
  m_CPUBuffer        = NULL;
  m_MemFlags         = CL_MEM_READ_WRITE; // default flag
  m_IsGPUBufferDirty = false;
  m_IsCPUBufferDirty = false;

  m_CPUBufferLock = false;
  m_GPUBufferLock = false;
}


//------------------------------------------------------------------------------
void
GPUDataManager::PrintSelf( std::ostream & os, Indent indent ) const
{
  os << indent << "GPUDataManager (" << this << ")" << std::endl;
  os << indent << "m_BufferSize: " << m_BufferSize << std::endl;
  os << indent << "m_IsGPUBufferDirty: " << m_IsGPUBufferDirty << std::endl;
  os << indent << "m_GPUBuffer: " << m_GPUBuffer << std::endl;
  os << indent << "m_IsCPUBufferDirty: " << m_IsCPUBufferDirty << std::endl;
  os << indent << "m_CPUBuffer: " << m_CPUBuffer << std::endl;
  os << indent << "m_CPUBufferLock: " << m_CPUBufferLock << std::endl;
  os << indent << "m_GPUBufferLock: " << m_GPUBufferLock << std::endl;
}


} // namespace itk

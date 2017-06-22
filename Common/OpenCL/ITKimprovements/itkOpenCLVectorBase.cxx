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
#include "itkOpenCLVectorBase.h"
#include "itkOpenCLContext.h"

// thread/SMP safe reference counter
#include <vcl_atomic_count.h>

namespace itk
{
#if defined( __APPLE__ ) || defined( __MACOSX )
#define ITK_CL_COPY_VECTOR 1
#endif

enum OpenCLVectorState {
  Uninitialized, // Buffer contains uninitialized contents.
  InHost,        // Data is currently in the host.
  InKernel       // Data is currently in a kernel.
};

class OpenCLVectorBasePimpl
{
public:

  OpenCLVectorBasePimpl() :
    referenceCount( 1 ), // reference count to 1
    state( Uninitialized ),
    context( 0 ),
    id( 0 ),
    hostCopy( 0 )
  {}

  void * GetHostPointer( const std::size_t size )
  {
    if( !hostCopy )
    {
      hostCopy = ::malloc( size );
    }
    return hostCopy;
  }


public:

  vcl_atomic_count                referenceCount;
  OpenCLVectorState               state;
  OpenCLContext *                 context;
  cl_mem                          id;
  void *                          hostCopy;
  std::list< OpenCLVectorBase * > owners;
};

//------------------------------------------------------------------------------
OpenCLVectorBase::OpenCLVectorBase( const std::size_t elemSize ) :
  d_ptr( 0 ),
  m_ElementSize( elemSize ),
  m_Size( 0 ),
  m_Mapped( 0 )
{}

//------------------------------------------------------------------------------
OpenCLVectorBase::OpenCLVectorBase( const std::size_t elemSize,
  const OpenCLVectorBase & other ) :
  d_ptr( other.d_ptr ),
  m_ElementSize( elemSize ),
  m_Size( other.m_Size ),
  m_Mapped( other.m_Mapped )
{
  if( this->d_ptr )
  {
    ++this->d_ptr->referenceCount;
    this->d_ptr->owners.push_back( this );
  }
}


//------------------------------------------------------------------------------
OpenCLVectorBase::~OpenCLVectorBase()
{
  this->Release();
}


//------------------------------------------------------------------------------
void
OpenCLVectorBase::Assign( const OpenCLVectorBase & other )
{
  if( this->d_ptr == other.d_ptr )
  {
    return;
  }
  this->Release();
  this->d_ptr    = other.d_ptr;
  this->m_Size   = other.m_Size;
  this->m_Mapped = other.m_Mapped;
  if( this->d_ptr )
  {
    ++this->d_ptr->referenceCount;
    this->d_ptr->owners.push_back( this );
  }
}


//------------------------------------------------------------------------------
void
OpenCLVectorBase::Create( OpenCLContext * context,
  const OpenCLMemoryObject::Access access, const std::size_t size  )
{
  this->Release();

  // Create base pimpl class
  this->d_ptr = new OpenCLVectorBasePimpl();
  itkAssertOrThrowMacro( ( this->d_ptr != 0 ), "OpenCLVectorBase::Create()"
      << " unable to create base pimpl class." );

  this->d_ptr->owners.push_back( this );
  cl_int error;
  cl_mem id = clCreateBuffer
      ( context->GetContextId(),
#ifndef ITK_CL_COPY_VECTOR
      cl_mem_flags( access ) | CL_MEM_ALLOC_HOST_PTR,
#else
      cl_mem_flags( access ),
#endif
      size * this->m_ElementSize, 0, &error );
  context->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  if( id )
  {
    this->d_ptr->context  = context;
    this->d_ptr->id       = id;
    this->d_ptr->state    = Uninitialized;
    this->d_ptr->hostCopy = 0;
    this->m_Size          = size;
  }
}


//------------------------------------------------------------------------------
void
OpenCLVectorBase::Release()
{
  if( !this->d_ptr )
  {
    return;
  }

  if( --this->d_ptr->referenceCount <= 0 )
  {
    this->d_ptr    = 0;
    this->m_Size   = 0;
    this->m_Mapped = 0;
    return;
  }
#ifndef ITK_CL_COPY_VECTOR
  this->Unmap();
#else
  // No need to write back if we will discard the contents anyway.
  this->m_Mapped = 0;
#endif
  if( this->d_ptr->id )
  {
    clReleaseMemObject( this->d_ptr->id );
    this->d_ptr->id = 0;
  }
  this->d_ptr->context = 0;
  this->d_ptr->state   = Uninitialized;
  this->m_Size         = 0;
  if( this->d_ptr->hostCopy )
  {
    ::free( this->d_ptr->hostCopy );
    this->d_ptr->hostCopy = 0;
  }
  delete this->d_ptr;
  this->d_ptr = 0;
}


//------------------------------------------------------------------------------
void
OpenCLVectorBase::Map()
{
  // Bail out if no buffer, or already mapped.
  if( !this->d_ptr || !this->d_ptr->id || this->m_Mapped )
  {
    return;
  }

#ifndef ITK_CL_COPY_VECTOR
  cl_int error;
  this->m_Mapped = clEnqueueMapBuffer
      ( this->d_ptr->context->GetActiveQueue(), this->d_ptr->id,
      CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
      0, this->m_Size * this->m_ElementSize, 0, 0, 0, &error );
  this->d_ptr->context->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
#else
  // We cannot map the buffer directly, so do an explicit read-back.
  // We skip the read-back if the buffer was not recently in a kernel.
  void * hostPtr = this->d_ptr->GetHostPointer( this->m_Size * this->m_ElementSize );
  if( this->d_ptr->state == InKernel )
  {
    const cl_int error = clEnqueueReadBuffer
        ( this->d_ptr->context->GetActiveQueue(), this->d_ptr->id, CL_TRUE,
        0, this->m_Size * this->m_ElementSize, hostPtr, 0, 0, 0 );
    this->d_ptr->context->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
    if( error == CL_SUCCESS )
    {
      this->m_Mapped = hostPtr;
    }
  }
  else
  {
    this->m_Mapped = hostPtr;
  }
  this->d_ptr->state = InHost;
#endif

  // Update all of the other owners with the map state.
  if( this->d_ptr->owners.size() > 1 )
  {
    std::list< OpenCLVectorBase * >::iterator it;
    for( it = this->d_ptr->owners.begin(); it != this->d_ptr->owners.end(); ++it )
    {
      if( *it != this )
      {
        ( *it )->m_Mapped = this->m_Mapped;
      }
    }
  }
}


//------------------------------------------------------------------------------
void
OpenCLVectorBase::Unmap() const
{
  if( this->m_Mapped )
  {
#ifndef ITK_CL_COPY_VECTOR
    const cl_int error = clEnqueueUnmapMemObject
        ( this->d_ptr->context->GetActiveQueue(), this->d_ptr->id, this->m_Mapped, 0, 0, 0 );
    this->d_ptr->context->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
#else
    // Write the local copy back to the OpenCL device.
    if( this->d_ptr->hostCopy && this->d_ptr->state == InHost )
    {
      const cl_int error = clEnqueueWriteBuffer
          ( this->d_ptr->context->GetActiveQueue(), this->d_ptr->id, CL_FALSE,
          0, this->m_Size * this->m_ElementSize, this->d_ptr->hostCopy, 0, 0, 0 );
      this->d_ptr->context->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
    }
    this->d_ptr->state = InKernel;
#endif
    this->m_Mapped = 0;

    // Update all of the other owners with the unmap state.
    if( this->d_ptr->owners.size() > 1 )
    {
      std::list< OpenCLVectorBase * >::iterator it;
      for( it = this->d_ptr->owners.begin(); it != this->d_ptr->owners.end(); ++it )
      {
        if( *it != this )
        {
          ( *it )->m_Mapped = 0;
        }
      }
    }
  }
}


//------------------------------------------------------------------------------
void
OpenCLVectorBase::Read( void * data, const std::size_t size, const std::size_t offset /*= 0 */ )
{
  if( size == 0 )
  {
    return;
  }
  if( this->m_Mapped )
  {
    memcpy( data, reinterpret_cast< unsigned char * >( this->m_Mapped ) + offset, size );
  }
  else if( this->d_ptr && this->d_ptr->id )
  {
    const cl_int error = clEnqueueReadBuffer
        ( this->d_ptr->context->GetActiveQueue(), this->d_ptr->id, CL_TRUE,
        offset, size, data, 0, 0, 0 );
    this->d_ptr->context->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
    this->d_ptr->state = InKernel;
  }
}


//------------------------------------------------------------------------------
void
OpenCLVectorBase::Write( const void * data, const std::size_t size, const std::size_t offset /*= 0 */ )
{
  if( size == 0 )
  {
    return;
  }
  if( this->m_Mapped )
  {
    memcpy( reinterpret_cast< unsigned char * >( this->m_Mapped ) + offset, data, size );
  }
  else if( this->d_ptr && this->d_ptr->id )
  {
    const cl_int error = clEnqueueWriteBuffer
        ( this->d_ptr->context->GetActiveQueue(), this->d_ptr->id, CL_TRUE,
        offset, size, data, 0, 0, 0 );
    this->d_ptr->context->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
    this->d_ptr->state = InKernel;
  }
}


//------------------------------------------------------------------------------
cl_mem
OpenCLVectorBase::GetMemoryId() const
{
  return this->d_ptr ? this->d_ptr->id : 0;
}


//------------------------------------------------------------------------------
OpenCLContext *
OpenCLVectorBase::GetContext() const
{
  return this->d_ptr ? this->d_ptr->context : 0;
}


//------------------------------------------------------------------------------
cl_mem
OpenCLVectorBase::GetKernelArgument() const
{
  if( this->d_ptr )
  {
    this->Unmap();
    this->d_ptr->state = InKernel;
    return this->d_ptr->id;
  }
  else
  {
    return 0;
  }
}


} // end namespace itk

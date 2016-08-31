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
#include "itkOpenCLMemoryObject.h"
#include "itkOpenCLContext.h"
#include "itkOpenCLMacro.h"

namespace itk
{
OpenCLMemoryObject::~OpenCLMemoryObject()
{
  if( !this->IsNull() )
  {
    clReleaseMemObject( this->m_Id );
  }
}


//------------------------------------------------------------------------------
cl_mem_object_type
OpenCLMemoryObject::GetMemoryType() const
{
  cl_mem_object_type mem_type;

  if( clGetMemObjectInfo( this->m_Id, CL_MEM_TYPE,
    sizeof( mem_type ), &mem_type, 0 ) != CL_SUCCESS )
  {
    return 0;
  }
  else
  {
    return mem_type;
  }
}


//------------------------------------------------------------------------------
cl_mem_flags
OpenCLMemoryObject::GetFlags() const
{
  cl_mem_flags flags;

  if( clGetMemObjectInfo( this->m_Id, CL_MEM_FLAGS,
    sizeof( flags ), &flags, 0 ) != CL_SUCCESS )
  {
    return 0;
  }
  else
  {
    return flags;
  }
}


//------------------------------------------------------------------------------
size_t
OpenCLMemoryObject::GetSize() const
{
  std::size_t size;

  if( clGetMemObjectInfo( this->m_Id, CL_MEM_SIZE,
    sizeof( size ), &size, 0 ) != CL_SUCCESS )
  {
    return 0;
  }
  else
  {
    return size;
  }
}


//------------------------------------------------------------------------------
void *
OpenCLMemoryObject::GetHostPointer() const
{
  void * ptr;

  if( clGetMemObjectInfo( this->m_Id, CL_MEM_HOST_PTR,
    sizeof( ptr ), &ptr, 0 ) != CL_SUCCESS )
  {
    return 0;
  }
  else
  {
    return ptr;
  }
}


//------------------------------------------------------------------------------
cl_uint
OpenCLMemoryObject::GetMapCount() const
{
  cl_uint map_count;

  if( clGetMemObjectInfo( this->m_Id, CL_MEM_MAP_COUNT,
    sizeof( map_count ), &map_count, 0 ) != CL_SUCCESS )
  {
    return 0;
  }
  else
  {
    return map_count;
  }
}


//------------------------------------------------------------------------------
cl_uint
OpenCLMemoryObject::GetReferenceCount() const
{
  cl_uint reference_count;

  if( clGetMemObjectInfo( this->m_Id, CL_MEM_REFERENCE_COUNT,
    sizeof( reference_count ), &reference_count, 0 ) != CL_SUCCESS )
  {
    return 0;
  }
  else
  {
    return reference_count;
  }
}


//------------------------------------------------------------------------------
OpenCLMemoryObject::Access
OpenCLMemoryObject::GetAccess() const
{
  cl_mem_flags flags;

  if( clGetMemObjectInfo( this->m_Id, CL_MEM_FLAGS,
    sizeof( flags ), &flags, 0 ) != CL_SUCCESS )
  {
    return OpenCLMemoryObject::ReadWrite; // Return default value
  }
  else
  {
    return OpenCLMemoryObject::Access( flags & ( CL_MEM_READ_WRITE | CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY ) );
  }
}


//------------------------------------------------------------------------------
void
OpenCLMemoryObject::Unmap( void * ptr, const bool wait )
{
  cl_event event;
  cl_int   error;

  if( wait )
  {
    error = clEnqueueUnmapMemObject
        ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(), ptr, 0, 0, &event );
  }
  else
  {
    error = clEnqueueUnmapMemObject
        ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(), ptr, 0, 0, 0 );
  }

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  if( error == CL_SUCCESS && wait )
  {
    clWaitForEvents( 1, &event );
    clReleaseEvent( event );
  }
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLMemoryObject::UnmapAsync( void * ptr, const OpenCLEventList & event_list )
{
  cl_event     event;
  const cl_int error = clEnqueueUnmapMemObject
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(), ptr,
      event_list.GetSize(), event_list.GetEventData(), &event );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  if( error == CL_SUCCESS )
  {
    return OpenCLEvent( event );
  }
  else
  {
    return OpenCLEvent();
  }
}


//------------------------------------------------------------------------------
void
OpenCLMemoryObject::SetId( OpenCLContext * context, const cl_mem id )
{
  this->m_Context = context;
  if( id )
  {
    clRetainMemObject( id );
  }
  if( !this->IsNull() )
  {
    clReleaseMemObject( this->m_Id );
  }
  this->m_Id = id;
}


//------------------------------------------------------------------------------
cl_map_flags
OpenCLMemoryObject::GetMapFlags( const OpenCLMemoryObject::Access access )
{
  if( access == OpenCLMemoryObject::ReadOnly )
  {
    return CL_MAP_READ;
  }
  else if( access == OpenCLMemoryObject::WriteOnly )
  {
    return CL_MAP_WRITE;
  }
  else
  {
    return CL_MAP_READ | CL_MAP_WRITE;
  }
}


//------------------------------------------------------------------------------
cl_int
OpenCLMemoryObject::SetDestructorCallback(
  void ( CL_CALLBACK * pfn_notify )( cl_mem, void * ), void * user_data /*= NULL*/ )
{
  if( this->IsNull() )
  {
    return 0;
  }

  const cl_int error = clSetMemObjectDestructorCallback( this->m_Id, pfn_notify, user_data );
  if( error != CL_SUCCESS )
  {
    itkOpenCLErrorMacroGeneric( << "OpenCLMemoryObject::SetDestructorCallback:"
                                << OpenCLContext::GetErrorName( error ) );
  }
  return error;
}


//------------------------------------------------------------------------------
//! Operator ==
bool
operator==( const OpenCLMemoryObject & lhs, const OpenCLMemoryObject & rhs )
{
  if( &rhs == &lhs )
  {
    return true;
  }
  return lhs.GetMemoryId() == rhs.GetMemoryId();
}


//------------------------------------------------------------------------------
//! Operator !=
bool
operator!=( const OpenCLMemoryObject & lhs, const OpenCLMemoryObject & rhs )
{
  return !( lhs == rhs );
}


} // end namespace itk

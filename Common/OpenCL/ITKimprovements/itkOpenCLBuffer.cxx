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
#include "itkOpenCLBuffer.h"
#include "itkOpenCLImage.h"
#include "itkOpenCLContext.h"
#include "itkOpenCLExtension.h"

namespace itk
{
OpenCLBuffer::OpenCLBuffer( OpenCLContext * context, const cl_mem id ) :
  OpenCLMemoryObject( context, id )
{}

//------------------------------------------------------------------------------
OpenCLBuffer::OpenCLBuffer( const OpenCLBuffer & other ) :
  OpenCLMemoryObject()
{
  this->SetId( other.GetContext(), other.GetMemoryId() );
}


//------------------------------------------------------------------------------
OpenCLBuffer &
OpenCLBuffer::operator=( const OpenCLBuffer & other )
{
  this->SetId( other.GetContext(), other.GetMemoryId() );
  return *this;
}


//------------------------------------------------------------------------------
bool
OpenCLBuffer::Read( void * data, const std::size_t size,
  const std::size_t offset /*= 0 */ )
{
  if( size == 0 )
  {
    return false;
  }

  const cl_int error = clEnqueueReadBuffer
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(),
      CL_TRUE, offset, size, data, 0, 0, 0 );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  return error == CL_SUCCESS;
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLBuffer::ReadAsync( void * data, const std::size_t size,
  const OpenCLEventList & event_list /*= OpenCLEventList()*/,
  const std::size_t offset /*= 0 */ )
{
  if( size == 0 )
  {
    return OpenCLEvent();
  }

  cl_event     event;
  const cl_int error = clEnqueueReadBuffer
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(),
      CL_FALSE, offset, size, data,
      event_list.GetSize(), event_list.GetEventData(), &event );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
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
bool
OpenCLBuffer::Write( const void * data, const std::size_t size,
  const std::size_t offset /*= 0 */ )
{
  if( size == 0 )
  {
    return false;
  }

  const cl_int error = clEnqueueWriteBuffer
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(),
      CL_TRUE, offset, size, data, 0, 0, 0 );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  return error == CL_SUCCESS;
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLBuffer::WriteAsync( const void * data, const std::size_t size,
  const OpenCLEventList & event_list /*= OpenCLEventList()*/,
  const std::size_t offset /*= 0 */ )
{
  if( size == 0 )
  {
    return OpenCLEvent();
  }

  cl_event     event;
  const cl_int error = clEnqueueWriteBuffer
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(),
      CL_FALSE, offset, size, data,
      event_list.GetSize(), event_list.GetEventData(), &event );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
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
bool
OpenCLBuffer::ReadRect( void * data, const RectangleType & rect,
  const std::size_t bufferBytesPerLine, const std::size_t hostBytesPerLine )
{
  const std::size_t bufferOrigin[ 3 ] = { rect[ 0 ], rect[ 1 ], 0 };
  const std::size_t bufferRegion[ 3 ] = { rect[ 2 ], rect[ 3 ], 1 };
  const std::size_t hostOrigin[ 3 ]   = { 0, 0, 0 };
  const cl_int      error             = clEnqueueReadBufferRect
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(),
      CL_TRUE, bufferOrigin, hostOrigin, bufferRegion,
      bufferBytesPerLine, 0, hostBytesPerLine, 0,
      data, 0, 0, 0 );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  return error == CL_SUCCESS;
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLBuffer::ReadRectAsync( void * data, const RectangleType & rect,
  const std::size_t bufferBytesPerLine, const std::size_t hostBytesPerLine,
  const OpenCLEventList & event_list /*= OpenCLEventList()*/ )
{
  const std::size_t bufferOrigin[ 3 ] = { rect[ 0 ], rect[ 1 ], 0 };
  const std::size_t bufferRegion[ 3 ] = { rect[ 2 ], rect[ 3 ], 1 };
  const std::size_t hostOrigin[ 3 ]   = { 0, 0, 0 };
  cl_event          event;
  const cl_int      error = clEnqueueReadBufferRect
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(),
      CL_FALSE, bufferOrigin, hostOrigin, bufferRegion,
      bufferBytesPerLine, 0, hostBytesPerLine, 0, data,
      event_list.GetSize(), event_list.GetEventData(), &event );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
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
bool
OpenCLBuffer::ReadRect( void * data,
  const std::size_t origin[ 3 ], const std::size_t size[ 3 ],
  const std::size_t bufferBytesPerLine, const std::size_t bufferBytesPerSlice,
  const std::size_t hostBytesPerLine, const std::size_t hostBytesPerSlice )
{
  const std::size_t hostOrigin[ 3 ] = { 0, 0, 0 };
  const cl_int      error           = clEnqueueReadBufferRect
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(),
      CL_TRUE, origin, hostOrigin, size,
      bufferBytesPerLine, bufferBytesPerSlice,
      hostBytesPerLine, hostBytesPerSlice, data, 0, 0, 0 );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  return error == CL_SUCCESS;
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLBuffer::ReadRectAsync
  ( void * data, const std::size_t origin[ 3 ], const std::size_t size[ 3 ],
  const std::size_t bufferBytesPerLine, const std::size_t bufferBytesPerSlice,
  const std::size_t hostBytesPerLine, const std::size_t hostBytesPerSlice,
  const OpenCLEventList & event_list /*= OpenCLEventList()*/ )
{
  const std::size_t hostOrigin[ 3 ] = { 0, 0, 0 };
  cl_event          event;
  const cl_int      error = clEnqueueReadBufferRect
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(),
      CL_FALSE, origin, hostOrigin, size,
      bufferBytesPerLine, bufferBytesPerSlice,
      hostBytesPerLine, hostBytesPerSlice, data,
      event_list.GetSize(), event_list.GetEventData(), &event );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
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
bool
OpenCLBuffer::WriteRect( const void * data, const RectangleType & rect,
  const std::size_t bufferBytesPerLine, const std::size_t hostBytesPerLine )
{
  const std::size_t bufferOrigin[ 3 ] = { rect[ 0 ], rect[ 1 ], 0 };
  const std::size_t bufferRegion[ 3 ] = { rect[ 2 ], rect[ 3 ], 1 };
  const std::size_t hostOrigin[ 3 ]   = { 0, 0, 0 };
  const cl_int      error             = clEnqueueWriteBufferRect
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(),
      CL_TRUE, bufferOrigin, hostOrigin, bufferRegion,
      bufferBytesPerLine, 0, hostBytesPerLine, 0,
      data, 0, 0, 0 );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  return error == CL_SUCCESS;
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLBuffer::WriteRectAsync( const void * data, const RectangleType & rect,
  const std::size_t bufferBytesPerLine, const std::size_t hostBytesPerLine,
  const OpenCLEventList & event_list /*= OpenCLEventList()*/ )
{
  const std::size_t bufferOrigin[ 3 ] = { rect[ 0 ], rect[ 1 ], 0 };
  const std::size_t bufferRegion[ 3 ] = { rect[ 2 ], rect[ 3 ], 1 };
  const std::size_t hostOrigin[ 3 ]   = { 0, 0, 0 };
  cl_event          event;
  const cl_int      error = clEnqueueWriteBufferRect
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(),
      CL_FALSE, bufferOrigin, hostOrigin, bufferRegion,
      bufferBytesPerLine, 0, hostBytesPerLine, 0, data,
      event_list.GetSize(), event_list.GetEventData(), &event );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
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
bool
OpenCLBuffer::WriteRect( const void * data,
  const std::size_t origin[ 3 ], const std::size_t size[ 3 ],
  const std::size_t bufferBytesPerLine, const std::size_t bufferBytesPerSlice,
  const std::size_t hostBytesPerLine, const std::size_t hostBytesPerSlice )
{
  const std::size_t hostOrigin[ 3 ] = { 0, 0, 0 };
  const cl_int      error           = clEnqueueWriteBufferRect
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(),
      CL_TRUE, origin, hostOrigin, size,
      bufferBytesPerLine, bufferBytesPerSlice,
      hostBytesPerLine, hostBytesPerSlice, data, 0, 0, 0 );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  return error == CL_SUCCESS;
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLBuffer::WriteRectAsync( const void * data,
  const std::size_t origin[ 3 ], const std::size_t size[ 3 ],
  const std::size_t bufferBytesPerLine, const std::size_t bufferBytesPerSlice,
  const std::size_t hostBytesPerLine, const std::size_t hostBytesPerSlice,
  const OpenCLEventList & event_list /*= OpenCLEventList()*/ )
{
  const std::size_t hostOrigin[ 3 ] = { 0, 0, 0 };
  cl_event          event;
  const cl_int      error = clEnqueueWriteBufferRect
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(),
      CL_FALSE, origin, hostOrigin, size,
      bufferBytesPerLine, bufferBytesPerSlice,
      hostBytesPerLine, hostBytesPerSlice, data,
      event_list.GetSize(), event_list.GetEventData(), &event );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
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
bool
OpenCLBuffer::CopyToBuffer( const OpenCLBuffer & dest,
  const std::size_t size,
  const std::size_t dst_offset /*= 0*/,
  const std::size_t offset /*= 0*/  )
{
  cl_event     event;
  const cl_int error = clEnqueueCopyBuffer
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(), dest.GetMemoryId(),
      offset, dst_offset, size, 0, 0, &event );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  if( error == CL_SUCCESS )
  {
    clWaitForEvents( 1, &event );
    clReleaseEvent( event );
    return true;
  }
  else
  {
    return false;
  }
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLBuffer::CopyToBufferAsync( const OpenCLBuffer & dest,
  const std::size_t size,
  const OpenCLEventList & event_list /*= OpenCLEventList()*/,
  const std::size_t dst_offset /*= 0*/,
  const std::size_t offset /*= 0*/ )
{
  cl_event     event;
  const cl_int error = clEnqueueCopyBuffer
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(), dest.GetMemoryId(),
      offset, dst_offset, size,
      event_list.GetSize(), event_list.GetEventData(), &event );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
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
bool
OpenCLBuffer::CopyToImage( const OpenCLImage & dest,
  const OpenCLSize & origin, const OpenCLSize & region,
  const std::size_t src_offset /*= 0*/ )
{
  if( this->IsNull() || region.IsZero() )
  {
    return false;
  }

  std::size_t origin_t[ 3 ], region_t[ 3 ];
  dest.SetOrigin( origin_t, origin );
  dest.SetRegion( region_t, region );

  cl_event     event;
  const cl_int error = clEnqueueCopyBufferToImage
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(), dest.GetMemoryId(),
      src_offset, origin_t, region_t, 0, 0, &event );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  if( error == CL_SUCCESS )
  {
    clWaitForEvents( 1, &event );
    clReleaseEvent( event );
    return true;
  }
  else
  {
    return false;
  }
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLBuffer::CopyToImageAsync( const OpenCLImage & dest,
  const OpenCLSize & origin, const OpenCLSize & region,
  const OpenCLEventList & event_list /*= OpenCLEventList()*/,
  const std::size_t src_offset /*= 0*/ )
{
  if( this->IsNull() || region.IsZero() )
  {
    return OpenCLEvent();
  }

  std::size_t origin_t[ 3 ], region_t[ 3 ];
  dest.SetOrigin( origin_t, origin );
  dest.SetRegion( region_t, region );

  cl_event     event;
  const cl_int error = clEnqueueCopyBufferToImage
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(), dest.GetMemoryId(),
      src_offset, origin_t, region_t,
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
bool
OpenCLBuffer::CopyToRect( const OpenCLBuffer & dest,
  const RectangleType & rect, const PointType & destPoint,
  const std::size_t bufferBytesPerLine, const std::size_t destBytesPerLine )
{
  const std::size_t src_origin[ 3 ] = { rect[ 0 ], rect[ 1 ], 0 };
  const std::size_t dst_origin[ 3 ] = { destPoint[ 0 ], destPoint[ 1 ], 0 };
  const std::size_t region[ 3 ]     = { rect[ 2 ], rect[ 3 ], 1 };
  cl_event          event;
  const cl_int      error = clEnqueueCopyBufferRect
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(), dest.GetMemoryId(),
      src_origin, dst_origin, region,
      bufferBytesPerLine, 0, destBytesPerLine, 0, 0, 0, &event );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  if( error == CL_SUCCESS )
  {
    clWaitForEvents( 1, &event );
    clReleaseEvent( event );
    return true;
  }
  else
  {
    return false;
  }
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLBuffer::CopyToRectAsync( const OpenCLBuffer & dest,
  const RectangleType & rect, const PointType & destPoint,
  const std::size_t bufferBytesPerLine, const std::size_t destBytesPerLine,
  const OpenCLEventList & event_list /*= OpenCLEventList()*/ )
{
  const std::size_t src_origin[ 3 ] = { rect[ 0 ], rect[ 1 ], 0 };
  const std::size_t dst_origin[ 3 ] = { destPoint[ 0 ], destPoint[ 1 ], 0 };
  const std::size_t region[ 3 ]     = { rect[ 2 ], rect[ 3 ], 1 };
  cl_event          event;
  const cl_int      error = clEnqueueCopyBufferRect
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(), dest.GetMemoryId(),
      src_origin, dst_origin, region,
      bufferBytesPerLine, 0, destBytesPerLine, 0,
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
bool
OpenCLBuffer::CopyToRect( const OpenCLBuffer & dest,
  const std::size_t origin[ 3 ], const std::size_t size[ 3 ],
  const std::size_t destOrigin[ 3 ],
  const std::size_t bufferBytesPerLine, const std::size_t bufferBytesPerSlice,
  const std::size_t destBytesPerLine, const std::size_t destBytesPerSlice )
{
  cl_event     event;
  const cl_int error = clEnqueueCopyBufferRect
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(), dest.GetMemoryId(),
      origin, destOrigin, size,
      bufferBytesPerLine, bufferBytesPerSlice,
      destBytesPerLine, destBytesPerSlice, 0, 0, &event );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  if( error == CL_SUCCESS )
  {
    clWaitForEvents( 1, &event );
    clReleaseEvent( event );
    return true;
  }
  else
  {
    return false;
  }
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLBuffer::CopyToRectAsync( const OpenCLBuffer & dest,
  const std::size_t origin[ 3 ], const std::size_t size[ 3 ],
  const std::size_t destOrigin[ 3 ],
  const std::size_t bufferBytesPerLine, const std::size_t bufferBytesPerSlice,
  const std::size_t destBytesPerLine, const std::size_t destBytesPerSlice,
  const OpenCLEventList & event_list /*= OpenCLEventList()*/ )
{
  cl_event     event;
  const cl_int error = clEnqueueCopyBufferRect
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(), dest.GetMemoryId(),
      origin, destOrigin, size,
      bufferBytesPerLine, bufferBytesPerSlice,
      destBytesPerLine, destBytesPerSlice,
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
void *
OpenCLBuffer::Map( const OpenCLMemoryObject::Access access,
  const std::size_t size, const std::size_t offset /*= 0*/ )
{
  cl_int error;
  void * data = clEnqueueMapBuffer
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(),
      CL_TRUE, this->GetMapFlags( access ), offset, size, 0, 0, 0, &error );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  return data;
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLBuffer::MapAsync( void ** ptr, const OpenCLMemoryObject::Access access,
  const std::size_t size,
  const OpenCLEventList & event_list /*= OpenCLEventList()*/,
  const std::size_t offset /*= 0*/ )
{
  cl_int   error;
  cl_event event;

  *ptr = clEnqueueMapBuffer
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(),
      CL_FALSE, this->GetMapFlags( access ), offset, size,
      event_list.GetSize(), event_list.GetEventData(), &event, &error );
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
void *
OpenCLBuffer::Map( const OpenCLMemoryObject::Access access )
{
  return this->Map( access, this->GetSize(), 0 );
}


//------------------------------------------------------------------------------
OpenCLBuffer
OpenCLBuffer::CreateSubBuffer( const OpenCLMemoryObject::Access access,
  const std::size_t size, const std::size_t offset /*= 0 */ )

{
  cl_int           error;
  cl_buffer_region region;

  region.origin = offset;
  region.size   = size;
  cl_mem mem = clCreateSubBuffer
      ( this->GetMemoryId(), cl_mem_flags( access ),
      CL_BUFFER_CREATE_TYPE_REGION, &region, &error );
  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  return OpenCLBuffer( this->GetContext(), mem );
}


//------------------------------------------------------------------------------
OpenCLBuffer
OpenCLBuffer::GetParentBuffer() const
{
  cl_mem parent;

  if( clGetMemObjectInfo( this->GetMemoryId(), CL_MEM_ASSOCIATED_MEMOBJECT,
    sizeof( parent ), &parent, 0 ) != CL_SUCCESS )
  {
    return OpenCLBuffer();
  }
  if( parent )
  {
    clRetainMemObject( parent );
  }
  return OpenCLBuffer( this->GetContext(), parent );
}


//------------------------------------------------------------------------------
size_t
OpenCLBuffer::GetOffset() const
{
  std::size_t value;

  if( clGetMemObjectInfo( this->GetMemoryId(), CL_MEM_OFFSET,
    sizeof( value ), &value, 0 ) != CL_SUCCESS )
  {
    return 0;
  }
  else
  {
    return value;
  }
}


} // end namespace itk

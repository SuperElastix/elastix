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
#include "itkOpenCLImage.h"
#include "itkOpenCLContext.h"
#include "itkOpenCLBuffer.h"

namespace itk
{
OpenCLImage::OpenCLImage( const OpenCLImage & other ) : OpenCLMemoryObject()
{
  this->SetId( other.GetContext(), other.GetMemoryId() );
}


//------------------------------------------------------------------------------
OpenCLImage &
OpenCLImage::operator=( const OpenCLImage & other )
{
  this->SetId( other.GetContext(), other.GetMemoryId() );
  return *this;
}


//------------------------------------------------------------------------------
itk::OpenCLImageFormat
OpenCLImage::GetFormat() const
{
  cl_image_format format;

  if( clGetImageInfo
      ( this->GetMemoryId(), CL_IMAGE_FORMAT, sizeof( format ), &format, 0 )
    != CL_SUCCESS )
  {
    return OpenCLImageFormat();
  }
  else
  {
    return OpenCLImageFormat(
      OpenCLImageFormat::ChannelOrder( format.image_channel_order ),
      OpenCLImageFormat::ChannelType( format.image_channel_data_type ) );
  }
}


//------------------------------------------------------------------------------
std::size_t
OpenCLImage::GetElementSizeInBytes() const
{
  return this->GetImageInfo( CL_IMAGE_ELEMENT_SIZE );
}


//------------------------------------------------------------------------------
std::size_t
OpenCLImage::GetRowSizeInBytes() const
{
  return this->GetImageInfo( CL_IMAGE_ROW_PITCH );
}


//------------------------------------------------------------------------------
std::size_t
OpenCLImage::GetSliceSizeInBytes() const
{
  return this->GetImageInfo( CL_IMAGE_SLICE_PITCH );
}


//------------------------------------------------------------------------------
std::size_t
OpenCLImage::GetDimension() const
{
  if( this->IsNull() )
  {
    return 0;
  }
  else
  {
    const std::size_t width  = this->GetWidth();
    const std::size_t height = this->GetHeight();
    const std::size_t depth  = this->GetDepth();

    if( width > 0 && height == 0 && depth == 0 )
    {
      return 1;
    }
    else if( width > 0 && height > 0 && depth == 0 )
    {
      return 2;
    }
    else if( width > 0 && height > 0 && depth > 0 )
    {
      return 3;
    }
    else
    {
      return 0;
    }
  }
}


//------------------------------------------------------------------------------
std::size_t
OpenCLImage::GetWidth() const
{
  return this->GetImageInfo( CL_IMAGE_WIDTH );
}


//------------------------------------------------------------------------------
std::size_t
OpenCLImage::GetHeight() const
{
  return this->GetImageInfo( CL_IMAGE_HEIGHT );
}


//------------------------------------------------------------------------------
std::size_t
OpenCLImage::GetDepth() const
{
  return this->GetImageInfo( CL_IMAGE_DEPTH );
}


//------------------------------------------------------------------------------
bool
OpenCLImage::Read( void * data,
  const OpenCLSize & origin, const OpenCLSize & region,
  const std::size_t rowPitch /*= 0*/, const std::size_t slicePitch /*= 0 */ )
{
  if( this->IsNull() || region.IsZero() )
  {
    return false;
  }

  std::size_t origin_t[ 3 ], region_t[ 3 ];
  this->SetOrigin( origin_t, origin );
  this->SetRegion( region_t, region );

  const cl_int error = clEnqueueReadImage( this->GetContext()->GetActiveQueue(),
    this->GetMemoryId(), CL_TRUE,
    origin_t, region_t, rowPitch, slicePitch, data, 0, 0, 0 );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  return error == CL_SUCCESS;
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLImage::ReadAsync( void * data,
  const OpenCLSize & origin, const OpenCLSize & region,
  const OpenCLEventList & event_list /* = OpenCLEventList()*/,
  const std::size_t rowPitch /*= 0*/, const std::size_t slicePitch /*= 0 */ )
{
  if( this->IsNull() || region.IsZero() )
  {
    return OpenCLEvent();
  }

  std::size_t origin_t[ 3 ], region_t[ 3 ];
  this->SetOrigin( origin_t, origin );
  this->SetRegion( region_t, region );

  cl_event     event;
  const cl_int error = clEnqueueReadImage( this->GetContext()->GetActiveQueue(),
    this->GetMemoryId(), CL_FALSE,
    origin_t, region_t, rowPitch, slicePitch, data,
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
OpenCLImage::Write( const void * data,
  const OpenCLSize & origin, const OpenCLSize & region,
  const std::size_t rowPitch /*= 0*/, const std::size_t slicePitch /*= 0 */ )
{
  if( this->IsNull() || region.IsZero() )
  {
    return false;
  }

  std::size_t origin_t[ 3 ], region_t[ 3 ];
  this->SetOrigin( origin_t, origin );
  this->SetRegion( region_t, region );

  const cl_int error = clEnqueueWriteImage( this->GetContext()->GetActiveQueue(),
    this->GetMemoryId(), CL_TRUE,
    origin_t, region_t, rowPitch, slicePitch, data, 0, 0, 0 );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  return error == CL_SUCCESS;
}


//------------------------------------------------------------------------------
OpenCLEvent
OpenCLImage::WriteAsync( const void * data,
  const OpenCLSize & origin, const OpenCLSize & region,
  const OpenCLEventList & event_list /* = OpenCLEventList()*/,
  const std::size_t rowPitch /*= 0*/, const std::size_t slicePitch /*= 0 */ )
{
  if( this->IsNull() || region.IsZero() )
  {
    return OpenCLEvent();
  }

  std::size_t origin_t[ 3 ], region_t[ 3 ];
  this->SetOrigin( origin_t, origin );
  this->SetRegion( region_t, region );

  cl_event     event;
  const cl_int error = clEnqueueWriteImage( this->GetContext()->GetActiveQueue(),
    this->GetMemoryId(), CL_FALSE,
    origin_t, region_t, rowPitch, slicePitch, data,
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
OpenCLImage::Map(
  const OpenCLMemoryObject::Access access,
  const OpenCLSize & origin, const OpenCLSize & region,
  std::size_t * rowPitch /*= 0*/,
  std::size_t * slicePitch /*= 0 */ )
{
  if( this->IsNull() || region.IsZero() )
  {
    return NULL;
  }

  cl_int      error;
  std::size_t row_pitch, slice_pitch;
  std::size_t origin_t[ 3 ], region_t[ 3 ];
  this->SetOrigin( origin_t, origin );
  this->SetRegion( region_t, region );

  void * data = clEnqueueMapImage( this->GetContext()->GetActiveQueue(), this->GetMemoryId(),
    CL_TRUE, this->GetMapFlags( access ), origin_t, region_t,
    &row_pitch, &slice_pitch, 0, 0, 0, &error );

  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  if( rowPitch )
  {
    *rowPitch = row_pitch;
  }
  if( slicePitch )
  {
    *slicePitch = slice_pitch;
  }
  return data;
}


//------------------------------------------------------------------------------
itk::OpenCLEvent
OpenCLImage::MapAsync( void ** data,
  const OpenCLMemoryObject::Access access,
  const OpenCLSize & origin, const OpenCLSize & region,
  const OpenCLEventList & event_list /*= OpenCLEventList()*/,
  std::size_t * rowPitch /*= 0*/,
  std::size_t * slicePitch /*= 0 */ )
{
  if( this->IsNull() || region.IsZero() )
  {
    return OpenCLEvent();
  }

  cl_int      error;
  cl_event    event;
  std::size_t row_pitch, slice_pitch;
  std::size_t origin_t[ 3 ], region_t[ 3 ];
  this->SetOrigin( origin_t, origin );
  this->SetRegion( region_t, region );

  *data = clEnqueueMapImage( this->GetContext()->GetActiveQueue(), this->GetMemoryId(),
    CL_FALSE, this->GetMapFlags( access ), origin_t, region_t,
    &row_pitch, &slice_pitch,
    event_list.GetSize(), event_list.GetEventData(), &event, &error );
  this->GetContext()->ReportError( error, __FILE__, __LINE__, ITK_LOCATION );
  if( rowPitch )
  {
    *rowPitch = row_pitch;
  }
  if( slicePitch )
  {
    *slicePitch = slice_pitch;
  }

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
OpenCLImage::Copy( const OpenCLImage & dest,
  const OpenCLSize & origin, const OpenCLSize & region,
  const OpenCLSize & destOrigin )
{
  if( this->IsNull() || region.IsZero() )
  {
    return false;
  }

  std::size_t origin_t[ 3 ], region_t[ 3 ], dest_origin_t[ 3 ];
  this->SetOrigin( origin_t, origin );
  this->SetRegion( region_t, region );
  this->SetRegion( dest_origin_t, destOrigin );

  cl_event     event;
  const cl_int error = clEnqueueCopyImage
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(), dest.GetMemoryId(),
      origin_t, dest_origin_t, region_t, 0, 0, &event );

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
OpenCLImage::CopyAsync( const OpenCLImage & dest,
  const OpenCLSize & origin, const OpenCLSize & region,
  const OpenCLSize & destOrigin,
  const OpenCLEventList & event_list /*= OpenCLEventList() */ )
{
  if( this->IsNull() || region.IsZero() )
  {
    return OpenCLEvent();
  }

  std::size_t origin_t[ 3 ], region_t[ 3 ], dest_origin_t[ 3 ];
  this->SetOrigin( origin_t, origin );
  this->SetRegion( region_t, region );
  this->SetRegion( dest_origin_t, destOrigin );

  cl_event     event;
  const cl_int error = clEnqueueCopyImage
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(), dest.GetMemoryId(),
      origin_t, dest_origin_t, region_t,
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
OpenCLImage::Copy( const OpenCLBuffer & dest,
  const OpenCLSize & origin, const OpenCLSize & region,
  const std::size_t dst_offset /*= 0*/ )
{
  if( this->IsNull() || region.IsZero() )
  {
    return false;
  }

  std::size_t origin_t[ 3 ], region_t[ 3 ];
  this->SetOrigin( origin_t, origin );
  this->SetRegion( region_t, region );

  cl_event     event;
  const cl_int error = clEnqueueCopyImageToBuffer
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(), dest.GetMemoryId(),
      origin_t, region_t, dst_offset, 0, 0, &event );

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
OpenCLImage::CopyAsync( const OpenCLBuffer & dest,
  const OpenCLSize & origin, const OpenCLSize & region,
  const OpenCLEventList & event_list, /*= OpenCLEventList() */
  const std::size_t dst_offset /*= 0*/ )
{
  if( this->IsNull() || region.IsZero() )
  {
    return OpenCLEvent();
  }

  std::size_t origin_t[ 3 ], region_t[ 3 ];
  this->SetOrigin( origin_t, origin );
  this->SetRegion( region_t, region );

  cl_event     event;
  const cl_int error = clEnqueueCopyImageToBuffer
      ( this->GetContext()->GetActiveQueue(), this->GetMemoryId(), dest.GetMemoryId(),
      origin_t, region_t, dst_offset,
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
std::size_t
OpenCLImage::GetImageInfo( const cl_image_info name ) const
{
  std::size_t value = 0;

  if( clGetImageInfo( this->GetMemoryId(), name, sizeof( value ), &value, 0 ) != CL_SUCCESS )
  {
    return 0;
  }
  else
  {
    return value;
  }
}


//------------------------------------------------------------------------------
void
OpenCLImage::SetOrigin( std::size_t * origin_t, const OpenCLSize & origin ) const
{
  this->SetSize( origin_t, origin, 0 );
}


//------------------------------------------------------------------------------
void
OpenCLImage::SetRegion( std::size_t * region_t, const OpenCLSize & region ) const
{
  this->SetSize( region_t, region, 1 );
}


//------------------------------------------------------------------------------
void
OpenCLImage::SetSize( std::size_t * s_t, const OpenCLSize & s, const std::size_t v ) const
{
  for( std::size_t i = 0; i < 3; ++i )
  {
    s_t[ i ] = v;
  }

  const std::size_t dim = this->GetDimension();

  switch( dim )
  {
    case 1:
      s_t[ 0 ] = s[ 0 ];
      break;
    case 2:
      for( std::size_t i = 0; i < 2; ++i )
      {
        s_t[ i ] = s[ i ];
      }
      break;
    case 3:
      for( std::size_t i = 0; i < 3; ++i )
      {
        s_t[ i ] = s[ i ];
      }
      break;
    default:
      break;
  }
}


//------------------------------------------------------------------------------
#ifdef CL_VERSION_1_2
void
OpenCLImage::SetImageDescription( cl_image_desc & imageDescription,
  const OpenCLImageFormat & format, const OpenCLSize & size )
{
  memset( &imageDescription, '\0', sizeof( cl_image_desc ) );
  imageDescription.image_type = format.m_ImageType;

  switch( size.GetDimension() )
  {
    case 1:
      imageDescription.image_width = size[ 0 ];
      break;
    case 2:
      imageDescription.image_width  = size[ 0 ];
      imageDescription.image_height = size[ 1 ];
      break;
    case 3:
      imageDescription.image_width  = size[ 0 ];
      imageDescription.image_height = size[ 1 ];
      imageDescription.image_depth  = size[ 2 ];
      break;
  }
}


#endif // CL_VERSION_1_2

} // end namespace itk

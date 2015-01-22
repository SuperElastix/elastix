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
#include "itkOpenCLEvent.h"
#include "itkOpenCLContext.h"
#include "itkOpenCLMacro.h"

namespace itk
{
OpenCLEvent::OpenCLEvent() :
  m_Id( 0 )
{}

//------------------------------------------------------------------------------
OpenCLEvent::OpenCLEvent( const cl_event id ) :
  m_Id( id )
{}

//------------------------------------------------------------------------------
OpenCLEvent::OpenCLEvent( const OpenCLEvent & other ) :
  m_Id( other.m_Id )
{
  if( !this->IsNull() )
  {
    clRetainEvent( this->m_Id );
  }
}


//------------------------------------------------------------------------------
OpenCLEvent::~OpenCLEvent()
{
  if( !this->IsNull() )
  {
    clReleaseEvent( this->m_Id );
  }
}


//------------------------------------------------------------------------------
inline OpenCLEvent &
OpenCLEvent::operator=( const OpenCLEvent & other )
{
  if( other.m_Id )
  {
    clRetainEvent( other.m_Id );
  }
  if( this->m_Id )
  {
    clReleaseEvent( this->m_Id );
  }
  this->m_Id = other.m_Id;
  return *this;
}


//------------------------------------------------------------------------------
cl_int
OpenCLEvent::GetStatus() const
{
  if( this->IsNull() )
  {
    return CL_INVALID_EVENT;
  }

  cl_int st, error;
  error = clGetEventInfo( m_Id, CL_EVENT_COMMAND_EXECUTION_STATUS,
    sizeof( st ), &st, 0 );
  if( error != CL_SUCCESS )
  {
    return error;
  }
  else
  {
    return st;
  }
}


//------------------------------------------------------------------------------
cl_command_type
OpenCLEvent::GetCommandType() const
{
  if( this->IsNull() )
  {
    return 0;
  }

  cl_command_type type;
  cl_int          error = clGetEventInfo( this->m_Id, CL_EVENT_COMMAND_TYPE,
    sizeof( type ), &type, 0 );
  if( error != CL_SUCCESS )
  {
    return 0;
  }
  else
  {
    return type;
  }
}


//------------------------------------------------------------------------------
cl_int
OpenCLEvent::WaitForFinished()
{
  if( this->IsNull() )
  {
    return 0;
  }

  const cl_int error = clWaitForEvents( 1, &this->m_Id );
  if( error != CL_SUCCESS )
  {
    itkOpenCLErrorMacroGeneric( << "OpenCLEvent::WaitForFinished:"
                                << OpenCLContext::GetErrorName( error ) );
  }
  return error;
}


//------------------------------------------------------------------------------
cl_int
OpenCLEvent::SetCallback( cl_int type,
  void( CL_CALLBACK * pfn_notify )( cl_event, cl_int, void * ), void * user_data )
{
  if( this->IsNull() )
  {
    return 0;
  }

  const cl_int error = clSetEventCallback( m_Id, type, pfn_notify, user_data );
  if( error != CL_SUCCESS )
  {
    itkOpenCLErrorMacroGeneric( << "OpenCLEvent::SetCallback:"
                                << OpenCLContext::GetErrorName( error ) );
  }
  return error;
}


//------------------------------------------------------------------------------
cl_ulong
OpenCLEvent::GetQueueTime() const
{
  if( this->IsNull() )
  {
    return 0;
  }

  cl_ulong time;
  if( clGetEventProfilingInfo
      ( this->m_Id, CL_PROFILING_COMMAND_QUEUED,
    sizeof( time ), &time, 0 ) != CL_SUCCESS )
  {
    return 0;
  }
  return time;
}


//------------------------------------------------------------------------------
cl_ulong
OpenCLEvent::GetSubmitTime() const
{
  if( this->IsNull() )
  {
    return 0;
  }

  cl_ulong time;
  if( clGetEventProfilingInfo
      ( this->m_Id, CL_PROFILING_COMMAND_SUBMIT,
    sizeof( time ), &time, 0 ) != CL_SUCCESS )
  {
    return 0;
  }
  return time;
}


//------------------------------------------------------------------------------
cl_ulong
OpenCLEvent::GetRunTime() const
{
  if( this->IsNull() )
  {
    return 0;
  }

  cl_ulong time;
  if( clGetEventProfilingInfo
      ( this->m_Id, CL_PROFILING_COMMAND_START,
    sizeof( time ), &time, 0 ) != CL_SUCCESS )
  {
    return 0;
  }
  return time;
}


//------------------------------------------------------------------------------
cl_ulong
OpenCLEvent::GetFinishTime() const
{
  if( this->IsNull() )
  {
    return 0;
  }

  cl_ulong time;
  if( clGetEventProfilingInfo
      ( this->m_Id, CL_PROFILING_COMMAND_END,
    sizeof( time ), &time, 0 ) != CL_SUCCESS )
  {
    return 0;
  }
  return time;
}


//------------------------------------------------------------------------------
//! Operator ==
bool
operator==( const OpenCLEvent & lhs, const OpenCLEvent & rhs )
{
  if( &rhs == &lhs )
  {
    return true;
  }
  return lhs.GetEventId() == rhs.GetEventId();
}


//------------------------------------------------------------------------------
//! Operator !=
bool
operator!=( const OpenCLEvent & lhs, const OpenCLEvent & rhs )
{
  return !( lhs == rhs );
}


} // namespace itk

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
#include "itkOpenCLEvent.h"
#include "itkOpenCLUtil.h"

namespace itk
{
//------------------------------------------------------------------------------
OpenCLEvent::OpenCLEvent(const OpenCLEvent & other):
  m_Id(other.m_Id)
{
  if ( m_Id )
    {
    clRetainEvent(m_Id);
    }
}

//------------------------------------------------------------------------------
OpenCLEvent::~OpenCLEvent()
{
  if ( m_Id )
    {
    clReleaseEvent(m_Id);
    }
}

//------------------------------------------------------------------------------
inline OpenCLEvent & OpenCLEvent::operator=(const OpenCLEvent & other)
{
  if ( other.m_Id )
    {
    clRetainEvent(other.m_Id);
    }
  if ( m_Id )
    {
    clReleaseEvent(m_Id);
    }
  m_Id = other.m_Id;
  return *this;
}

//------------------------------------------------------------------------------
cl_int OpenCLEvent::GetStatus() const
{
  if ( !m_Id )
    {
    return CL_INVALID_EVENT;
    }

  cl_int st, error;
  error = clGetEventInfo(m_Id, CL_EVENT_COMMAND_EXECUTION_STATUS,
                         sizeof( st ), &st, 0);
  if ( error != CL_SUCCESS )
    {
    return error;
    }
  else
    {
    return st;
    }
}

//------------------------------------------------------------------------------
cl_command_type OpenCLEvent::GetCommandType() const
{
  if ( !m_Id )
    {
    return 0;
    }

  cl_command_type type;
  cl_int          error = clGetEventInfo(m_Id, CL_EVENT_COMMAND_TYPE,
                                         sizeof( type ), &type, 0);
  if ( error != CL_SUCCESS )
    {
    return 0;
    }
  else
    {
    return type;
    }
}

//------------------------------------------------------------------------------
void OpenCLEvent::WaitForFinished()
{
  if ( m_Id )
    {
    cl_int error = clWaitForEvents(1, &m_Id);
    OpenCLCheckError(error, __FILE__, __LINE__, ITK_LOCATION);
    }
}

//------------------------------------------------------------------------------
cl_ulong OpenCLEvent::GetQueueTime() const
{
  cl_ulong time;

  if ( clGetEventProfilingInfo
         (m_Id, CL_PROFILING_COMMAND_QUEUED,
         sizeof( time ), &time, 0) != CL_SUCCESS )
    {
    return 0;
    }
  return time;
}

//------------------------------------------------------------------------------
cl_ulong OpenCLEvent::GetSubmitTime() const
{
  cl_ulong time;

  if ( clGetEventProfilingInfo
         (m_Id, CL_PROFILING_COMMAND_SUBMIT,
         sizeof( time ), &time, 0) != CL_SUCCESS )
    {
    return 0;
    }
  return time;
}

//------------------------------------------------------------------------------
cl_ulong OpenCLEvent::GetRunTime() const
{
  cl_ulong time;

  if ( clGetEventProfilingInfo
         (m_Id, CL_PROFILING_COMMAND_START,
         sizeof( time ), &time, 0) != CL_SUCCESS )
    {
    return 0;
    }
  return time;
}

//------------------------------------------------------------------------------
cl_ulong OpenCLEvent::GetFinishTime() const
{
  cl_ulong time;

  if ( clGetEventProfilingInfo
         (m_Id, CL_PROFILING_COMMAND_END,
         sizeof( time ), &time, 0) != CL_SUCCESS )
    {
    return 0;
    }
  return time;
}

//------------------------------------------------------------------------------
OpenCLEventList::OpenCLEventList(const OpenCLEvent & event)
{
  cl_event id = event.GetEventId();

  if ( id )
    {
    clRetainEvent(id);
    m_Events.push_back(id);
    }
}

//------------------------------------------------------------------------------
OpenCLEventList::OpenCLEventList(const OpenCLEventList & other):
  m_Events(other.m_Events)
{
  for ( std::size_t index = 0; index < m_Events.size(); ++index )
    {
    clRetainEvent(m_Events[index]);
    }
}

//------------------------------------------------------------------------------
OpenCLEventList::~OpenCLEventList()
{
  for ( std::size_t index = 0; index < m_Events.size(); ++index )
    {
    clReleaseEvent(m_Events[index]);
    }
}

//------------------------------------------------------------------------------
OpenCLEventList & OpenCLEventList::operator=(const OpenCLEventList & other)
{
  if ( this != &other )
    {
    for ( std::size_t index = 0; index < m_Events.size(); ++index )
      {
      clReleaseEvent(m_Events[index]);
      }
    m_Events = other.m_Events;
    for ( int index = 0; index < m_Events.size(); ++index )
      {
      clRetainEvent(m_Events[index]);
      }
    }
  return *this;
}

//------------------------------------------------------------------------------
void OpenCLEventList::Append(const OpenCLEvent & event)
{
  cl_event id = event.GetEventId();

  if ( id )
    {
    clRetainEvent(id);
    m_Events.push_back(id);
    }
}

//------------------------------------------------------------------------------
void OpenCLEventList::Append(const OpenCLEventList & other)
{
  for ( std::size_t index = 0; index < other.m_Events.size(); ++index )
    {
    cl_event id = other.m_Events[index];
    clRetainEvent(id);
    m_Events.push_back(id);
    }
}

//------------------------------------------------------------------------------
void OpenCLEventList::Remove(const OpenCLEvent & event)
{
  std::vector< cl_event >::iterator it;

  for ( it = m_Events.begin(); it < m_Events.end(); it++ )
    {
    if ( *it == event.GetEventId() )
      {
      clReleaseEvent(*it);
      m_Events.erase(it);
      }
    }
}

//------------------------------------------------------------------------------
OpenCLEvent OpenCLEventList::Get(const std::size_t index) const
{
  if ( index >= 0 && index < m_Events.size() )
    {
    cl_event id = m_Events[index];
    clRetainEvent(id);
    return OpenCLEvent(id);
    }
  else
    {
    return OpenCLEvent();
    }
}

//------------------------------------------------------------------------------
bool OpenCLEventList::Contains(const OpenCLEvent & event) const
{
  std::vector< cl_event >::const_iterator it;

  for ( it = m_Events.begin(); it < m_Events.end(); it++ )
    {
    if ( *it == event.GetEventId() )
      {
      return true;
      }
    }
  return false;
}

//------------------------------------------------------------------------------
void OpenCLEventList::WaitForFinished()
{
  if ( m_Events.empty() )
    {
    return;
    }
  cl_int error = clWaitForEvents( GetSize(), GetEventData() );
  OpenCLCheckError(error, __FILE__, __LINE__, ITK_LOCATION);
}
} // namespace itk

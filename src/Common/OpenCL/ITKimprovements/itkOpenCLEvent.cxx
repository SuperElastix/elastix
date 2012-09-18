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
    //if (error != CL_SUCCESS) {
    //    qWarning() << "OpenCLEventList::waitForFinished:"
    //               << QCLContext::errorName(error);
    //}
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

#ifndef QT_NO_DEBUG_STREAM

//QDebug operator<<(QDebug dbg, const OpenCLEvent &event)
//{
//    cl_event id = event.eventId();
//    if (!id) {
//        dbg << "OpenCLEvent()";
//        return dbg;
//    }
//    cl_command_type command = event.commandType();
//    cl_int status = event.status();
//    const char *commandName;
//    switch (command) {
//    case CL_COMMAND_NDRANGE_KERNEL:
//        commandName = "clEnqueueNDRangeKernel"; break;
//    case CL_COMMAND_TASK:
//        commandName = "clEnqueueTask"; break;
//    case CL_COMMAND_NATIVE_KERNEL:
//        commandName = "clEnqueueNativeKernel"; break;
//    case CL_COMMAND_READ_BUFFER:
//        commandName = "clEnqueueReadBuffer"; break;
//    case CL_COMMAND_WRITE_BUFFER:
//        commandName = "clEnqueueWriteBuffer"; break;
//    case CL_COMMAND_COPY_BUFFER:
//        commandName = "clEnqueueCopyBuffer"; break;
//    case CL_COMMAND_READ_IMAGE:
//        commandName = "clEnqueueReadImage"; break;
//    case CL_COMMAND_WRITE_IMAGE:
//        commandName = "clEnqueueWriteImage"; break;
//    case CL_COMMAND_COPY_IMAGE:
//        commandName = "clEnqueueCopyImage"; break;
//    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
//        commandName = "clEnqueueCopyImageToBuffer"; break;
//    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
//        commandName = "clEnqueueCopyBufferToImage"; break;
//    case CL_COMMAND_MAP_BUFFER:
//        commandName = "clEnqueueMapBuffer"; break;
//    case CL_COMMAND_MAP_IMAGE:
//        commandName = "clEnqueueMapImage"; break;
//    case CL_COMMAND_UNMAP_MEM_OBJECT:
//        commandName = "clEnqueueUnmapMemObject"; break;
//    case CL_COMMAND_MARKER:
//        commandName = "clEnqueueMarker"; break;
//    case CL_COMMAND_ACQUIRE_GL_OBJECTS:
//        commandName = "clEnqueueAcquireGLObjects"; break;
//    case CL_COMMAND_RELEASE_GL_OBJECTS:
//        commandName = "clEnqueueReleaseGLObjects"; break;
//    // OpenCL 1.1 event types.
//    case CL_COMMAND_READ_BUFFER_RECT:
//        commandName = "clEnqueueReadBufferRect"; break;
//    case CL_COMMAND_WRITE_BUFFER_RECT:
//        commandName = "clEnqueueWriteBufferRect"; break;
//    case CL_COMMAND_COPY_BUFFER_RECT:
//        commandName = "clEnqueueCopyBufferRect"; break;
//    case CL_COMMAND_USER:
//        commandName = "clCreateUserEvent"; break;
//    default:
//        commandName = "Unknown"; break;
//    }
//    const char *statusName;
//    switch (status) {
//    case CL_COMPLETE:   statusName = "finished"; break;
//    case CL_RUNNING:    statusName = "running"; break;
//    case CL_SUBMITTED:  statusName = "submitted"; break;
//    case CL_QUEUED:     statusName = "queued"; break;
//    default:            statusName = "Unknown"; break;
//    }
//    if (status != CL_COMPLETE) {
//        // Command is not complete: no profiling information yet.
//        dbg << "OpenCLEvent(id:" << reinterpret_cast<long>(id)
//            << "request:" << commandName
//            << "status:" << statusName
//            << ")";
//    } else {
//        cl_ulong queueTime, runTime, finishTime;
//        if (clGetEventProfilingInfo
//                (id, CL_PROFILING_COMMAND_QUEUED,
//                 sizeof(queueTime), &queueTime, 0) != CL_SUCCESS ||
//            clGetEventProfilingInfo
//                (id, CL_PROFILING_COMMAND_START,
//                 sizeof(runTime), &runTime, 0) != CL_SUCCESS ||
//            clGetEventProfilingInfo
//                (id, CL_PROFILING_COMMAND_END,
//                 sizeof(finishTime), &finishTime, 0) != CL_SUCCESS) {
//            // Profiling information is not available, probably
//            // because it was not enabled on the command queue.
//            dbg << "OpenCLEvent(id:" << reinterpret_cast<long>(id)
//                << "request:" << commandName
//                << "status:" << statusName
//                << ")";
//        } else {
//            // Include profiling information in the debug output.
//            qreal fullDuration = (finishTime - queueTime) / 1000000.0f;
//            qreal runDuration = (finishTime - runTime) / 1000000.0f;
//            dbg << "OpenCLEvent(id:" << reinterpret_cast<long>(id)
//                << "request:" << commandName
//                << "status:" << statusName
//                << "full-time:" << fullDuration
//                << "ms running-time:" << runDuration
//                << "ms)";
//        }
//    }
//    return dbg;
//}

#endif // QT_NO_DEBUG_STREAM

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
  for ( int index = 0; index < m_Events.size(); ++index )
    {
    clRetainEvent(m_Events[index]);
    }
}

//------------------------------------------------------------------------------
OpenCLEventList::~OpenCLEventList()
{
  for ( int index = 0; index < m_Events.size(); ++index )
    {
    clReleaseEvent(m_Events[index]);
    }
}

//------------------------------------------------------------------------------
OpenCLEventList & OpenCLEventList::operator=(const OpenCLEventList & other)
{
  if ( this != &other )
    {
    for ( int index = 0; index < m_Events.size(); ++index )
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
  for ( int index = 0; index < other.m_Events.size(); ++index )
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
OpenCLEvent OpenCLEventList::At(size_t index) const
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
  //if (error != CL_SUCCESS) {
  //    qWarning() << "OpenCLEventList::waitForFinished:"
  //               << QCLContext::errorName(error);
  //}
}
} // namespace itk

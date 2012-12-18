/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkOpenCLEvent_h
#define __itkOpenCLEvent_h

#include "itkOpenCL.h"
#include "itkOpenCLExtension.h"
#include <vector>

namespace itk
{
/** \class OpenCLEvent
 * \brief OpenCLEvent class represents an OpenCL event object.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * This implementation was taken from elastix (http://elastix.isi.uu.nl/).
 *
 * \note This work was funded by the Netherlands Organisation for
 *  Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup ITKGPUCommon
 */
class OpenCLUserEvent;

class ITKOpenCL_EXPORT OpenCLEvent
{
public:
  /** Constructs a null OpenCL event object. */
  OpenCLEvent():m_Id(0) {}

  /** Constructs an OpenCL event object from the identifier id.
   * This class takes over ownership of id and will release it in
   * the destructor. */
  OpenCLEvent(cl_event id):m_Id(id) {}

  /** Constructs a copy of  other.  The \c{clRetainEvent()} function
   * will be called to update the reference count on eventId(). */
  OpenCLEvent(const OpenCLEvent & other);

  /** Releases this OpenCL event object by calling clReleaseEvent(). */
  ~OpenCLEvent();

  /** Assigns other to this OpenCL event object. The current eventId() will
   * be released with clReleaseEvent(), and the new eventId() will be
   * retained with clRetainEvent(). */
  OpenCLEvent & operator=(const OpenCLEvent & other);

  /** Returns true if this OpenCL event object is null; false otherwise. */
  bool IsNull() const { return m_Id == 0; }

  /** Returns the OpenCL identifier for this event. */
  cl_event GetEventId() const { return m_Id; }

  /** Returns true if the command associated with this OpenCL event has been
   * queued for execution on the host, but has not yet been submitted to
   * the device yet.
   * \sa IsSubmitted(), IsRunning(), IsFinished(), IsError(), GetStatus() */
  bool IsQueued() const { return GetStatus() == CL_QUEUED; }

  /** Returns true if the command associated with this OpenCL event has been
   * submitted for execution on the device yet, but is not yet running.
   * \sa IsQueued(), IsRunning(), IsFinished(), IsError(), GetStatus() */
  bool IsSubmitted() const { return GetStatus() == CL_SUBMITTED; }

  /** Returns true if the command associated with this OpenCL event is
   * running on the device, but has not yet finished.
   * \sa IsQueued(), IsSubmitted(), IsFinished(), IsError(), GetStatus() */
  bool IsRunning() const { return GetStatus() == CL_RUNNING; }

  /** Returns true if the command associated with this OpenCL event
   * has finished execution on the device.
   * \sa IsQueued(), IsSubmitted(), IsRunning(), IsError(), GetStatus() */
  bool IsFinished() const { return GetStatus() == CL_COMPLETE; }

  /* Returns true if an error has occurred on this OpenCL event.
   * \sa IsQueued(), IsSubmitted(), IsRunning(), IsFinished(), GetStatus() */
  bool IsError() const { return GetStatus() < 0; }

  /** Returns the status of this event, which is an error code or one
   * of CL_QUEUED, CL_SUBMITTED, CL_RUNNING or CL_COMPLETE.
   * \sa IsQueued(), IsSubmitted(), IsRunning(), IsFinished(), IsError() */
  cl_int GetStatus() const;

  /** Returns the type of command that generated this event. */
  cl_command_type GetCommandType() const;

  /** Waits for this event to be signaled as finished. The calling thread
   * is blocked until the event is signaled. This function returns immediately
   * if the event is null.
   * \sa IsFinished(), OpenCLEventList::WaitForFinished() */
  void WaitForFinished();

  /** Returns the device time in nanoseconds when the command was queued for
   * execution on the host. The return value is only valid if the command has
   * finished execution and profiling was enabled on the command queue.
   * \sa GetSubmitTime(), GetRunTime(), GetFinishTime(), IsQueued() */
  cl_ulong GetQueueTime() const;

  /** Returns the device time in nanoseconds when the command was submitted by
   * the host for execution on the device. The return value is only valid if
   * the command has finished execution and profiling was enabled on the command queue.
   * \sa GetQueueTime(), GetRunTime(), GetFinishTime(), IsSubmitted() */
  cl_ulong GetSubmitTime() const;

  /** Returns the device time in nanoseconds when the command started
   * running on the device. The return value is only valid if the command
   * has finished execution and profiling was enabled on the command queue.
   * \sa GetQueueTime(), GetSubmitTime(), GetFinishTime(), IsRunning() */
  cl_ulong GetRunTime() const;

  /** Returns the device time in nanoseconds when the command finished running
   * on the device. The return value is only valid if the command has finished
   * execution and profiling was enabled on the command queue.
   * \sa queueTime(), submitTime(), runTime(), isFinished() */
  cl_ulong GetFinishTime() const;

  /** Returns true if this OpenCL event is the same as other;
   * false otherwise.
   * \sa operator!=() */
  bool operator==(const OpenCLEvent & other) const;

  /** Returns true if this OpenCL event is not the same as  other;
   *false otherwise.
   * \sa operator==() */
  bool operator!=(const OpenCLEvent & other) const;

private:
  cl_event m_Id;
  friend class OpenCLUserEvent;
};

/** Stream out operator for OpenCLEvent */
template< class charT, class traits >
inline
std::basic_ostream< charT, traits > & operator<<(std::basic_ostream< charT, traits > & strm,
                                                 const OpenCLEvent & event)
{
  const cl_event id = event.GetEventId();

  if ( !id )
    {
    strm << "OpenCLEvent()";
    return strm;
    }

  const cl_command_type command = event.GetCommandType();
  const cl_int          status = event.GetStatus();

  // Get command name, check for clGetEventInfo() specification
  const char *commandName;
  switch ( command )
    {
    case CL_COMMAND_NDRANGE_KERNEL:
      commandName = "clEnqueueNDRangeKernel"; break;
    case CL_COMMAND_TASK:
      commandName = "clEnqueueTask"; break;
    case CL_COMMAND_NATIVE_KERNEL:
      commandName = "clEnqueueNativeKernel"; break;
    case CL_COMMAND_READ_BUFFER:
      commandName = "clEnqueueReadBuffer"; break;
    case CL_COMMAND_WRITE_BUFFER:
      commandName = "clEnqueueWriteBuffer"; break;
    case CL_COMMAND_COPY_BUFFER:
      commandName = "clEnqueueCopyBuffer"; break;
    case CL_COMMAND_READ_IMAGE:
      commandName = "clEnqueueReadImage"; break;
    case CL_COMMAND_WRITE_IMAGE:
      commandName = "clEnqueueWriteImage"; break;
    case CL_COMMAND_COPY_IMAGE:
      commandName = "clEnqueueCopyImage"; break;
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
      commandName = "clEnqueueCopyImageToBuffer"; break;
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
      commandName = "clEnqueueCopyBufferToImage"; break;
    case CL_COMMAND_MAP_BUFFER:
      commandName = "clEnqueueMapBuffer"; break;
    case CL_COMMAND_MAP_IMAGE:
      commandName = "clEnqueueMapImage"; break;
    case CL_COMMAND_UNMAP_MEM_OBJECT:
      commandName = "clEnqueueUnmapMemObject"; break;
    case CL_COMMAND_MARKER:
      commandName = "clEnqueueMarker"; break;
    case CL_COMMAND_ACQUIRE_GL_OBJECTS:
      commandName = "clEnqueueAcquireGLObjects"; break;
    case CL_COMMAND_RELEASE_GL_OBJECTS:
      commandName = "clEnqueueReleaseGLObjects"; break;
    // OpenCL 1.1 event types.
    case CL_COMMAND_READ_BUFFER_RECT:
      commandName = "clEnqueueReadBufferRect"; break;
    case CL_COMMAND_WRITE_BUFFER_RECT:
      commandName = "clEnqueueWriteBufferRect"; break;
    case CL_COMMAND_COPY_BUFFER_RECT:
      commandName = "clEnqueueCopyBufferRect"; break;
    case CL_COMMAND_USER:
      commandName = "clCreateUserEvent"; break;
    // OpenCL 1.2 event types.
    case CL_COMMAND_BARRIER:
      commandName = "clEnqueueBarrierWithWaitList"; break;
    case CL_COMMAND_MIGRATE_MEM_OBJECTS:
      commandName = "clEnqueueFillImage"; break;
    case CL_COMMAND_FILL_BUFFER:
      commandName = "clEnqueueFillBuffer"; break;
    case CL_COMMAND_FILL_IMAGE:
      commandName = "clEnqueueFillImage"; break;
    default:
      commandName = "Unknown"; break;
    }

  // Get command status
  const char *statusName;
  switch ( status )
    {
    case CL_COMPLETE:
      statusName = "finished"; break;
    case CL_RUNNING:
      statusName = "running"; break;
    case CL_SUBMITTED:
      statusName = "submitted"; break;
    case CL_QUEUED:
      statusName = "queued"; break;
    default:
      statusName = "Unknown"; break;
    }
  if ( status != CL_COMPLETE )
    {
    // Command is not complete : no profiling information available yet.
    strm << "OpenCLEvent(id:" << reinterpret_cast< long >( id )
         << " command:" << commandName
         << " status:" << statusName
         << ")";
    }
  else
    {
    cl_ulong queueTime, runTime, finishTime;
    if ( clGetEventProfilingInfo
           (id, CL_PROFILING_COMMAND_QUEUED,
           sizeof( queueTime ), &queueTime, 0) != CL_SUCCESS
         || clGetEventProfilingInfo
           (id, CL_PROFILING_COMMAND_START,
           sizeof( runTime ), &runTime, 0) != CL_SUCCESS
         || clGetEventProfilingInfo
           (id, CL_PROFILING_COMMAND_END,
           sizeof( finishTime ), &finishTime, 0) != CL_SUCCESS )
      {
      // Profiling information is not available, probably
      // because it was not enabled on the command queue.
      strm << "OpenCLEvent(id:" << reinterpret_cast< long >( id )
           << " command:" << commandName
           << " status:" << statusName
           << ")";
      }
    else
      {
      // Include profiling information in the debug output.
      const double fullDuration = ( finishTime - queueTime ) / 1000000.0f;
      const double runDuration = ( finishTime - runTime ) / 1000000.0f;
      strm << "OpenCLEvent(id:" << reinterpret_cast< long >( id )
           << " command:" << commandName
           << " status:" << statusName
           << " full-time:" << fullDuration
           << " ms running-time:" << runDuration
           << "ms)";
      }
    }

  return strm;
}

//------------------------------------------------------------------------------
/** \class OpenCLEventList
 * \brief OpenCLEventList class represents a list of OpenCLEvent objects.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * This implementation was taken from elastix (http://elastix.isi.uu.nl/).
 *
 * \note This work was funded by the Netherlands Organisation for
 *  Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup ITKGPUCommon
 */
class ITKOpenCL_EXPORT OpenCLEventList
{
public:
  typedef std::vector< cl_event > OpenCLEventListArrayType;

  /** Constructs an empty list of OpenCL events. */
  OpenCLEventList() {}

  /** Constructs a list of OpenCL events that contains event. If event is null,
   * this constructor will construct an empty list.
   * \sa Append() */
  OpenCLEventList(const OpenCLEvent & event);

  /** Constructs a copy of other.
   * \sa operator=()*/
  OpenCLEventList(const OpenCLEventList & other);

  /** Destroys this list of OpenCL events. */
  ~OpenCLEventList();

  /** Assigns the contents of other to this object. */
  OpenCLEventList & operator=(const OpenCLEventList & other);

  /** Returns true if this is an empty list; false otherwise.
   * \sa GetSize() */
  bool IsEmpty() const { return m_Events.empty(); }

  /** Returns the size of this event list.
   * \sa IsEmpty(), Get() */
  std::size_t GetSize() const { return m_Events.size(); }

  /** Appends event to this list of OpenCL events if it is not null.
   * Does nothing if  event is null.
   * \sa Remove() */
  void Append(const OpenCLEvent & event);

  /** Appends the contents of other to this event list. */
  void Append(const OpenCLEventList & other);

  /** Removes event from this event list.
   * \sa Append(), Contains() */
  void Remove(const OpenCLEvent & event);

  /** Returns the event at index in this event list, or a null OpenCLEvent
   * if index is out of range.
   * \sa GetSize(), Contains() */
  OpenCLEvent Get(const std::size_t index) const;

  /** Returns true if this event list contains event; false otherwise.
   * \sa Get(), Remove() */
  bool Contains(const OpenCLEvent & event) const;

  /** Returns a const pointer to the raw OpenCL event data in this event list;
   * null if the list is empty. This function is intended for use with native
   * OpenCL library functions that take an array of cl_event objects as
   * an argument.
   * \sa GetSize() */
  const cl_event * GetEventData() const;

  /** Returns a const reference to the array of OpenCL events.
   * \sa GetSize() */
  const OpenCLEventListArrayType & GetEventArray() const;

  /** Same as append event.
   * \sa Append() */
  OpenCLEventList & operator+=(const OpenCLEvent & event);

  /** Same as append event.
   * \sa Append() */
  OpenCLEventList & operator+=(const OpenCLEventList & other);

  /** Same as append event.
   * \sa Append() */
  OpenCLEventList & operator<<(const OpenCLEvent & event);

  /** Same as append event.
   * \sa Append() */
  OpenCLEventList & operator<<(const OpenCLEventList & other);

  /** Waits for all of the events in this list to be signaled as finished.
   * The calling thread is blocked until all of the events are signaled.
   * If the list is empty, then this function returns immediately.
   * \sa OpenCLEvent::WaitForFinished() */
  void WaitForFinished();

private:
  OpenCLEventListArrayType m_Events;
};

inline bool OpenCLEvent::operator==(const OpenCLEvent & other) const
{
  return m_Id == other.m_Id;
}

inline bool OpenCLEvent::operator!=(const OpenCLEvent & other) const
{
  return m_Id != other.m_Id;
}

inline const cl_event * OpenCLEventList::GetEventData() const
{
  return m_Events.empty() ? 0 : &m_Events[0];
}

inline const OpenCLEventList::OpenCLEventListArrayType & OpenCLEventList::GetEventArray() const
{
  return m_Events;
}

inline OpenCLEventList & OpenCLEventList::operator+=(const OpenCLEvent & event)
{
  Append(event);
  return *this;
}

inline OpenCLEventList & OpenCLEventList::operator+=(const OpenCLEventList & other)
{
  Append(other);
  return *this;
}

inline OpenCLEventList & OpenCLEventList::operator<<(const OpenCLEvent & event)
{
  Append(event);
  return *this;
}

inline OpenCLEventList & OpenCLEventList::operator<<(const OpenCLEventList & other)
{
  Append(other);
  return *this;
}

/** Stream out operator for OpenCLEventList */
template< class charT, class traits >
inline
std::basic_ostream< charT, traits > & operator<<(std::basic_ostream< charT, traits > & strm,
                                                 const OpenCLEventList & eventlist)
{
  if ( !eventlist.GetSize() )
    {
    strm << "OpenCLEventList()";
    return strm;
    }

  OpenCLEventList::OpenCLEventListArrayType::const_iterator it;
  const OpenCLEventList::OpenCLEventListArrayType &eventsArray = eventlist.GetEventArray();

  std::size_t id = 0;
  strm << "OpenCLEventList contains:" << std::endl;
  for ( it = eventsArray.begin(); it < eventsArray.end(); it++ )
    {
    // const OpenCLEvent &event = *it;
    // strm << "array id: " << id << " " << event << std::endl;

    // Let's print only address, printing the OpenCLEvent
    // could halt execution of the program
    strm << "array id: " << id << " " << *it << std::endl;
    ++id;
    }

  return strm;
}
} // end namespace itk

#endif /* __itkOpenCLEvent_h */

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
#ifndef __itkOpenCLEvent_h
#define __itkOpenCLEvent_h

#include "itkOpenCL.h"
#include <vector>

namespace itk
{
/** \class OpenCLEvent
 * \brief OpenCLEvent class represents an OpenCL event object.
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

//------------------------------------------------------------------------------
/** \class OpenCLEventList
 * \brief OpenCLEventList class represents a list of OpenCLEvent objects.
 *
 * \ingroup ITKGPUCommon
 */
class ITKOpenCL_EXPORT OpenCLEventList
{
public:
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
   * \sa IsEmpty(), At() */
  size_t GetSize() const { return m_Events.size(); }

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
  OpenCLEvent At(size_t index) const;

  /** Returns true if this event list contains event; false otherwise.
   * \sa At(), Remove() */
  bool Contains(const OpenCLEvent & event) const;

  /** Returns a const pointer to the raw OpenCL event data in this event list;
   * null if the list is empty. This function is intended for use with native
   * OpenCL library functions that take an array of cl_event objects as
   * an argument.
   * \sa GetSize() */
  const cl_event * GetEventData() const;

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
  std::vector< cl_event > m_Events;
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
} // end namespace itk

#endif /* __itkOpenCLEvent_h */

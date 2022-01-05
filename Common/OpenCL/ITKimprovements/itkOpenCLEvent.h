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
#ifndef itkOpenCLEvent_h
#define itkOpenCLEvent_h

#include "itkOpenCL.h"
#include "itkOpenCLExtension.h"
#include <ostream>

namespace itk
{
/** \class OpenCLEvent
 * \brief OpenCLEvent class represents an OpenCL event object.
 *
 * An event object can be used to track the execution status of a command.
 * The API calls that enqueue commands to a command queue create a new event
 * object that is returned in the event argument. In case of an error enqueuing
 * the command in the command queue the event argument does not
 * return an event object.
 *
 * \table
 * \row \o IsQueued() \o The command command has been enqueued in the command-queue.
 * \row \o IsSubmitted() \o The enqueued command has been submitted by the host
 * to the device associated with the command-queue.
 * \row \o IsRunning() \o The command is currently executing on the OpenCL device.
 * \row \o IsComplete() \o The command has successfully completed.
 * \row \o IsError() \o The command has finished execution due to an error.
 * \endtable
 *
 * The method WaitForFinished() waits on the host thread for commands identified
 * by event objects in event list to complete. A command is considered complete
 * if its execution status is \c{CL_COMPLETE} or a negative value.
 *
 * \code
 * OpenCLBuffer buffer = context->CreateBufferDevice(size, OpenCLMemoryObject::WriteOnly);
 * OpenCLEvent event = buffer.ReadAsync(offset, data, size);
 * ...
 * event.WaitForFinished();
 * \endcode
 *
 * The events specified in OpenCLEventList act as synchronization points.
 * The OpenCLEventList are used to control execution order:
 *
 * \code
 * OpenCLBuffer buffer = context->CreateBufferDevice(size, OpenCLMemoryObject::WriteOnly);
 * OpenCLEvent event1 = buffer.ReadAsync(offset1, data1, size1);
 * OpenCLEvent event2 = buffer.ReadAsync(offset2, data2, size2);
 *
 * OpenCLEventList events;
 * events << event1 << event2;
 * OpenCLEvent event3 = buffer.ReadAsync(offset3, data3, size3, events);
 * ...
 * event3.WaitForFinished();
 * \endcode
 *
 * The OpenCL functions that are submitted to a command-queue are enqueued
 * in the order the calls are made but can be configured to execute in-order
 * or out-of-order. The properties argument in context->CreateCommandQueue()
 * can be used to specify the execution order.
 *
 * If the \c{CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE} property of a command-queue
 * is not set, the commands enqueued to a command-queue execute in order.
 * For example, if an application calls buffer.ReadAsync() to execute event1
 * followed by a buffer.ReadAsync() to execute event2, the application can
 * assume that event1 finishes first and then event2 is executed.
 * If the OpenCLCommandQueue::IsOutOfOrder() property of a command-queue
 * is set, then there is no guarantee that event1 will finish before event2
 * starts execution.
 *
 * \ingroup OpenCL
 * \sa OpenCLCommandQueue::IsOutOfOrder(), OpenCLEventList, OpenCLUserEvent
 */

// Forward declaration
class OpenCLUserEvent;

class ITKOpenCL_EXPORT OpenCLEvent
{
public:
  /** Standard class typedefs. */
  using Self = OpenCLEvent;

  /** Constructs a null OpenCL event object. */
  OpenCLEvent();

  /** Constructs an OpenCL event object from the identifier id.
   * This class takes over ownership of id and will release it in
   * the destructor. */
  OpenCLEvent(const cl_event id);

  /** Constructs a copy of other. The \c{clRetainEvent()} function
   * will be called to update the reference count on GetEventId(). */
  OpenCLEvent(const OpenCLEvent & other);

  /** Releases this OpenCL event object by calling \c{clReleaseEvent()}. */
  ~OpenCLEvent();

  /** Assigns other to this OpenCL event object. The current GetEventId() will
   * be released with \c{clReleaseEvent()}, and the new GetEventId() will be
   * retained with \c{clRetainEvent()}. */
  OpenCLEvent &
  operator=(const OpenCLEvent & other);

  /** Returns true if this OpenCL event object is null, false otherwise. */
  bool
  IsNull() const
  {
    return this->m_Id == 0;
  }

  /** Returns the OpenCL identifier for this event. */
  cl_event
  GetEventId() const
  {
    return this->m_Id;
  }

  /** Returns true if the command associated with this OpenCL event has been
   * queued for execution on the host, but has not yet been submitted to
   * the device yet.
   * \sa IsSubmitted(), IsRunning(), IsComplete(), IsError(), GetStatus() */
  bool
  IsQueued() const
  {
    return GetStatus() == CL_QUEUED;
  }

  /** Returns true if the command associated with this OpenCL event has been
   * submitted for execution on the device yet, but is not yet running.
   * \sa IsQueued(), IsRunning(), IsComplete(), IsError(), GetStatus() */
  bool
  IsSubmitted() const
  {
    return GetStatus() == CL_SUBMITTED;
  }

  /** Returns true if the command associated with this OpenCL event is
   * running on the device, but has not yet finished.
   * \sa IsQueued(), IsSubmitted(), IsComplete(), IsError(), GetStatus() */
  bool
  IsRunning() const
  {
    return GetStatus() == CL_RUNNING;
  }

  /** Returns true if the command associated with this OpenCL event
   * has completed execution on the device.
   * \sa IsQueued(), IsSubmitted(), IsRunning(), IsError(), GetStatus() */
  bool
  IsComplete() const
  {
    return GetStatus() == CL_COMPLETE;
  }

  /** Returns true if an error has occurred on this OpenCL event.
   * \sa IsQueued(), IsSubmitted(), IsRunning(), IsComplete(), GetStatus() */
  bool
  IsError() const
  {
    return GetStatus() < 0;
  }

  /** Returns the status of this event, which is an error code or one
   * of \c{CL_QUEUED}, \c{CL_SUBMITTED}, \c{CL_RUNNING} or \c{CL_COMPLETE}.
   * \sa IsQueued(), IsSubmitted(), IsRunning(), IsComplete(), IsError() */
  cl_int
  GetStatus() const;

  /** Returns the type of command that generated this event. */
  cl_command_type
  GetCommandType() const;

  /** Waits for this event to be signaled as finished. The calling thread
   * is blocked until the event is signaled. This function returns immediately
   * if the event is null.
   * \sa IsComplete(), OpenCLEventList::WaitForFinished() */
  cl_int
  WaitForFinished();

  /** Registers a user callback function for a specific command execution status.
   * If the execution of a command is terminated, the command-queue associated
   * with this terminated command, and the associated context (and all other
   * command-queues in this context) may no longer be available. The behavior of
   * OpenCL API calls that use this context (and command-queues associated with
   * this context) are now considered to be implementation defined. The user
   * registered callback function specified when context is created can be used
   * to report appropriate error information. */
  cl_int
  SetCallback(cl_int type, void(CL_CALLBACK * pfn_notify)(cl_event, cl_int, void *), void * user_data = nullptr);

  /** Returns the device time in nanoseconds when the command was queued for
   * execution on the host. The return value is only valid if the command has
   * finished execution and profiling was enabled on the command queue.
   * \sa GetSubmitTime(), GetRunTime(), GetFinishTime(), IsQueued() */
  cl_ulong
  GetQueueTime() const;

  /** Returns the device time in nanoseconds when the command was submitted by
   * the host for execution on the device. The return value is only valid if
   * the command has finished execution and profiling was enabled on the command queue.
   * \sa GetQueueTime(), GetRunTime(), GetFinishTime(), IsSubmitted() */
  cl_ulong
  GetSubmitTime() const;

  /** Returns the device time in nanoseconds when the command started
   * running on the device. The return value is only valid if the command
   * has finished execution and profiling was enabled on the command queue.
   * \sa GetQueueTime(), GetSubmitTime(), GetFinishTime(), IsRunning() */
  cl_ulong
  GetRunTime() const;

  /** Returns the device time in nanoseconds when the command finished running
   * on the device. The return value is only valid if the command has finished
   * execution and profiling was enabled on the command queue.
   * \sa GetQueueTime(), GetSubmitTime(), GetRunTime(), IsComplete() */
  cl_ulong
  GetFinishTime() const;

private:
  cl_event m_Id;

  /** friends from OpenCL core */
  friend class OpenCLUserEvent;
};

/** Operator ==
 * Returns true if \a lhs OpenCL event is the same as \a rhs, false otherwise.
 * \sa operator!= */
bool ITKOpenCL_EXPORT
     operator==(const OpenCLEvent & lhs, const OpenCLEvent & rhs);

/** Operator !=
 * Returns true if \a lhs OpenCL event is not the same as \a rhs, false otherwise.
 * \sa operator== */
bool ITKOpenCL_EXPORT
     operator!=(const OpenCLEvent & lhs, const OpenCLEvent & rhs);

/** Stream out operator for OpenCLEvent */
template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const OpenCLEvent & event)
{
  const cl_event id = event.GetEventId();

  if (!id)
  {
    strm << "OpenCLEvent()";
    return strm;
  }

  const cl_command_type command = event.GetCommandType();
  const cl_int          status = event.GetStatus();

  // Get command name, check for clGetEventInfo() specification
  const char * commandName;
  switch (command)
  {
    case CL_COMMAND_NDRANGE_KERNEL:
      commandName = "clEnqueueNDRangeKernel";
      break;
    case CL_COMMAND_TASK:
      commandName = "clEnqueueTask";
      break;
    case CL_COMMAND_NATIVE_KERNEL:
      commandName = "clEnqueueNativeKernel";
      break;
    case CL_COMMAND_READ_BUFFER:
      commandName = "clEnqueueReadBuffer";
      break;
    case CL_COMMAND_WRITE_BUFFER:
      commandName = "clEnqueueWriteBuffer";
      break;
    case CL_COMMAND_COPY_BUFFER:
      commandName = "clEnqueueCopyBuffer";
      break;
    case CL_COMMAND_READ_IMAGE:
      commandName = "clEnqueueReadImage";
      break;
    case CL_COMMAND_WRITE_IMAGE:
      commandName = "clEnqueueWriteImage";
      break;
    case CL_COMMAND_COPY_IMAGE:
      commandName = "clEnqueueCopyImage";
      break;
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
      commandName = "clEnqueueCopyImageToBuffer";
      break;
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
      commandName = "clEnqueueCopyBufferToImage";
      break;
    case CL_COMMAND_MAP_BUFFER:
      commandName = "clEnqueueMapBuffer";
      break;
    case CL_COMMAND_MAP_IMAGE:
      commandName = "clEnqueueMapImage";
      break;
    case CL_COMMAND_UNMAP_MEM_OBJECT:
      commandName = "clEnqueueUnmapMemObject";
      break;
    case CL_COMMAND_MARKER:
      commandName = "clEnqueueMarker";
      break;
    case CL_COMMAND_ACQUIRE_GL_OBJECTS:
      commandName = "clEnqueueAcquireGLObjects";
      break;
    case CL_COMMAND_RELEASE_GL_OBJECTS:
      commandName = "clEnqueueReleaseGLObjects";
      break;
    // OpenCL 1.1 event types.
    case CL_COMMAND_READ_BUFFER_RECT:
      commandName = "clEnqueueReadBufferRect";
      break;
    case CL_COMMAND_WRITE_BUFFER_RECT:
      commandName = "clEnqueueWriteBufferRect";
      break;
    case CL_COMMAND_COPY_BUFFER_RECT:
      commandName = "clEnqueueCopyBufferRect";
      break;
    case CL_COMMAND_USER:
      commandName = "clCreateUserEvent";
      break;
    // OpenCL 1.2 event types.
    case CL_COMMAND_BARRIER:
      commandName = "clEnqueueBarrierWithWaitList";
      break;
    case CL_COMMAND_MIGRATE_MEM_OBJECTS:
      commandName = "clEnqueueFillImage";
      break;
    case CL_COMMAND_FILL_BUFFER:
      commandName = "clEnqueueFillBuffer";
      break;
    case CL_COMMAND_FILL_IMAGE:
      commandName = "clEnqueueFillImage";
      break;
    default:
      commandName = "Unknown";
      break;
  }

  // Get command status
  const char * statusName;
  switch (status)
  {
    case CL_COMPLETE:
      statusName = "completed";
      break;
    case CL_RUNNING:
      statusName = "running";
      break;
    case CL_SUBMITTED:
      statusName = "submitted";
      break;
    case CL_QUEUED:
      statusName = "queued";
      break;
    default:
      statusName = "Unknown";
      break;
  }
  if (status != CL_COMPLETE)
  {
    // Command is not complete : no profiling information available yet.
    strm << "OpenCLEvent(id:" << reinterpret_cast<long>(id) << " command:" << commandName << " status:" << statusName
         << ")";
  }
  else
  {
    cl_ulong queueTime, runTime, finishTime;
    if (clGetEventProfilingInfo(id, CL_PROFILING_COMMAND_QUEUED, sizeof(queueTime), &queueTime, 0) != CL_SUCCESS ||
        clGetEventProfilingInfo(id, CL_PROFILING_COMMAND_START, sizeof(runTime), &runTime, 0) != CL_SUCCESS ||
        clGetEventProfilingInfo(id, CL_PROFILING_COMMAND_END, sizeof(finishTime), &finishTime, 0) != CL_SUCCESS)
    {
      // Profiling information is not available, probably
      // because it was not enabled on the command queue.
      strm << "OpenCLEvent(id:" << reinterpret_cast<long>(id) << " command:" << commandName << " status:" << statusName
           << ")";
    }
    else
    {
      // Include profiling information in the debug output.
      const double fullDuration = (finishTime - queueTime) / 1000000.0f;
      const double runDuration = (finishTime - runTime) / 1000000.0f;
      strm << "OpenCLEvent(id:" << reinterpret_cast<long>(id) << " command:" << commandName << " status:" << statusName
           << " full-time:" << fullDuration << " ms running-time:" << runDuration << "ms)";
    }
  }

  return strm;
}


} // end namespace itk

#endif /* itkOpenCLEvent_h */

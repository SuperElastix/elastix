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
#ifndef itkOpenCLUserEvent_h
#define itkOpenCLUserEvent_h

#include "itkOpenCLEvent.h"

namespace itk
{
/** \class OpenCLUserEvent
 * \brief The OpenCLUserEvent class represents OpenCL user events.
 *
 * User events allow applications to enqueue commands that wait on a
 * user event to finish before the command is executed by the device.
 * Commands that depend upon the user event will not be executed until the
 * application triggers the user event with SetComplete().
 * User events are constructed with OpenCLContext::CreateUserEvent().
 *
 * \code
 * OpenCLBuffer buffer = context->CreateBufferDevice(size, OpenCLMemoryObject::WriteOnly);
 * OpenCLUserEvent user_event = context->CreateUserEvent();
 * OpenCLEventList events;
 * events << user_event;
 * OpenCLEvent event1 = buffer.WriteAsync(..., events, ...);
 * OpenCLEvent event2 = buffer.WriteAsync(..., events, ...);
 * events << event1 << event2;
 * user_event.SetComplete();
 * \endcode
 *
 * \note Developers should be careful when releasing their last reference count
 * on events created by OpenCLContext::CreateUserEvent() that have not yet been
 * set to status of \c{CL_COMPLETE} or an error. If the user event was used in the
 * OpenCLEventList argument passed to a clEnqueue*** API or another application
 * host thread is waiting for it in OpenCLEvent::WaitForFinished(),
 * those commands and host threads will continue to wait for the event status
 * to reach \c{CL_COMPLETE} or error, even after the user has released the object.
 * Since in this scenario the developer has released his last reference count
 * to the user event, it would be in principle no longer valid for him to change
 * the status of the event to unblock all the other machinery. As a result the
 * waiting tasks will wait forever, and associated events, cl_mem objects,
 * command queues and contexts are likely to leak. In-order command queues
 * caught up in this deadlock may cease to do any work.
 *
 * \ingroup OpenCL
 * \sa OpenCLEvent, OpenCLContext::CreateUserEvent()
 */

// Forward declaration
class OpenCLContext;

class ITKOpenCL_EXPORT OpenCLUserEvent : public OpenCLEvent
{
public:
  /** Standard class typedefs. */
  using Self = OpenCLUserEvent;

  /** Constructs a null user event. */
  OpenCLUserEvent()
    : OpenCLEvent()
  {}

  /** Constructs an OpenCL event object from the native identifier \a id.
   * This class takes over ownership of \a id and will release it in
   * the destructor.
   * If \a id is not a user event, then the newly constructed event
   * will be set to null, and \a id will be released. */
  OpenCLUserEvent(cl_event id);

  /** Constructs a copy of \a other. The \c{clRetainEvent()} function
   * will be called to update the reference count on GetEventId().
   * If \a other is not a user event, then the newly constructed event
   * will be set to null. */
  OpenCLUserEvent(const OpenCLEvent & other);

  /** Assigns \a other to this OpenCL event object. The current GetEventId() will
   * be released with \c{clReleaseEvent()}, and the new GetEventId() will be
   * retained with \c{clRetainEvent()}.
   * If \a other is not a user event, then this event will be
   * set to null. */
  OpenCLUserEvent &
  operator=(const OpenCLEvent & other);

  /** Sets this user event to the complete state. Any queued commands that
   * depend upon this event can now proceed.
   * \sa SetStatus() */
  void
  SetComplete();

  /** Sets the \a status of this user event. The \a status should
   * be either \c{CL_COMPLETE} or a negative OpenCL error code.
   * \sa SetComplete() */
  void
  SetStatus(const cl_int status);

private:
  /** Checks the event type agains \c{CL_COMMAND_USER} and release it if not. */
  void
  ReleaseIfNotUserEvent();

  // Used by OpenCLContext::CreateUserEvent() to avoid ReleaseIfNotUserEvent().
  OpenCLUserEvent(cl_event id, bool)
    : OpenCLEvent(id)
  {}

  /** friends from OpenCL core */
  friend class OpenCLContext;
};

} // end namespace itk

#endif /* itkOpenCLUserEvent_h */

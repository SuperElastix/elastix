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
#ifndef itkOpenCLCommandQueue_h
#define itkOpenCLCommandQueue_h

#include "itkOpenCL.h"
#include <ostream>

namespace itk
{
/** \class OpenCLCommandQueue
 * \brief The OpenCLCommandQueue class represents an OpenCL a
 * command-queue on a specific device and valid OpenCLContext.
 *
 * OpenCL objects such as OpenCLBuffer, OpenCLProgram and OpenCLKernel objects
 * are created using a OpenCLContext. Operations on these objects are performed
 * using a command-queue. The command-queue can be used to queue a set of
 * operations (referred to as commands) in order. Having multiple command-queues
 * allows applications to queue multiple independent commands without requiring
 * synchronization. Note that this should work as long as these objects are
 * not being shared. Sharing of objects across multiple command-queues will
 * require the application to perform appropriate synchronization. Commands
 * are added to the command-queue by calling methods on OpenCLContext,
 * OpenCLBuffer, OpenCLImage, OpenCLKernel. These methods use
 * OpenCLContext::GetCommandQueue() as the command destination.
 * OpenCLContext::SetCommandQueue() can be used to alter the
 * destination queue.
 *
 * \ingroup OpenCL
 * \sa OpenCLContext, OpenCLBuffer, OpenCLImage, OpenCLKernel
 */

// Forward declaration
class OpenCLContext;

class ITKOpenCL_EXPORT OpenCLCommandQueue
{
public:
  /** Standard class typedefs. */
  using Self = OpenCLCommandQueue;

  /** Constructs a null OpenCL command queue object. */
  OpenCLCommandQueue()
    : m_Context(0)
    , m_Id(0)
  {}

  /** Constructs an OpenCL command queue object based on the supplied
   * native OpenCL \a id, and associates it with \a context. This class
   * will take over ownership of \a id and release it in the destructor. */
  OpenCLCommandQueue(OpenCLContext * context, cl_command_queue id)
    : m_Context(context)
    , m_Id(id)
  {}

  /** Constructs a copy of \a other. */
  OpenCLCommandQueue(const OpenCLCommandQueue & other);

  /** Releases this OpenCL command queue. If this object is the last reference,
   * the queue will be destroyed. */
  ~OpenCLCommandQueue();

  /** Assigns \a other to this object. */
  OpenCLCommandQueue &
  operator=(const OpenCLCommandQueue & other);

  /** Returns true if this OpenCL command queue is null. */
  bool
  IsNull() const
  {
    return this->m_Id == 0;
  }

  /** Returns true if this command queue executes commands out of order,
   * otherwise false if commands are executed in order. */
  bool
  IsOutOfOrder() const;

  /** Returns true if this command queue will perform profiling on
   * commands; false otherwise.
   * Profiling information is made available when a OpenCLEvent finishes execution.
   * \sa OpenCLEvent::GetFinishTime() */
  bool
  IsProfilingEnabled() const;

  /** Returns the native OpenCL command queue identifier for this object. */
  cl_command_queue
  GetQueueId() const
  {
    return this->m_Id;
  }

  /** Returns the OpenCL context that created this queue object. */
  OpenCLContext *
  GetContext() const
  {
    return this->m_Context;
  }

private:
  OpenCLContext *  m_Context;
  cl_command_queue m_Id;
};

/** Operator ==
 * Returns true if \a lhs OpenCL command queue is the same as \a rhs, false otherwise.
 * \sa operator!= */
bool ITKOpenCL_EXPORT
     operator==(const OpenCLCommandQueue & lhs, const OpenCLCommandQueue & rhs);

/** Operator !=
 * Returns true if \a lhs OpenCL command queue is not the same as \a rhs, false otherwise.
 * \sa operator== */
bool ITKOpenCL_EXPORT
     operator!=(const OpenCLCommandQueue & lhs, const OpenCLCommandQueue & rhs);

/** Stream out operator for OpenCLCommandQueue */
template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const OpenCLCommandQueue & queue)
{
  if (queue.IsNull())
  {
    strm << "OpenCLCommandQueue(null)";
    return strm;
  }

  const char indent = ' ';

  strm << "OpenCLCommandQueue\n"
       << indent << "Id: " << queue.GetQueueId() << '\n'
       << indent << "Out of order: " << (queue.IsOutOfOrder() ? "Yes" : "No") << '\n'
       << indent << "Profiling enabled: " << (queue.IsProfilingEnabled() ? "Yes" : "No") << std::endl;

  return strm;
}


} // end namespace itk

#endif /* itkOpenCLCommandQueue_h */

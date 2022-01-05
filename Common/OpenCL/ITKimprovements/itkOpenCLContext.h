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
#ifndef itkOpenCLContext_h
#define itkOpenCLContext_h

#include "itkLightObject.h"
#include "itkObjectFactory.h"

#include "itkOpenCLDevice.h"
#include "itkOpenCLCommandQueue.h"
#include "itkOpenCLBuffer.h"
#include "itkOpenCLVector.h"
#include "itkOpenCLImage.h"
#include "itkOpenCLSampler.h"
#include "itkOpenCLProgram.h"
#include "itkOpenCLUserEvent.h"

namespace itk
{
/** \class OpenCLContext
 * \brief The OpenCLContext class represents an OpenCL context.
 *
 * The host defines a context for the execution of the kernels.
 * The context includes the following resources:
 *
 * \table
 * \row \o Devices \o The collection of OpenCL devices to be used by the host.
 * \row \o Kernels \o The OpenCL functions that run on OpenCL devices.
 * \row \o Program Objects \o The program source and executable that
 *         implement the kernels.
 * \row \o Memory Objects \o A set of memory objects visible to the host and
 *         the OpenCL devices. Memory objects contain values that can be
 *         operated on by instances of a kernel.
 * \endtable
 *
 * The context is created and manipulated by the host using functions from
 * the OpenCL API. The host creates a data structure called a command-queue to
 * coordinate execution of the kernels on the devices. The host places commands
 * into the command-queue which are then scheduled onto the devices within the context.
 *
 * Context are constructed using OpenCLContext::Create().
 * \code
 * OpenCLContext::Pointer context = OpenCLContext::GetInstance();
 * context->Create( itk::OpenCLContext::SingleMaximumFlopsDevice );
 * if( !context->IsCreated() )
 * {
 *   std::cerr << "OpenCL-enabled device is not present." << std::endl;
 * }
 * \endcode
 *
 * \ingroup OpenCL
 * \sa OpenCLDevice, OpenCLProgram, OpenCLKernel, OpenCLCommandQueue,
 * \sa OpenCLBuffer, OpenCLVector, OpenCLUserEvent
 */

// Forward declaration
class OpenCLKernel;
class OpenCLVectorBase;
class OpenCLContextPimpl; // OpenCLContext private implementation idiom.

class ITKOpenCL_EXPORT OpenCLContext : public LightObject
{
public:
  /** Standard class typedefs. */
  using Self = OpenCLContext;
  using Superclass = LightObject;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(OpenCLContext, LightObject);

  /** This is a singleton pattern New. There will only be ONE
   * reference to a OpenCLContext object per process. Clients that
   * call this must call Delete on the object so that the reference
   * counting will work. The single instance will be unreferenced when
   * the program exits. */
  static Pointer
  New();

  /** Return the singleton instance with no reference counting. */
  static Pointer
  GetInstance();

  /** Supply a user defined OpenCL context. Call ->Delete() on the supplied
   * instance after setting it. */
  static void
  SetInstance(OpenCLContext * instance);

  /** Returns true if the underlying OpenCL GetContextId() has been
   * created, false otherwise.
   * \sa Create(), SetContextId() */
  bool
  IsCreated() const;

  /** \enum OpenCLContext::CreateMethod
   * This enum defines the OpenCL context create method.
   * \value Default.
   * \value DevelopmentSingleMaximumFlopsDevice.
   * \value DevelopmentMultipleMaximumFlopsDevices.
   * \value SingleMaximumFlopsDevice.
   * \value MultipleMaximumFlopsDevices. */
  enum CreateMethod
  {
    Default = 0x0000,
    DevelopmentSingleMaximumFlopsDevice = 0x0001,
    DevelopmentMultipleMaximumFlopsDevices = 0x0002,
    SingleMaximumFlopsDevice = 0x0004,
    MultipleMaximumFlopsDevices = 0x0008
  };

  /** Creates a new OpenCL context that matches \a type.
   * Does nothing if the context has already been created.
   * This function will search for the first platform that has a device
   * that matches \a type. The following code can be used to select
   * devices that match \a type on a specific platform:
   * \code
   * context.Create(OpenCLDevice::GetDevices(type, platform));
   * \endcode
   * Returns true if the context was created, false otherwise.
   * On error, the status can be retrieved by calling GetLastError().
   * \sa IsCreated(), SetContextId(), Release() */
  bool
  Create(const OpenCLDevice::DeviceType type);

  /** Creates a new OpenCL context that matches \a devices.
   * Does nothing if the context has already been created.
   * All of the \a devices must be associated with the same platform.
   * Returns true if the context was created, false otherwise.
   * On error, the status can be retrieved by calling GetLastError().
   * \sa IsCreated(), SetContextId(), Release() */
  bool
  Create(const std::list<OpenCLDevice> & devices);

  /** Creates a new OpenCL context that matches \a method of creating.
   * Does nothing if the context has already been created.
   * Returns true if the context was created, false otherwise.
   * On error, the status can be retrieved by calling GetLastError().
   * \sa IsCreated(), SetContextId(), Release() */
  bool
  Create(const OpenCLContext::CreateMethod method);

  /** Creates a new OpenCL context that matches \a platform and \a type.
   * Does nothing if the context has already been created.
   * Returns true if the context was created, false otherwise.
   * On error, the status can be retrieved by calling GetLastError().
   * \sa IsCreated(), SetContextId(), Release() */
  bool
  Create(const OpenCLPlatform & platfrom, const OpenCLDevice::DeviceType type = OpenCLDevice::Default);

  /** Creates a new OpenCL context that is defined by CMake.
   * See CMake OPENCL_USE_* variables.
   * Does nothing if the context has already been created.
   * Returns true if the context was created, false otherwise.
   * On error, the status can be retrieved by calling GetLastError().
   * \note This is most simple method to create OpenCL context which uses
   * \c{clCreateContextFromType} and target existing OpenCL platform,
   * which may not be found on the user computer. For production better use
   * context::Create( OpenCLContext::SingleMaximumFlopsDevice ) or similar.
   * \sa IsCreated(), SetContextId(), Release() */
  bool
  Create();

  /** Releases this context, destroying it if the reference count is zero.
   * Does nothing if the context has not been created or is already released.
   * \sa Create() */
  virtual void
  Release();

  /** Returns the native OpenCL context identifier associated with this object.
   * \sa SetContextId() */
  cl_context
  GetContextId() const;

  /** Sets the native OpenCL context identifier associated with this
   * object to \a id.
   * This function will call \c{clRetainContext()} to increase the
   * reference count on \a id. If the identifier was previously set
   * to something else, then \c{clReleaseContext()} will be called
   * on the previous value.
   * \sa GetContextId(), Create() */
  void
  SetContextId(cl_context id);

  /** Returns the list of devices that are in use by this context.
   * If the context has not been created, returns an empty list.
   * \sa GetDefaultDevice() */
  std::list<OpenCLDevice>
  GetDevices() const;

  /** Returns the default device in use by this context, which is typically
   * the first element of the GetDevices() list or a null OpenCLDevice if the
   * context has not been created yet.
   * \sa GetDevices() */
  OpenCLDevice
  GetDefaultDevice() const;

  /** Returns the last OpenCL error that occurred while executing an
   * operation on this context or any of the objects created by
   * the context. Returns \c{CL_SUCCESS} if the last operation succeeded.
   * \sa SetLastError(), GetErrorName() */
  cl_int
  GetLastError() const;

  /** Sets the last error code to \a error.
   * \sa GetLastError(), GetErrorName() */
  void
  SetLastError(const cl_int error);

  /** Returns the name of the supplied OpenCL error \a code. For example,
   * \c{CL_SUCCESS}, \c{CL_INVALID_CONTEXT}, etc.
   * \sa GetLastError() */
  static std::string
  GetErrorName(const cl_int code);

  /** Report the error based on OpenCL error \a code with exception object. */
  void
  ReportError(const cl_int code, const char * fileName = "", const int lineNumber = 0, const char * location = "");

  /** Returns the context's active command queue, which will be
   * GetDefaultCommandQueue() if the queue has not yet been set.
   * \sa SetCommandQueue(), GetDefaultCommandQueue() */
  OpenCLCommandQueue
  GetCommandQueue();

  /** Sets the context's active command \a queue. If \a queue is
   * null, then GetDefaultCommandQueue() will be used.
   * \sa GetCommandQueue(), GetDefaultCommandQueue() */
  void
  SetCommandQueue(const OpenCLCommandQueue & queue);

  /** Returns the default command queue for GetDefaultDevice(). If the queue
   * has not been created, it will be created with the default properties
   * of in-order execution of commands, and profiling disabled.
   * Use CreateCommandQueue() to create a queue that supports
   * out-of-order execution or profiling. For example:
   * \code
   * OpenCLCommandQueue queue = context.CreateCommandQueue(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
   * context.SetCommandQueue(queue);
   * \endcode
   * \sa GetCommandQueue(), CreateCommandQueue(), GetLastError() */
  OpenCLCommandQueue
  GetDefaultCommandQueue();

  /** Creates a new command queue on this context for \a device with
   * the specified \a properties. If \a device is null, then
   * GetDefaultDevice() will be used instead.
   * Unlike GetDefaultCommandQueue(), this function will create a new queue
   * every time it is called. The queue will be deleted when the last
   * reference to the returned object is removed.
   * \sa GetDefaultCommandQueue(), GetLastError() */
  OpenCLCommandQueue
  CreateCommandQueue(const cl_command_queue_properties properties, const OpenCLDevice & device = OpenCLDevice());

  /** Creates an OpenCL memory buffer of \a size bytes in length,
   * with the specified \a access mode.
   * The memory is created on the device and will not be accessible
   * to the host via a direct pointer. Use CreateBufferHost() to
   * create a host-accessible buffer.
   * Returns the new OpenCL memory buffer object, or a null object
   * if the buffer could not be created.
   * \sa CreateBufferHost(), CreateBufferCopy(), CreateVector() */
  OpenCLBuffer
  CreateBufferDevice(const OpenCLMemoryObject::Access access, const std::size_t size);

  /** Creates an OpenCL memory buffer of \a size bytes in length,
   * with the specified \a access mode.
   * If \a data is not null, then it will be used as the storage
   * for the buffer. If \a data is null, then a new block of
   * host-accessible memory will be allocated.
   * Returns the new OpenCL memory buffer object, or a null object
   * if the buffer could not be created.
   * \sa CreateBufferDevice(), CreateBufferCopy(), CreateVector() */
  OpenCLBuffer
  CreateBufferHost(void * data, const OpenCLMemoryObject::Access access, const std::size_t size);

  /** Creates an OpenCL memory buffer of \a size bytes in length,
   * with the specified \a access mode.
   * The buffer is initialized with a copy of the contents of \a data.
   * The application's \a data can be discarded after the buffer is created.
   * Returns the new OpenCL memory buffer object, or a null object
   * if the buffer could not be created.
   * \sa CreateBufferDevice(), CreateBufferHost(), CreateVector() */
  OpenCLBuffer
  CreateBufferCopy(const void * data, const OpenCLMemoryObject::Access access, const std::size_t size);

  /** Creates a host-accessible vector of \a size elements of type T
   * on this context and returns it. The elements will be initially in
   * an undefined state.
   * Note that the \a access mode indicates how the OpenCL device (e.g. GPU)
   * will access the vector. When the host maps the vector, it will always
   * be mapped as ReadWrite.
   * \sa CreateBufferHost() */
  template <typename T>
  OpenCLVector<T>
  CreateVector(const OpenCLMemoryObject::Access access, const std::size_t size)
  {
    return OpenCLVector<T>(this, access, size);
  }


  /** Creates a OpenCL image object with the specified \a format,
   * \a size, and \a access mode.
   * The image memory is created on the device and will not be accessible
   * to the host via a direct pointer. Use CreateImageHost() to create a
   * host-accessible image. Returns the new OpenCL image object,
   * or a null object if the image could not be created.
   * \sa CreateImageHost(), CreateImageCopy() */
  OpenCLImage
  CreateImageDevice(const OpenCLImageFormat & format, const OpenCLMemoryObject::Access access, const OpenCLSize & size);

  /** Creates a OpenCL image object with the specified \a format,
   * \a size, and \a access mode. If \a data is not null, then it will be used
   * as the storage for the image. If \a data is null, then a new block of
   * host-accessible memory will be allocated. Returns the new OpenCL image
   * object, or a null object if the image could not be created.
   * \sa CreateImageDevice(), CreateImageCopy() */
  OpenCLImage
  CreateImageHost(const OpenCLImageFormat &        format,
                  void *                           data,
                  const OpenCLSize &               size,
                  const OpenCLMemoryObject::Access access);

  /** Creates a OpenCL image object with the specified \a format,
   * \a size, and \a access mode. The image is initialized with a copy of the
   * contents of \a data. The application's \a data can be discarded after the
   * image is created. Returns the new OpenCL image object, or a null object
   * if the image could not be created.
   * \sa CreateImageDevice(), CreateImageHost() */
  OpenCLImage
  CreateImageCopy(const OpenCLImageFormat &        format,
                  const void *                     data,
                  const OpenCLSize &               size,
                  const OpenCLMemoryObject::Access access);

  /** Creates an OpenCL program object from the supplied STL strings
   * \a sourceCode, \a prefixSourceCode and \a postfixSourceCode.
   * \sa CreateProgramFromSourceFile(), BuildProgramFromSourceCode() */
  OpenCLProgram
  CreateProgramFromSourceCode(const std::string & sourceCode,
                              const std::string & prefixSourceCode = std::string(),
                              const std::string & postfixSourceCode = std::string());

  /** Creates an OpenCL program object from the contents of the specified
   * by the STL string \a filename, \a prefixSourceCode and \a postfixSourceCode.
   * \sa CreateProgramFromSourceCode(), BuildProgramFromSourceFile() */
  OpenCLProgram
  CreateProgramFromSourceFile(const std::string & filename,
                              const std::string & prefixSourceCode = std::string(),
                              const std::string & postfixSourceCode = std::string());

  /** Creates an OpenCL program object from \a binary for GetDefaultDevice().
   * This function can only load the binary for a single device. For multiple
   * devices, use CreateProgramFromBinaries() instead.
   * \sa CreateProgramFromBinaryFile(), CreateProgramFromBinaries() */
  OpenCLProgram
  CreateProgramFromBinaryCode(const unsigned char * binary, const std::size_t size);

  /** Creates an OpenCL program object from the supplied STL strings
   * \a sourceCode, \a prefixSourceCode and then builds it.
   * Returns a null OpenCLProgram if the program could not be built.
   * \sa CreateProgramFromSourceCode(), BuildProgramFromSourceFile() */
  OpenCLProgram
  BuildProgramFromSourceCode(const std::string & sourceCode,
                             const std::string & prefixSourceCode = std::string(),
                             const std::string & postfixSourceCode = std::string());

  OpenCLProgram
  BuildProgramFromSourceCode(const std::list<OpenCLDevice> & devices,
                             const std::string &             sourceCode,
                             const std::string &             prefixSourceCode = std::string(),
                             const std::string &             postfixSourceCode = std::string(),
                             const std::string &             extraBuildOptions = std::string());

  /** Creates an OpenCL program object from the contents of the supplied
   * by the STL strings \a filename, \a prefixSourceCode and then builds it.
   * Returns a \b null OpenCLProgram if the program could not be built.
   * \sa CreateProgramFromSourceFile(), BuildProgramFromSourceFile() */
  OpenCLProgram
  BuildProgramFromSourceFile(const std::string & fileName,
                             const std::string & prefixSourceCode = std::string(),
                             const std::string & postfixSourceCode = std::string());

  OpenCLProgram
  BuildProgramFromSourceFile(const std::list<OpenCLDevice> & devices,
                             const std::string &             fileName,
                             const std::string &             prefixSourceCode = std::string(),
                             const std::string &             postfixSourceCode = std::string(),
                             const std::string &             extraBuildOptions = std::string());

  /** Returns the list of supported image formats for processing
   * images with the specified image type \a image_type and memory \a flags. */
  std::list<OpenCLImageFormat>
  GetSupportedImageFormats(const OpenCLImageFormat::ImageType image_type, const cl_mem_flags flags) const;

  /** Creates a sampler for this context from the arguments
   * \a normalizedCoordinates, \a addressingMode, and \a filterMode. */
  OpenCLSampler
  CreateSampler(const bool                          normalizedCoordinates,
                const OpenCLSampler::AddressingMode addressingMode,
                const OpenCLSampler::FilterMode     filterMode);

  /** Creates a user event. User events allow applications to enqueue commands
   * that wait on a user event to finish before the command is executed by the
   * device. Commands that depend upon the user event will not be executed
   * until the application triggers the user event with SetComplete(). */
  OpenCLUserEvent
  CreateUserEvent();

  /** Flushes all previously queued commands to the device associated
   * with the active command queue. The commands are delivered to
   * the device, but no guarantees are given that they will be executed.
   * \sa Finish() */
  void
  Flush();

  /** Blocks until all previously queued commands on the active
   * command queue have finished execution.
   * \sa Flush() */
  void
  Finish();

  /** Enqueues a marker command which waits for either a list of events to
   * complete, or if the list is empty it waits for all commands previously
   * enqueued in command_queue to complete before it completes.
   * \sa MarkerAsync() */
  cl_int
  Marker(const OpenCLEventList & event_list);

  /** Asynchronous version of the Marker() method.
   * This function will queue the request and return immediately. Returns an
   * OpenCLEvent object that can be used to wait for the request to finish.
   * The request will not start until all of the events in \a event_list
   * have been signaled as completed.
   * \sa Marker() */
  OpenCLEvent
  MarkerAsync(const OpenCLEventList & event_list);

  /** Enqueues a barrier command which waits for either a list of events to
   * complete, or if the list is empty it waits for all commands previously
   * enqueued in command_queue to complete before it completes.
   * \sa BarrierAsync() */
  cl_int
  Barrier(const OpenCLEventList & event_list);

  /** Asynchronous version of the Barrier() method.
   * This function will queue the request and return immediately. Returns an
   * OpenCLEvent object that can be used to wait for the request to finish.
   * The request will not start until all of the events in \a event_list
   * have been signaled as completed.
   * \sa Marker() */
  OpenCLEvent
  BarrierAsync(const OpenCLEventList & event_list);

  /** Waits on the host thread for commands identified by event objects in
   * event_list to complete. A command is considered complete if its execution
   * status is \c{CL_COMPLETE} or a negative value. The events specified in
   * event_list act as synchronization points. */
  static cl_int
  WaitForFinished(const OpenCLEventList & event_list);

protected:
  /** Constructs a new OpenCL context object. This constructor is
   * typically followed by calls to SetPlatform() and Create(). */
  OpenCLContext();

  /** Destroys this OpenCL context object. If the underlying
   * GetContextId() has been created, then it will be released. */
  ~OpenCLContext() override;

  /** \internal
   * This method is only used when CMake OpenCL profiling is enabled. */
  void
  OpenCLProfile(cl_event clEvent, const std::string & message, const bool releaseEvent = false);

  /** \internal
   * Used by OpenCLContextGL::Create() to set the default device found
   * by querying \c{CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR}. */
  void
  SetDefaultDevice(const OpenCLDevice & device);

  /** \internal
   * Used by the CreateProgramFromSourceCode(), CreateProgramFromSourceFile() */
  OpenCLProgram
  CreateOpenCLProgram(const std::string & filename, const std::string & source, const std::size_t sourceSize);

private:
  OpenCLContext(const Self & other) = delete;
  const Self &
  operator=(const Self &) = delete;

  ITK_OPENCL_DECLARE_PRIVATE(OpenCLContext)

  std::unique_ptr<OpenCLContextPimpl> d_ptr;
  static Pointer                      m_Instance;

  /** Quick get active queue method for friend classes. */
  cl_command_queue
  GetActiveQueue();

  /** \internal
   * Create method from list of devices. */
  void
  CreateContext(const std::list<OpenCLDevice> & devices, OpenCLContextPimpl * d);

  /** \internal
   * Create method from platfrom. */
  void
  CreateContext(const OpenCLPlatform & platfrom, const OpenCLDevice::DeviceType type, OpenCLContextPimpl * d);

  /** \internal
   */
  void
  SetUpProfiling();

  /** \internal
   */
  void
  OpenCLDebug(const std::string & callname);

  /** friends from OpenCL core */
  friend class OpenCLMemoryObject;
  friend class OpenCLBuffer;
  friend class OpenCLImage;
  friend class OpenCLKernel;
  friend class OpenCLCommandQueue;
  friend class OpenCLProgram;
  friend class OpenCLVectorBase;
  friend class OpenCLSampler;
};

} // end namespace itk

#endif /* itkOpenCLContext_h */

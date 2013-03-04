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
#include "itkGPUKernelManager.h"
#include "itkOpenCLKernels.h"

#include <iostream>
#include <fstream>
#include "itkTimeProbe.h"

#include "itksys/MD5.h"

namespace itk
{
std::string GetOpenCLDebugFileName(const std::string & source)
{
  // Create unique filename based on the source code
  const std::size_t sourceSize = source.size();
  itksysMD5 *       md5 = itksysMD5_New();

  itksysMD5_Initialize(md5);
  itksysMD5_Append(md5, (unsigned char *)source.c_str(), sourceSize);
  const std::size_t DigestSize = 32u;
  char              Digest[DigestSize];
  itksysMD5_FinalizeHex(md5, Digest);
  const std::string hex(Digest, DigestSize);

  // construct the name
  std::string fileName(itk::OpenCLKernelsDebugDirectory);
  fileName.append("/ocl-");
  fileName.append(hex);
  fileName.append(".cl");

  return fileName;
}

//------------------------------------------------------------------------------
class OpenCLKernelPrivate
{
public:
  OpenCLKernelPrivate(GPUContextManager *ctx, cl_kernel kid):
    context(ctx),
    id(kid),
    global_work_offset(OpenCLSize::null),
    global_work_size(1),
    local_work_size(OpenCLSize::null)
  {}
  OpenCLKernelPrivate(const OpenCLKernelPrivate *other):
    context(other->context),
    id(other->id),
    global_work_offset(other->global_work_offset),
    global_work_size(other->global_work_size),
    local_work_size(other->local_work_size)
  {
    if ( id )
      {
      clRetainKernel(id);
      }
  }

  ~OpenCLKernelPrivate()
  {
    if ( id )
      {
      clReleaseKernel(id);
      }
  }

  void copy(const OpenCLKernelPrivate *other)
  {
    context = other->context;

    global_work_offset = other->global_work_offset;
    global_work_size = other->global_work_size;
    local_work_size = other->local_work_size;

    if ( id != other->id )
      {
      if ( id )
        {
        clReleaseKernel(id);
        }
      id = other->id;
      if ( id )
        {
        clRetainKernel(id);
        }
      }
  }

  GPUContextManager *context;
  cl_kernel          id;
  OpenCLSize         global_work_offset;
  OpenCLSize         global_work_size;
  OpenCLSize         local_work_size;
};

//------------------------------------------------------------------------------
GPUKernelManager::GPUKernelManager()
{
  m_Manager = GPUContextManager::GetInstance();

  if ( m_Manager->GetNumberOfCommandQueues() > 0 )
    {
    // default command queue
    m_CommandQueueId = 0;
    }

  //d_ptr = new OpenCLKernelPrivate(m_Manager, 0);
}

//------------------------------------------------------------------------------
GPUKernelManager::~GPUKernelManager()
{
  for ( std::size_t i = 0; i < d_ptr.size(); i++ )
    {
    OpenCLKernelPrivate *d = d_ptr[i];
    delete d;
    }
  d_ptr.clear();
}

//------------------------------------------------------------------------------
void GPUKernelManager::SetGlobalWorkSize(const std::size_t kernelId,
                                         const OpenCLSize & size)
{
  OpenCLKernelPrivate *d = d_ptr[kernelId];

  d->global_work_size = size;
}

//------------------------------------------------------------------------------
OpenCLSize GPUKernelManager::GetGlobalWorkSize(const std::size_t kernelId) const
{
  const OpenCLKernelPrivate *d = d_ptr[kernelId];

  return d->global_work_size;
}

//------------------------------------------------------------------------------
void GPUKernelManager::SetLocalWorkSize(const std::size_t kernelId,
                                        const OpenCLSize & size)
{
  OpenCLKernelPrivate *d = d_ptr[kernelId];

  d->local_work_size = size;
}

//------------------------------------------------------------------------------
OpenCLSize GPUKernelManager::GetLocalWorkSize(const std::size_t kernelId) const
{
  const OpenCLKernelPrivate *d = d_ptr[kernelId];

  return d->local_work_size;
}

//------------------------------------------------------------------------------
void GPUKernelManager::SetGlobalWorkOffset(const std::size_t kernelId,
                                           const OpenCLSize & offset)
{
  OpenCLKernelPrivate *d = d_ptr[kernelId];

  d->global_work_offset = offset;
}

//------------------------------------------------------------------------------
OpenCLSize GPUKernelManager::GetGlobalWorkOffset(const std::size_t kernelId) const
{
  const OpenCLKernelPrivate *d = d_ptr[kernelId];

  return d->global_work_offset;
}

//------------------------------------------------------------------------------
bool GPUKernelManager::LoadProgramFromFile(const char *filename, const char *preamble)
{
  if ( filename == NULL )
    {
    itkWarningMacro(<< "The filename must be specified.");
    return false;
    }

  // open the file
  std::ifstream inputFile(filename, std::ifstream::in | std::ifstream::binary);
  if ( inputFile.is_open() == false )
    {
    itkWarningMacro(<< "Cannot open OpenCL source file: " << filename);
    return false;
    }

  std::stringstream sstream;
  if ( preamble != "" )
    {
    sstream << preamble << std::endl << inputFile.rdbuf();
    }
  else
    {
    sstream << inputFile.rdbuf();
    }

  inputFile.close();

  const std::string oclSource = sstream.str();
  const std::size_t oclSourceSize = oclSource.size();

  if ( oclSourceSize == 0 )
    {
    itkWarningMacro(<< "Cannot build OpenCL source file: " << filename << " is empty.");
    return false;
    }

  bool        created = false;
  std::string fileName(filename);

#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  // To work with the Intel SDK for OpenCL* - Debugger plug-in, the OpenCL*
  // kernel code must exist in a text file separate from the code of the host.
  // Also the full path to the file has to be provided.
  if ( preamble != "" )
    {
    fileName = GetOpenCLDebugFileName( oclSource.c_str() );
    }
  if ( preamble != "" )
    {
    std::ofstream debugfile( fileName.c_str() );
    if ( debugfile.is_open() == false )
      {
      itkWarningMacro(<< "Cannot create OpenCL debug source file: " << fileName);
      return false;
      }
    debugfile << oclSource;
    debugfile.close();

    itkWarningMacro(<< "For Debugging your OpenCL kernel use :"
                    << fileName << " , not original .cl file.");
    }

  std::cout << "Creating OpenCL program from : " << fileName << std::endl;
#endif

  // Create
  created = CreateOpenCLProgram(fileName, oclSource, oclSourceSize);
  return created;
}

//------------------------------------------------------------------------------
bool GPUKernelManager::LoadProgramFromString(const char *source, const char *preamble)
{
  if ( source == NULL )
    {
    itkWarningMacro(<< "The source must be specified.");
    return false;
    }

  std::stringstream sstream;
  if ( preamble != "" )
    {
    sstream << preamble << std::endl << source;
    }
  else
    {
    sstream << source;
    }

  const std::string oclSource = sstream.str();
  const std::size_t oclSourceSize = oclSource.size();

  if ( oclSourceSize == 0 )
    {
    itkWarningMacro(<< "Cannot build empty OpenCL source.");
    return false;
    }

  bool created = false;

#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  // To work with the Intel SDK for OpenCL* - Debugger plug-in, the OpenCL*
  // kernel code must exist in a text file separate from the code of the host.
  // Also the full path to the file has to be provided.
  const std::string fileName = GetOpenCLDebugFileName( oclSource.c_str() );
  if ( preamble != "" )
    {
    std::ofstream debugfile( fileName.c_str() );
    if ( debugfile.is_open() == false )
      {
      itkWarningMacro(<< "Cannot create OpenCL debug source file: " << fileName);
      return false;
      }
    debugfile << oclSource;
    debugfile.close();

    itkWarningMacro(<< "For Debugging your OpenCL kernel use :" << fileName);
    }

  std::cout << "Creating OpenCL program from source." << std::endl;
#endif

  // Create
#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  created = CreateOpenCLProgram(fileName, oclSource, oclSourceSize);
#else
  created = CreateOpenCLProgram(std::string(), oclSource, oclSourceSize);
#endif

  return created;
}

//------------------------------------------------------------------------------
bool GPUKernelManager::CreateOpenCLProgram(const std::string & filename,
                                           const std::string & source,
                                           const std::size_t sourceSize)
{
#ifdef OPENCL_PROFILING
  itk::TimeProbe buildtimer;
  buildtimer.Start();
#endif

  cl_int error;
#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  std::cout << "clCreateProgramWithSource" << "..." << std::endl;
#endif
  const char *code = source.c_str();
  m_Program = clCreateProgramWithSource(m_Manager->GetCurrentContext(), 1,
                                        &code, &sourceSize, &error);
  OpenCLCheckError(error, __FILE__, __LINE__, ITK_LOCATION);

  if ( error != CL_SUCCESS )
    {
    itkWarningMacro("Cannot create GPU program");
    return false;
    }

  // Get OpenCL math and optimization options
  std::string oclMathAndOptimization;
  const bool  oclMathAndOptimizationEnabled =
    GetOpenCLMathAndOptimizationOptions(oclMathAndOptimization);

#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  if ( filename.size() > 0 )
    {
    std::cout << "clBuildProgram '" << filename << "' ..." << std::endl;
    }
  else
    {
    std::cout << "clBuildProgram from source ..." << std::endl;
    }
#endif

#if defined( ITK_USE_INTEL_CPU_OPENCL ) && defined( _DEBUG )
  // Enable debugging mode in the Intel OpenCL runtime
  if ( filename.size() > 0 )
    {
    std::string oclDebugOptions = "-g -s \"" + filename + "\"";
    if ( oclMathAndOptimizationEnabled )
      {
      oclDebugOptions = oclDebugOptions + " " + oclMathAndOptimization;
      error = clBuildProgram(m_Program, 1,
                             &m_Manager->GetDevices()[m_Manager->GetTargetDeviceId()],
                             oclDebugOptions.c_str(), NULL, NULL);
      }
    else
      {
      error = clBuildProgram(m_Program, 1,
                             &m_Manager->GetDevices()[m_Manager->GetTargetDeviceId()],
                             oclDebugOptions.c_str(), NULL, NULL);
      }
    }
#else
  if ( oclMathAndOptimizationEnabled )
    {
    error = clBuildProgram(m_Program, 1,
                           &m_Manager->GetDevices()[m_Manager->GetTargetDeviceId()],
                           oclMathAndOptimization.c_str(), NULL, NULL);
    }
  else
    {
    error = clBuildProgram(m_Program, 1,
                           &m_Manager->GetDevices()[m_Manager->GetTargetDeviceId()],
                           NULL, NULL, NULL);
    }
#endif

  if ( error != CL_SUCCESS )
    {
    //itkWarningMacro("OpenCL program build error");

    // print out build error
    std::size_t paramValueSize = 0;

    // get error message size
    clGetProgramBuildInfo(m_Program, m_Manager->GetDeviceId(0),
                          CL_PROGRAM_BUILD_LOG, 0, NULL, &paramValueSize);

    char *paramValue;
    paramValue = (char *)malloc(paramValueSize);

    // get error message
    clGetProgramBuildInfo(m_Program, m_Manager->GetDeviceId(0),
                          CL_PROGRAM_BUILD_LOG, paramValueSize, paramValue, NULL);

    /*
    std::ostringstream itkmsg;
    itkmsg << "ERROR: In " __FILE__ ", line " << __LINE__ << "\n"
    << this->GetNameOfClass() << " (" << this << "): "
    << "OpenCL program build error:" << paramValue
    << "\n\n";
    ::itk::OutputWindowDisplayErrorText( itkmsg.str().c_str() );
    */

    std::cerr << paramValue << std::endl;

    free(paramValue);

    OpenCLCheckError(error, __FILE__, __LINE__, ITK_LOCATION);

    return false;
    }

#ifdef OPENCL_PROFILING
  buildtimer.Stop();
  std::cout << "GPU Build program took " << buildtimer.GetMean() << " seconds." << std::endl;
#endif

  return true;
}

//------------------------------------------------------------------------------
int GPUKernelManager::CreateKernel(const char *kernelName)
{
  cl_int error;

  // create kernel
#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  std::cout << "clCreateKernel" << "..." << std::endl;
#endif
  cl_kernel newKernel = clCreateKernel(m_Program, kernelName, &error);
  OpenCLCheckError(error, __FILE__, __LINE__, ITK_LOCATION);

  if ( error != CL_SUCCESS ) // failed
    {
    itkWarningMacro("Fail to create GPU kernel");
    return -1;
    }

  m_KernelContainer.push_back(newKernel);
  d_ptr.push_back( new OpenCLKernelPrivate(m_Manager, newKernel) );

  // argument list
  m_KernelArgumentReady.push_back( std::vector< KernelArgumentList >() );
  cl_uint nArg;

#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  std::cout << "clGetKernelInfo" << "..." << std::endl;
#endif

  error = clGetKernelInfo(newKernel, CL_KERNEL_NUM_ARGS, sizeof( cl_uint ), &nArg, NULL);
  OpenCLCheckError(error, __FILE__, __LINE__, ITK_LOCATION);

  ( m_KernelArgumentReady.back() ).resize(nArg);

  ResetArguments( (int)m_KernelContainer.size() - 1 );

  return (int)m_KernelContainer.size() - 1;
}

//------------------------------------------------------------------------------
cl_int GPUKernelManager::GetKernelWorkGroupInfo(const std::size_t kernelId,
                                                cl_kernel_work_group_info paramName,
                                                void *value)
{
  std::size_t valueSize, valueSizeRet;

  switch ( paramName )
    {
    case CL_KERNEL_WORK_GROUP_SIZE:
      valueSize = sizeof( std::size_t );
      break;
    case CL_KERNEL_COMPILE_WORK_GROUP_SIZE:
      valueSize = 3 * sizeof( std::size_t );
      break;
    case CL_KERNEL_LOCAL_MEM_SIZE:
      valueSize = sizeof( cl_ulong );
      break;
    default:
      itkGenericExceptionMacro (<< "Unknown type of work goup information");
      break;
    }
#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  std::cout << "clGetKernelWorkGroupInfo" << "..." << std::endl;
#endif
  cl_int error = clGetKernelWorkGroupInfo(m_KernelContainer[kernelId], m_Manager->GetDeviceId(0),
                                          paramName, valueSize, value, &valueSizeRet);

  OpenCLCheckError(error, __FILE__, __LINE__, ITK_LOCATION);

  return error;
}

//------------------------------------------------------------------------------
bool GPUKernelManager::SetKernelArg(const std::size_t kernelId,
                                    const cl_uint argId, const std::size_t argSize,
                                    const void *argVal)
{
  if ( kernelId >= m_KernelContainer.size() ) { return false; }

  cl_int error;
#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  std::cout << "clSetKernelArg" << "..." << std::endl;
#endif
  error = clSetKernelArg(m_KernelContainer[kernelId], argId, argSize, argVal);

  if ( error != CL_SUCCESS )
    {
    itkWarningMacro("Setting kernel argument failed with GPUKernelManager::SetKernelArg("
                    << kernelId << ", " << argId << ", " << argSize << ". " << argVal << ")");
    }

  OpenCLCheckError(error, __FILE__, __LINE__, ITK_LOCATION);

  m_KernelArgumentReady[kernelId][argId].m_IsReady = true;
  m_KernelArgumentReady[kernelId][argId].m_GPUDataManager = (GPUDataManager::Pointer)NULL;

  return true;
}

//------------------------------------------------------------------------------
bool GPUKernelManager::SetKernelArgWithImage(const std::size_t kernelId, const cl_uint argId,
                                             const GPUDataManager::Pointer manager)
{
  if ( kernelId >= m_KernelContainer.size() ) { return false; }

  cl_int error;
#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  std::cout << "clSetKernelArg" << "..." << std::endl;
#endif

  if ( manager->GetBufferSize() > 0 )
    {
    error = clSetKernelArg( m_KernelContainer[kernelId], argId, sizeof( cl_mem ), manager->GetGPUBufferPointer() );
    }
  else
    {
    // Check and remove it for Intel SDK for OpenCL 2013
#if defined( ITK_USE_INTEL_CPU_OPENCL )
    // http://software.intel.com/en-us/forums/topic/281206
    itkWarningMacro("Intel SDK for OpenCL 2012 does not support setting NULL buffers.");
    return false;
#endif
    // According OpenCL 1.1 specification clSetKernelArg arg_value could be NULL
    // object.
    cl_mem null_buffer = NULL;
    error = clSetKernelArg(m_KernelContainer[kernelId], argId, sizeof( cl_mem ), &null_buffer);
    }

  if ( error != CL_SUCCESS )
    {
    itkWarningMacro("Setting kernel argument failed with GPUKernelManager::SetKernelArgWithImage("
                    << kernelId << ", " << argId << ", " << manager << ")");
    }

  OpenCLCheckError(error, __FILE__, __LINE__, ITK_LOCATION);

  m_KernelArgumentReady[kernelId][argId].m_IsReady = true;
  m_KernelArgumentReady[kernelId][argId].m_GPUDataManager = manager;

  return true;
}

//------------------------------------------------------------------------------
// this function must be called right before GPU kernel is launched
bool GPUKernelManager::CheckArgumentReady(const std::size_t kernelId)
{
  const std::size_t nArg = m_KernelArgumentReady[kernelId].size();

  for ( std::size_t i = 0; i < nArg; i++ )
    {
    if ( !( m_KernelArgumentReady[kernelId][i].m_IsReady ) ) { return false; }

    // automatic synchronization before kernel launch
    if ( m_KernelArgumentReady[kernelId][i].m_GPUDataManager != (GPUDataManager::Pointer)NULL )
      {
      m_KernelArgumentReady[kernelId][i].m_GPUDataManager->SetCPUBufferDirty();
      }
    }
  return true;
}

//------------------------------------------------------------------------------
void GPUKernelManager::ResetArguments(const std::size_t kernelIdx)
{
  const std::size_t nArg = m_KernelArgumentReady[kernelIdx].size();

  for ( std::size_t i = 0; i < nArg; i++ )
    {
    m_KernelArgumentReady[kernelIdx][i].m_IsReady = false;
    m_KernelArgumentReady[kernelIdx][i].m_GPUDataManager = (GPUDataManager::Pointer)NULL;
    }
}

//------------------------------------------------------------------------------
OpenCLEvent GPUKernelManager::LaunchKernel(const std::size_t kernelId)
{
  if ( kernelId >= m_KernelContainer.size() )
    {
    return OpenCLEvent();
    }

  if ( !CheckArgumentReady(kernelId) )
    {
    // it is a bit ugly way to retrieve kernel name, but fast and work, remove it later
    char name[128];
    size_t size = 0;
    cl_kernel kernel = m_KernelContainer[kernelId];
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 128, name, NULL);
    itkWarningMacro("GPU kernel arguments are not completely assigned for the kernel '" << name << "'");
    return OpenCLEvent();
    }

  cl_event event;

#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  std::cout << "clEnqueueNDRangeKernel[" << kernelId << "]..." << std::endl;
#endif

  const OpenCLKernelPrivate *d = d_ptr[kernelId];

  cl_command_queue command_queue = m_Manager->GetCommandQueue(m_CommandQueueId);
  cl_kernel        kernel = m_KernelContainer[kernelId];
  const cl_uint    work_dim = d->global_work_size.GetDimension();

  const bool gwoNull = d->global_work_offset.IsNull();
  const bool lwsNull = d->local_work_size.IsNull();

  cl_int error;

  if ( gwoNull && lwsNull )
    {
    error = clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
                                   NULL, d->global_work_size.GetSizes(), NULL, 0, 0, &event);
    }
  else if ( gwoNull && !lwsNull )
    {
    error = clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
                                   NULL, d->global_work_size.GetSizes(),
                                   ( d->local_work_size.GetWidth() ? d->local_work_size.GetSizes() : 0 ),

                                   0, 0, &event);
    }
  else if ( !gwoNull && lwsNull )
    {
    error = clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
                                   d->global_work_offset.GetSizes(), d->global_work_size.GetSizes(),
                                   NULL, 0, 0, &event);
    }
  else
    {
    error = clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
                                   d->global_work_offset.GetSizes(), d->global_work_size.GetSizes(),
                                   ( d->local_work_size.GetWidth() ? d->local_work_size.GetSizes() : 0 ),
                                   0, 0, &event);
    }

  if ( error != CL_SUCCESS )
    {
    // it is a bit ugly way to retrieve kernel name, but fast and work, remove it later
    char name[128];
    size_t size = 0;
    cl_kernel kernel = m_KernelContainer[kernelId];
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 128, name, NULL);
    itkWarningMacro("Launch kernel '" << name << "' failed with GPUKernelManager::LaunchKernel("
                    << kernelId << ")");
    }

  OpenCLCheckError(error, __FILE__, __LINE__, ITK_LOCATION);
  m_Manager->OpenCLProfile(event, "clEnqueueNDRangeKernel");

  if ( error != CL_SUCCESS )
    {
    return OpenCLEvent();
    }
  else
    {
    return OpenCLEvent(event);
    }
}

//------------------------------------------------------------------------------
OpenCLEvent GPUKernelManager::LaunchKernel(const std::size_t kernelId,
                                           const OpenCLSize & global_work_size,
                                           const OpenCLSize & local_work_size,
                                           const OpenCLSize & global_work_offset)
{
  SetGlobalWorkSize(kernelId, global_work_size);
  SetLocalWorkSize(kernelId, local_work_size);
  SetGlobalWorkOffset(kernelId, global_work_offset);
  return LaunchKernel(kernelId);
}

//------------------------------------------------------------------------------
OpenCLEvent GPUKernelManager::LaunchKernel(const std::size_t kernelId,
                                           const OpenCLEventList & after)
{
  if ( kernelId >= m_KernelContainer.size() )
    {
    return OpenCLEvent();
    }

  if ( !CheckArgumentReady(kernelId) )
    {
    // it is a bit ugly way to retrieve kernel name, but fast and work, remove it later
    char name[128];
    size_t size = 0;
    cl_kernel kernel = m_KernelContainer[kernelId];
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 128, name, NULL);
    itkWarningMacro("GPU kernel arguments are not completely assigned for the kernel '" << name << "'");
    return OpenCLEvent();
    }

  cl_event event;

#if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
  std::cout << "clEnqueueNDRangeKernel[" << kernelId << "]..." << std::endl;
#endif

  const OpenCLKernelPrivate *d = d_ptr[kernelId];

  cl_command_queue command_queue = m_Manager->GetCommandQueue(m_CommandQueueId);
  cl_kernel        kernel = m_KernelContainer[kernelId];
  const cl_uint    work_dim = d->global_work_size.GetDimension();

  const bool gwoNull = d->global_work_offset.IsNull();
  const bool lwsNull = d->local_work_size.IsNull();

  cl_int error;

  if ( gwoNull && lwsNull )
    {
    error = clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
                                   NULL, d->global_work_size.GetSizes(), NULL,
                                   after.GetSize(), after.GetEventData(), &event);
    }
  else if ( gwoNull && !lwsNull )
    {
    error = clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
                                   NULL, d->global_work_size.GetSizes(),
                                   ( d->local_work_size.GetWidth() ? d->local_work_size.GetSizes() : 0 ),
                                   after.GetSize(), after.GetEventData(), &event);
    }
  else if ( !gwoNull && lwsNull )
    {
    error = clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
                                   d->global_work_offset.GetSizes(), d->global_work_size.GetSizes(),
                                   NULL,
                                   after.GetSize(), after.GetEventData(), &event);
    }
  else
    {
    error = clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
                                   d->global_work_offset.GetSizes(), d->global_work_size.GetSizes(),
                                   ( d->local_work_size.GetWidth() ? d->local_work_size.GetSizes() : 0 ),
                                   after.GetSize(), after.GetEventData(), &event);
    }

  if ( error != CL_SUCCESS )
    {
    // it is a bit ugly way to retrieve kernel name, but fast and work, remove it later
    char name[128];
    size_t size = 0;
    cl_kernel kernel = m_KernelContainer[kernelId];
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 128, name, NULL);

    itkWarningMacro("Launch kernel '" << name << "' failed with GPUKernelManager::LaunchKernel("
      << kernelId << ",\n" << after << ")");
    }

  OpenCLCheckError(error, __FILE__, __LINE__, ITK_LOCATION);
  m_Manager->OpenCLProfile(event, "clEnqueueNDRangeKernel");

  if ( error != CL_SUCCESS )
    {
    return OpenCLEvent();
    }
  else
    {
    return OpenCLEvent(event);
    }
}

//------------------------------------------------------------------------------
OpenCLEvent GPUKernelManager::LaunchKernel(const std::size_t kernelId,
                                           const OpenCLEventList & after,
                                           const OpenCLSize & global_work_size,
                                           const OpenCLSize & local_work_size,
                                           const OpenCLSize & global_work_offset)
{
  SetGlobalWorkSize(kernelId, global_work_size);
  SetLocalWorkSize(kernelId, local_work_size);
  SetGlobalWorkOffset(kernelId, global_work_offset);
  return LaunchKernel(kernelId, after);
}

//------------------------------------------------------------------------------
void GPUKernelManager::SetCurrentCommandQueue(const std::size_t queueid)
{
  if ( queueid >= 0 && queueid < m_Manager->GetNumberOfCommandQueues() )
    {
    // Assumption: different command queue is assigned to different device
    m_CommandQueueId = queueid;
    }
  else
    {
    itkWarningMacro("Not a valid command queue id");
    }
}

//------------------------------------------------------------------------------
int GPUKernelManager::GetCurrentCommandQueueID()
{
  return m_CommandQueueId;
}
} // end namespace itk

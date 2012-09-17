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
#include <assert.h>
#include "itkGPUContextManager.h"

namespace itk
{
// static variable initialization
GPUContextManager *GPUContextManager::m_Instance = NULL;

//------------------------------------------------------------------------------
GPUContextManager * GPUContextManager::GetInstance()
{
  if ( m_Instance == NULL )
    {
    m_Instance = new GPUContextManager();
    }
  return m_Instance;
}

//------------------------------------------------------------------------------
void GPUContextManager::DestroyInstance()
{
  delete m_Instance;
  m_Instance = NULL;
}

//------------------------------------------------------------------------------
GPUContextManager::GPUContextManager():
  m_Context(NULL),
  m_CommandQueue(NULL),
  m_Devices(NULL),
  m_Platform(NULL),
  m_NumberOfDevices(0),
  m_NumberOfPlatforms(0),
  m_TargetDevice(0)
{
// Intel platform
#ifdef ITK_USE_INTEL_OPENCL
#ifdef ITK_USE_INTEL_GPU_OPENCL
  SetPlatform("Intel(R) OpenCL", CL_DEVICE_TYPE_GPU);
#elif ITK_USE_INTEL_CPU_OPENCL
  SetPlatform("Intel(R) OpenCL", CL_DEVICE_TYPE_CPU);
#else
  itkGenericExceptionMacro(<< "Unknown Intel OpenCL platform.");
#endif

// NVidia platform
#elif ITK_USE_NVIDIA_OPENCL
  SetPlatform("NVIDIA", CL_DEVICE_TYPE_GPU);

// AMD platform
#elif ITK_USE_AMD_OPENCL
#ifdef ITK_USE_AMD_GPU_OPENCL
  SetPlatform("Advanced Micro Devices, Inc.", CL_DEVICE_TYPE_GPU);
#elif ITK_USE_AMD_CPU_OPENCL
  SetPlatform("Advanced Micro Devices, Inc.", CL_DEVICE_TYPE_CPU);
#else
  itkGenericExceptionMacro(<< "Unknown AMD OpenCL platform.");
#endif

// Unknown platform
#else
  itkGenericExceptionMacro(<< "Not supported OpenCL platform by GPUContextManager.");
#endif
}

//------------------------------------------------------------------------------
GPUContextManager::~GPUContextManager()
{
  // TODO: check if this is correct
  free(m_Platform);
  free(m_Devices);
}

//------------------------------------------------------------------------------
void GPUContextManager::SetPlatform(const std::string & platformName,
                                    const cl_device_type deviceType)
{
  // TODO: check if this is correct
  if ( m_Platform )
    {
    free(m_Platform);
    }
  if ( m_Devices )
    {
    free(m_Devices);
    }

  cl_int errid;

  // Get the platforms
  errid = clGetPlatformIDs(0, NULL, &m_NumberOfPlatforms);
  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);

  // Set platform
#ifdef OPENCL_PROFILING
  m_Platform = OpenCLSelectPlatform(platformName.c_str(), true);
#else
  m_Platform = OpenCLSelectPlatform(platformName.c_str(), false);
#endif

  if ( m_Platform == NULL )
    {
    itkGenericExceptionMacro(<< "Unknow " << platformName << "platform.");
    return;
    }

  // Get the devices
  m_Devices = OpenCLGetAvailableDevices(m_Platform, deviceType, &m_NumberOfDevices);

  // create context
  m_Context = clCreateContext(0, m_NumberOfDevices, &m_Devices[m_TargetDevice], NULL, NULL, &errid);
  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);

#ifdef OPENCL_PROFILING
  OpenCLPrintDeviceInfo(m_Devices[m_TargetDevice], true);
#endif

  // create command queues
  m_CommandQueue = (cl_command_queue *)malloc( m_NumberOfDevices * sizeof( cl_command_queue ) );
  for ( unsigned int i = 0; i < m_NumberOfDevices; i++ )
    {
#ifdef OPENCL_PROFILING
    m_CommandQueue[i] = clCreateCommandQueue(m_Context, m_Devices[i], CL_QUEUE_PROFILING_ENABLE, &errid);
#else
    m_CommandQueue[i] = clCreateCommandQueue(m_Context, m_Devices[i], 0, &errid);
#endif
    //OclPrintDeviceName(m_Devices[i]);
    OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
    }
}

//------------------------------------------------------------------------------
cl_command_queue GPUContextManager::GetCommandQueue(int i)
{
  if ( i < 0 || i >= (int)m_NumberOfDevices )
    {
    printf("Error: requested queue id is not available. Default queue will be used (queue id = 0)\n");
    return m_CommandQueue[0];
    }
  return m_CommandQueue[i];
}

//------------------------------------------------------------------------------
cl_device_id GPUContextManager::GetDeviceId(int i)
{
  if ( i < 0 || i >= (int)m_NumberOfDevices )
    {
    printf("Error: requested queue id is not available. Default queue will be used (queue id = 0)\n");
    return m_Devices[0];
    }
  return m_Devices[i];
}

//------------------------------------------------------------------------------
void GPUContextManager::OpenCLProfile(cl_event clEvent, const std::string & message)
{
#ifdef OPENCL_PROFILING
  if ( !clEvent )
    {
    return;
    }

  cl_int errid;

  // Execution time
  errid = clWaitForEvents(1, &clEvent);
  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
  if ( errid != CL_SUCCESS )
    {
    itkWarningMacro("clWaitForEvents failed");
    return;
    }
  cl_ulong start, end;
  errid = clGetEventProfilingInfo(clEvent, CL_PROFILING_COMMAND_END, sizeof( cl_ulong ), &end, NULL);
  errid |= clGetEventProfilingInfo(clEvent, CL_PROFILING_COMMAND_START, sizeof( cl_ulong ), &start, NULL);
  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
  if ( errid != CL_SUCCESS )
    {
    itkWarningMacro("clGetEventProfilingInfo failed");
    return;
    }
  double dSeconds = 1.0e-9 * (double)( end - start );
  std::cout << "GPU " << message << " execution took " << dSeconds << " seconds." << std::endl;

  // Release event
  errid = clReleaseEvent(clEvent);
  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
  if ( errid != CL_SUCCESS )
    {
    itkWarningMacro("clReleaseEvent failed");
    return;
    }
  clEvent = 0;
#endif
}
} // namespace itk

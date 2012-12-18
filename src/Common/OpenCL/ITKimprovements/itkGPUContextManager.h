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
#ifndef __itkGPUContextManager_h
#define __itkGPUContextManager_h

#include "itkOpenCLUtil.h"
#include <itkLightObject.h>

// Singleton class for GPUContextManager
namespace itk
{
/** \class GPUContextManager
 *
 * \brief Class to store the GPU context.
 *
 *  Won-Ki to write more documentation here...
 *
 * \note This file was taken from ITK 4.1.0.
 * It was modified by Denis P. Shamonin and Marius Staring.
 * Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands.
 * Added functionality is described in the Insight Journal paper:
 * http://hdl.handle.net/xxxx/yyy
 *
 *
 * \ingroup ITKGPUCommon
 */
class ITKOpenCL_EXPORT GPUContextManager:public LightObject
{
public:

  static GPUContextManager * GetInstance();

  void DestroyInstance();

  cl_command_queue GetCommandQueue(int i);

  unsigned int GetNumberOfCommandQueues()
  {
    return m_NumberOfDevices;
  }

  cl_context GetCurrentContext()
  {
    return m_Context;
  }

  cl_device_id GetDeviceId(int i);

  cl_device_id * GetDevices()
  {
    return m_Devices;
  }

  cl_uint GetTargetDeviceId()
  {
    return m_TargetDevice;
  }

  void SetPlatform(const std::string & platformName, const cl_device_type deviceType);

  void OpenCLProfile(cl_event clEvent, const std::string & message);

private:

  GPUContextManager();
  ~GPUContextManager();

  cl_platform_id    m_Platform;
  cl_context        m_Context;
  cl_device_id *    m_Devices;
  cl_command_queue *m_CommandQueue;  // one queue per device
  cl_uint           m_NumberOfDevices;
  cl_uint           m_NumberOfPlatforms;
  cl_uint           m_TargetDevice;

  static GPUContextManager *m_Instance;
};
} // namespace itk

#endif

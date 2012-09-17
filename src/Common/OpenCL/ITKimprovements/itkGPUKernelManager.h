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

//
// GPU Kernel Manager Class
//

#ifndef __itkGPUKernelManager_h
#define __itkGPUKernelManager_h

#include <vector>
#include <itkLightObject.h>
#include <itkObjectFactory.h>
#include "itkOpenCLUtil.h"
#include "itkOpenCLEvent.h"
#include "itkOpenCLSize.h"
#include "itkGPUImage.h"
#include "itkGPUContextManager.h"
#include "itkGPUDataManager.h"

namespace itk
{
/** \class GPUKernelManager
 * \brief GPU kernel manager implemented using OpenCL.
 *
 * This class is responsible for managing the GPU kernel and
 * command queue.
 *
 * \ingroup ITKGPUCommon
 */

class OpenCLKernelPrivate;

//------------------------------------------------------------------------------
class ITKOpenCL_EXPORT GPUKernelManager:public LightObject
{
public:

  struct KernelArgumentList {
    bool m_IsReady;
    GPUDataManager::Pointer m_GPUDataManager;
  };

  typedef GPUKernelManager           Self;
  typedef LightObject                Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  itkNewMacro(Self);
  itkTypeMacro(GPUKernelManager, LightObject);

  void SetGlobalWorkSize(const size_t kernelId, const OpenCLSize & size);

  OpenCLSize GetGlobalWorkSize(const std::size_t kernelId) const;

  void SetLocalWorkSize(const size_t kernelId, const OpenCLSize & size);

  OpenCLSize GetLocalWorkSize(const std::size_t kernelId) const;

  void SetGlobalWorkOffset(const size_t kernelId, const OpenCLSize & offset);

  OpenCLSize GetGlobalWorkOffset(const std::size_t kernelId) const;

  OpenCLEvent LaunchKernel(const size_t kernelId);

  OpenCLEvent LaunchKernel(const size_t kernelId,
                           const OpenCLSize & global_work_size,
                           const OpenCLSize & local_work_size = OpenCLSize::null,
                           const OpenCLSize & global_work_offset = OpenCLSize::null);

  OpenCLEvent LaunchKernel(const size_t kernelId, const OpenCLEventList & after);

  OpenCLEvent LaunchKernel(const size_t kernelId, const OpenCLEventList & after,
                           const OpenCLSize & global_work_size,
                           const OpenCLSize & local_work_size = OpenCLSize::null,
                           const OpenCLSize & global_work_offset = OpenCLSize::null);

  bool LoadProgramFromFile(const char *filename, const char *preamble = "");

  bool LoadProgramFromString(const char *source, const char *preamble = "");

  int  CreateKernel(const char *kernelName);

  cl_int GetKernelWorkGroupInfo(const size_t kernelId,
                                cl_kernel_work_group_info paramName, void *value);

  bool SetKernelArg(const size_t kernelId,
                    const cl_uint argId, const size_t argSize, const void *argVal);

  bool SetKernelArgWithImage(const size_t kernelId, cl_uint argId, GPUDataManager::Pointer manager);

  void SetCurrentCommandQueue(const size_t queueid);

  int  GetCurrentCommandQueueID();

protected:
  GPUKernelManager();
  virtual ~GPUKernelManager();

  bool CheckArgumentReady(const size_t kernelId);

  void ResetArguments(const size_t kernelIdx);

  bool CreateOpenCLProgram(const std::string & filename,
                           const std::string & source, const std::size_t sourceSize);

private:
  GPUKernelManager(const Self &); // purposely not implemented
  void operator=(const Self &);   // purposely not implemented

  cl_program m_Program;

  GPUContextManager *m_Manager;
  int                m_CommandQueueId;

  std::vector< OpenCLKernelPrivate * > d_ptr;
  //WeakPointer<OpenCLKernelPrivate> d_ptr;

  std::vector< cl_kernel >                         m_KernelContainer;
  std::vector< std::vector< KernelArgumentList > > m_KernelArgumentReady;
};
} // end namespace itk

#endif

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
#ifndef itkGPUImageDataManager_hxx
#define itkGPUImageDataManager_hxx

#include "itkGPUImageDataManager.h"

//#define VERBOSE

namespace itk
{
template <typename ImageType>
void
GPUImageDataManager<ImageType>::SetImagePointer(typename ImageType::Pointer img)
{
  m_Image = img;
}


//------------------------------------------------------------------------------
template <typename ImageType>
void
GPUImageDataManager<ImageType>::UpdateCPUBuffer()
{
  if (this->m_CPUBufferLock)
  {
    return;
  }

  if (m_Image.IsNotNull())
  {
    m_Mutex.lock();

    unsigned long gpu_time = this->GetMTime();
    TimeStamp     cpu_time_stamp = m_Image->GetTimeStamp();
    unsigned long cpu_time = cpu_time_stamp.GetMTime();

    /* Why we check dirty flag and time stamp together?
     * Because existing CPU image filters do not use pixel/buffer
     * access function in GPUImage and therefore dirty flag is not
     * correctly managed. Therefore, we check the time stamp of
     * CPU and GPU data as well
     */
    if ((m_IsCPUBufferDirty || (gpu_time > cpu_time)) && m_GPUBuffer != nullptr && m_CPUBuffer != nullptr)
    {
      cl_int errid;
#if (defined(_WIN32) && defined(_DEBUG)) || !defined(NDEBUG)
      std::cout << "clEnqueueReadBuffer GPU->CPU..." << std::endl;
#endif

#ifdef OPENCL_PROFILING
      cl_event clEvent = NULL;
      errid = clEnqueueReadBuffer(m_Context->GetCommandQueue().GetQueueId(),
                                  m_GPUBuffer,
                                  CL_TRUE,
                                  0,
                                  m_BufferSize,
                                  m_CPUBuffer,
                                  0,
                                  nullptr,
                                  &clEvent);
#else
      errid = clEnqueueReadBuffer(
        m_Context->GetCommandQueue().GetQueueId(), m_GPUBuffer, CL_TRUE, 0, m_BufferSize, m_CPUBuffer, 0, 0, 0);
#endif

      m_Context->ReportError(errid, __FILE__, __LINE__, ITK_LOCATION);
      // m_ContextManager->OpenCLProfile(clEvent, "clEnqueueReadBuffer GPU->CPU");

      m_Image->Modified();
      this->SetTimeStamp(m_Image->GetTimeStamp());

      m_IsCPUBufferDirty = false;
      m_IsGPUBufferDirty = false;
    }

    m_Mutex.unlock();
  }
}


//------------------------------------------------------------------------------
template <typename ImageType>
void
GPUImageDataManager<ImageType>::UpdateGPUBuffer()
{
  if (this->m_GPUBufferLock)
  {
    return;
  }

  if (m_Image.IsNotNull())
  {
    m_Mutex.lock();

    unsigned long gpu_time = this->GetMTime();
    TimeStamp     cpu_time_stamp = m_Image->GetTimeStamp();
    unsigned long cpu_time = m_Image->GetMTime();

    /* Why we check dirty flag and time stamp together?
     * Because existing CPU image filters do not use pixel/buffer
     * access function in GPUImage and therefore dirty flag is not
     * correctly managed. Therefore, we check the time stamp of
     * CPU and GPU data as well
     */
    if ((m_IsGPUBufferDirty || (gpu_time < cpu_time)) && m_CPUBuffer != nullptr && m_GPUBuffer != nullptr)
    {
      cl_int errid;
#if (defined(_WIN32) && defined(_DEBUG)) || !defined(NDEBUG)
      std::cout << "clEnqueueWriteBuffer CPU->GPU..." << std::endl;
#endif

#ifdef OPENCL_PROFILING
      cl_event clEvent = NULL;
      errid = clEnqueueWriteBuffer(m_Context->GetCommandQueue().GetQueueId(),
                                   m_GPUBuffer,
                                   CL_TRUE,
                                   0,
                                   m_BufferSize,
                                   m_CPUBuffer,
                                   0,
                                   nullptr,
                                   &clEvent);
#else
      errid = clEnqueueWriteBuffer(m_Context->GetCommandQueue().GetQueueId(),
                                   m_GPUBuffer,
                                   CL_TRUE,
                                   0,
                                   m_BufferSize,
                                   m_CPUBuffer,
                                   0,
                                   nullptr,
                                   nullptr);
#endif
      m_Context->ReportError(errid, __FILE__, __LINE__, ITK_LOCATION);
      // m_ContextManager->OpenCLProfile(clEvent, "clEnqueueWriteBuffer CPU->GPU");

      this->SetTimeStamp(cpu_time_stamp);

      m_IsCPUBufferDirty = false;
      m_IsGPUBufferDirty = false;
    }

    m_Mutex.unlock();
  }
}


//------------------------------------------------------------------------------
template <typename ImageType>
void
GPUImageDataManager<ImageType>::Graft(const GPUImageDataManager * data)
{
  // std::cout << "GPU timestamp : " << this->GetMTime() << ", CPU timestamp : "
  // << m_Image->GetMTime() << std::endl;

  Superclass::Graft(data);

  // std::cout << "GPU timestamp : " << this->GetMTime() << ", CPU timestamp : "
  // << m_Image->GetMTime() << std::endl;
}


} // namespace itk

#endif

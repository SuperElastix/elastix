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
#ifndef itkGPURecursiveGaussianImageFilter_hxx
#define itkGPURecursiveGaussianImageFilter_hxx

#include "itkGPURecursiveGaussianImageFilter.h"
#include "itkOpenCLUtil.h"
#include "itkOpenCLEvent.h"
#include "itkOpenCLDevice.h"

namespace itk
{
template <typename TInputImage, typename TOutputImage>
GPURecursiveGaussianImageFilter<TInputImage, TOutputImage>::GPURecursiveGaussianImageFilter()
{
  std::ostringstream defines;

  if (TInputImage::ImageDimension > 3 || TInputImage::ImageDimension < 1)
  {
    itkExceptionMacro("GPURecursiveGaussianImageFilter supports 1/2/3D image.");
  }

  if (TInputImage::ImageDimension == 1)
  {
    defines << "#define DIM_1\n";
  }
  else
  {
    defines << "#define DIM_" << int(TInputImage::ImageDimension - 1) << "\n";
  }

  // Define m_DeviceLocalMemorySize as:
  // local memory: 16384 bytes / 3 buffers of float = 1365
  const unsigned long localMemSize = this->m_GPUKernelManager->GetContext()->GetDefaultDevice().GetLocalMemorySize();

  this->m_DeviceLocalMemorySize = (localMemSize / 3) / sizeof(float);

  defines << "#define BUFFSIZE " << this->m_DeviceLocalMemorySize << "\n";
  defines << "#define BUFFPIXELTYPE float\n";
  defines << "#define INPIXELTYPE ";
  GetTypenameInString(typeid(typename TInputImage::PixelType), defines);
  defines << "#define OUTPIXELTYPE ";
  GetTypenameInString(typeid(typename TOutputImage::PixelType), defines);

  // OpenCL kernel source
  const char * GPUSource = GPURecursiveGaussianImageFilterKernel::GetOpenCLSource();
  // Build and create kernel
  OpenCLProgram program = this->m_GPUKernelManager->BuildProgramFromSourceCode(GPUSource, defines.str());
  if (!program.IsNull())
  {
    this->m_FilterGPUKernelHandle = this->m_GPUKernelManager->CreateKernel(program, "RecursiveGaussianImageFilter");
  }
  else
  {
    itkExceptionMacro(<< "Kernel has not been loaded from:\n" << GPUSource);
  }
}


//------------------------------------------------------------------------------
template <typename TInputImage, typename TOutputImage>
void
GPURecursiveGaussianImageFilter<TInputImage, TOutputImage>::GPUGenerateData()
{
  itkDebugMacro(<< "Calling GPURecursiveGaussianImageFilter::GPUGenerateData()");

  using GPUInputImage = typename GPUTraits<TInputImage>::Type;
  using GPUOutputImage = typename GPUTraits<TOutputImage>::Type;

  const typename GPUInputImage::Pointer inPtr = dynamic_cast<GPUInputImage *>(this->ProcessObject::GetInput(0));
  typename GPUOutputImage::Pointer      otPtr = dynamic_cast<GPUOutputImage *>(this->ProcessObject::GetOutput(0));

  // Perform the safe check
  if (inPtr.IsNull())
  {
    itkExceptionMacro(<< "The GPU InputImage is NULL. Filter unable to perform.");
    return;
  }
  if (otPtr.IsNull())
  {
    itkExceptionMacro(<< "The GPU OutputImage is NULL. Filter unable to perform.");
    return;
  }

  const typename GPUOutputImage::SizeType outSize = otPtr->GetLargestPossibleRegion().GetSize();
  const unsigned int                      ln = outSize[this->GetDirection()];
  const unsigned int                      ImageDim = (unsigned int)(TInputImage::ImageDimension);

  // Check if GPU filter are able to perform for this image
  if (ln > this->m_DeviceLocalMemorySize)
  {
    itkExceptionMacro(<< "GPURecursiveGaussianImageFilter unable to perform.");
    return;
  }

  int imgSize[TInputImage::ImageDimension];
  for (unsigned int i = 0; i < ImageDim; ++i)
  {
    imgSize[i] = outSize[i];
  }

  std::size_t globalSize1D = 0, globalSize2D[2];
  for (unsigned int i = 0; i < 2; ++i)
  {
    globalSize2D[i] = 0;
  }

  // Initialize globalSize, localSize here
  if (ImageDim == 3)
  {
    // 0 (direction x) : y/z
    // 1 (direction y) : x/z
    // 2 (direction z) : x/y
    switch (this->GetDirection())
    {
      case 0:
        globalSize2D[0] = imgSize[1];
        globalSize2D[1] = imgSize[2];
        break;
      case 1:
        globalSize2D[0] = imgSize[0];
        globalSize2D[1] = imgSize[2];
        break;
      case 2:
        globalSize2D[0] = imgSize[0];
        globalSize2D[1] = imgSize[1];
        break;
    }
  }
  else if (ImageDim == 2)
  {
    globalSize1D = ln;
  }
  else if (ImageDim == 1)
  {
    globalSize1D = 1;
  }

  // Arguments set up
  int argidx = 0;
  this->m_GPUKernelManager->SetKernelArgWithImage(this->m_FilterGPUKernelHandle, argidx++, inPtr->GetGPUDataManager());
  this->m_GPUKernelManager->SetKernelArgWithImage(this->m_FilterGPUKernelHandle, argidx++, otPtr->GetGPUDataManager());

  // Set ln
  this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(ln), &(ln));

  // Set direction
  const int direction = this->GetDirection();
  this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(int), &(direction));

  // Set causal coefficients
  cl_float4 N;
  N.s[0] = static_cast<float>(this->m_N0);
  N.s[1] = static_cast<float>(this->m_N1);
  N.s[2] = static_cast<float>(this->m_N2);
  N.s[3] = static_cast<float>(this->m_N3);
  this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_float4), (void *)&N);

  // Set recursive coefficients
  cl_float4 D;
  D.s[0] = static_cast<float>(this->m_D1);
  D.s[1] = static_cast<float>(this->m_D2);
  D.s[2] = static_cast<float>(this->m_D3);
  D.s[3] = static_cast<float>(this->m_D4);
  this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_float4), (void *)&D);

  // Set anti-causal coefficients
  cl_float4 M;
  M.s[0] = static_cast<float>(this->m_M1);
  M.s[1] = static_cast<float>(this->m_M2);
  M.s[2] = static_cast<float>(this->m_M3);
  M.s[3] = static_cast<float>(this->m_M4);
  this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_float4), (void *)&M);

  // Set recursive coefficients to be used at the boundaries
  cl_float4 BN;
  BN.s[0] = static_cast<float>(this->m_BN1);
  BN.s[1] = static_cast<float>(this->m_BN2);
  BN.s[2] = static_cast<float>(this->m_BN3);
  BN.s[3] = static_cast<float>(this->m_BN4);
  this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_float4), (void *)&BN);

  cl_float4 BM;
  BM.s[0] = static_cast<float>(this->m_BM1);
  BM.s[1] = static_cast<float>(this->m_BM2);
  BM.s[2] = static_cast<float>(this->m_BM3);
  BM.s[3] = static_cast<float>(this->m_BM4);
  this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_float4), (void *)&BM);

  // Set image size
  for (unsigned int i = 0; i < ImageDim; ++i)
  {
    this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint), &(imgSize[i]));
  }
  if (ImageDim == 1)
  {
    const int height = 0;
    this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint), &(height));
  }

  // Launch kernel
  switch (ImageDim)
  {
    case 1:
    case 2:
    {
      OpenCLEvent event = this->m_GPUKernelManager->LaunchKernel(m_FilterGPUKernelHandle, OpenCLSize(globalSize1D));
      event.WaitForFinished();
    }
    break;
    case 3:
    {
      OpenCLEvent event =
        this->m_GPUKernelManager->LaunchKernel(m_FilterGPUKernelHandle, OpenCLSize(globalSize2D[0], globalSize2D[1]));
      event.WaitForFinished();
    }
    break;
  }

  itkDebugMacro(<< "GPURecursiveGaussianImageFilter::GPUGenerateData() finished");
}


//------------------------------------------------------------------------------
template <typename TInputImage, typename TOutputImage>
void
GPURecursiveGaussianImageFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  CPUSuperclass::PrintSelf(os, indent);
  GPUSuperclass::PrintSelf(os, indent);
}


} // end namespace itk

#endif /* itkGPURecursiveGaussianImageFilter_hxx */

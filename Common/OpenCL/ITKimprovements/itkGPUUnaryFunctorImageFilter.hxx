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
#ifndef itkGPUUnaryFunctorImageFilter_hxx
#define itkGPUUnaryFunctorImageFilter_hxx

#include "itkGPUUnaryFunctorImageFilter.h"
#include "itkOpenCLUtil.h"

namespace itk
{
template <typename TInputImage, typename TOutputImage, typename TFunction, typename TParentImageFilter>
void
GPUUnaryFunctorImageFilter<TInputImage, TOutputImage, TFunction, TParentImageFilter>::GenerateOutputInformation()
{
  CPUSuperclass::GenerateOutputInformation();
}


//------------------------------------------------------------------------------
template <typename TInputImage, typename TOutputImage, typename TFunction, typename TParentImageFilter>
void
GPUUnaryFunctorImageFilter<TInputImage, TOutputImage, TFunction, TParentImageFilter>::GPUGenerateData()
{
  // Applying functor using GPU kernel
  using GPUInputImage = typename itk::GPUTraits<TInputImage>::Type;
  using GPUOutputImage = typename itk::GPUTraits<TOutputImage>::Type;

  typename GPUInputImage::Pointer  inPtr = dynamic_cast<GPUInputImage *>(this->ProcessObject::GetInput(0));
  typename GPUOutputImage::Pointer otPtr = dynamic_cast<GPUOutputImage *>(this->ProcessObject::GetOutput(0));

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

  int imgSize[3];
  imgSize[0] = imgSize[1] = imgSize[2] = 1;

  const unsigned int ImageDim = (unsigned int)TInputImage::ImageDimension;

  for (std::size_t i = 0; i < ImageDim; ++i)
  {
    imgSize[i] = outSize[i];
  }

  typename GPUInputImage::SizeType localSize, globalSize;
  for (std::size_t i = 0; i < ImageDim; ++i)
  {
    localSize[i] = OpenCLGetLocalBlockSize(InputImageDimension);
    // total # of threads
    globalSize[i] =
      localSize[i] *
      (static_cast<unsigned int>(std::ceil(static_cast<float>(outSize[i]) / static_cast<float>(localSize[i]))));
  }

  // arguments set up using Functor
  int argidx =
    (this->GetFunctor()).SetGPUKernelArguments(this->m_GPUKernelManager, m_UnaryFunctorImageFilterGPUKernelHandle);

  // arguments set up
  this->m_GPUKernelManager->SetKernelArgWithImage(
    m_UnaryFunctorImageFilterGPUKernelHandle, argidx++, inPtr->GetGPUDataManager());
  this->m_GPUKernelManager->SetKernelArgWithImage(
    m_UnaryFunctorImageFilterGPUKernelHandle, argidx++, otPtr->GetGPUDataManager());

  for (std::size_t i = 0; i < ImageDim; ++i)
  {
    this->m_GPUKernelManager->SetKernelArg(
      m_UnaryFunctorImageFilterGPUKernelHandle, argidx++, sizeof(cl_uint), &(imgSize[i]));
  }

  // launch kernel
  this->m_GPUKernelManager->LaunchKernel(
    m_UnaryFunctorImageFilterGPUKernelHandle, OpenCLSize(globalSize), OpenCLSize(localSize));
}


} // end of namespace itk

#endif

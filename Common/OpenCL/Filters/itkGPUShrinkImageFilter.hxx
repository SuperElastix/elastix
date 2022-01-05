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
#ifndef itkGPUShrinkImageFilter_hxx
#define itkGPUShrinkImageFilter_hxx

#include "itkGPUShrinkImageFilter.h"
#include "itkOpenCLUtil.h"

namespace itk
{
/**
 * ****************** Constructor ***********************
 */

template <typename TInputImage, typename TOutputImage>
GPUShrinkImageFilter<TInputImage, TOutputImage>::GPUShrinkImageFilter()
{
  std::ostringstream defines;

  if (TInputImage::ImageDimension > 3 || TInputImage::ImageDimension < 1)
  {
    itkExceptionMacro("GPUShrinkImageFilter supports 1/2/3D image.");
  }
  defines << "#define DIM_" << int(TInputImage::ImageDimension) << "\n";

  defines << "#define INPIXELTYPE ";
  GetTypenameInString(typeid(typename TInputImage::PixelType), defines);
  defines << "#define OUTPIXELTYPE ";
  GetTypenameInString(typeid(typename TOutputImage::PixelType), defines);

  // OpenCL kernel source
  const char * GPUSource = GPUShrinkImageFilterKernel::GetOpenCLSource();
  // Build and create kernel
  OpenCLProgram program = this->m_GPUKernelManager->BuildProgramFromSourceCode(GPUSource, defines.str());
  if (!program.IsNull())
  {
    this->m_FilterGPUKernelHandle = this->m_GPUKernelManager->CreateKernel(program, "ShrinkImageFilter");
  }
  else
  {
    itkExceptionMacro(<< "Kernel has not been loaded from:\n" << GPUSource);
  }
} // end Constructor()


/**
 * ****************** GPUGenerateData ***********************
 */

template <typename TInputImage, typename TOutputImage>
void
GPUShrinkImageFilter<TInputImage, TOutputImage>::GPUGenerateData()
{
  itkDebugMacro(<< "Calling GPUShrinkImageFilter::GPUGenerateData()");

  using GPUInputImage = typename GPUTraits<TInputImage>::Type;
  using GPUOutputImage = typename GPUTraits<TOutputImage>::Type;

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

  // Convert the factor for convenient multiplication
  typename TOutputImage::SizeType factorSize;
  const ShrinkFactorsType         shrinkFactors = this->GetShrinkFactors();
  for (std::size_t i = 0; i < InputImageDimension; ++i)
  {
    factorSize[i] = shrinkFactors[i];
  }

  // Define a few indices that will be used to transform from an input pixel
  // to an output pixel
  OutputIndexType  outputIndex;
  InputIndexType   inputIndex;
  OutputOffsetType offsetIndex;

  typename TOutputImage::PointType tempPoint;

  // Use this index to compute the offset everywhere in this class
  outputIndex = otPtr->GetLargestPossibleRegion().GetIndex();

  // We wish to perform the following mapping of outputIndex to
  // inputIndex on all points in our region
  otPtr->TransformIndexToPhysicalPoint(outputIndex, tempPoint);
  inPtr->TransformPhysicalPointToIndex(tempPoint, inputIndex);

  // Given that the size is scaled by a constant factor eq:
  // inputIndex = outputIndex * factorSize
  // is equivalent up to a fixed offset which we now compute
  OffsetValueType zeroOffset = 0;
  for (std::size_t i = 0; i < InputImageDimension; ++i)
  {
    offsetIndex[i] = inputIndex[i] - outputIndex[i] * shrinkFactors[i];
    // It is plausible that due to small amounts of loss of numerical
    // precision that the offset is negative, this would cause sampling
    // out of out region, this is insurance against that possibility
    offsetIndex[i] = std::max(zeroOffset, offsetIndex[i]);
  }

  const typename GPUOutputImage::SizeType inSize = inPtr->GetLargestPossibleRegion().GetSize();
  const typename GPUOutputImage::SizeType outSize = otPtr->GetLargestPossibleRegion().GetSize();

  const OpenCLSize localSize = OpenCLSize::GetLocalWorkSize(this->m_GPUKernelManager->GetContext()->GetDefaultDevice());

  typename GPUInputImage::SizeType globalSize;
  for (std::size_t i = 0; i < InputImageDimension; ++i)
  {
    // total # of threads
    globalSize[i] =
      localSize[i] *
      (static_cast<unsigned int>(std::ceil(static_cast<float>(outSize[i]) / static_cast<float>(localSize[i]))));
  }

  // arguments set up
  int argidx = 0;
  this->m_GPUKernelManager->SetKernelArgWithImage(this->m_FilterGPUKernelHandle, argidx++, inPtr->GetGPUDataManager());
  this->m_GPUKernelManager->SetKernelArgWithImage(this->m_FilterGPUKernelHandle, argidx++, otPtr->GetGPUDataManager());

  // set arguments for image size/offset/shrinkfactors
  unsigned int inImageSize[InputImageDimension];
  unsigned int outImageSize[InputImageDimension];
  for (unsigned int i = 0; i < InputImageDimension; ++i)
  {
    inImageSize[i] = inSize[i];
    outImageSize[i] = outSize[i];
  }

  unsigned int offset[InputImageDimension];
  unsigned int shrinkfactors[InputImageDimension];

  for (std::size_t i = 0; i < InputImageDimension; ++i)
  {
    offset[i] = offsetIndex[i];
    shrinkfactors[i] = factorSize[i];
  }

  const unsigned int ImageDim = static_cast<unsigned int>(InputImageDimension);
  switch (ImageDim)
  {
    case 1:
      this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint), &inImageSize);
      this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint), &outImageSize);
      this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint), &offset);
      this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint), &shrinkfactors);
      break;
    case 2:
      this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint2), &inImageSize);
      this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint2), &outImageSize);
      this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint2), &offset);
      this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint2), &shrinkfactors);
      break;
    case 3:
      this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint3), &inImageSize);
      this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint3), &outImageSize);
      this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint3), &offset);
      this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint3), &shrinkfactors);
      break;
  }

  // launch kernel
  OpenCLEvent event =
    this->m_GPUKernelManager->LaunchKernel(this->m_FilterGPUKernelHandle, OpenCLSize(globalSize), localSize);

  event.WaitForFinished();

  itkDebugMacro(<< "GPUShrinkImageFilter::GPUGenerateData() finished");
} // end GPUGenerateData()


/**
 * ****************** PrintSelf ***********************
 */

template <typename TInputImage, typename TOutputImage>
void
GPUShrinkImageFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  CPUSuperclass::PrintSelf(os, indent);
  GPUSuperclass::PrintSelf(os, indent);
} // end PrintSelf()


} // end namespace itk

#endif /* itkGPUShrinkImageFilter_hxx */

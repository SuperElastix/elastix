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
#ifndef __itkGPUShrinkImageFilter_hxx
#define __itkGPUShrinkImageFilter_hxx

#include "itkGPUShrinkImageFilter.h"
#include "itkGPUKernels.h"

namespace itk
{
template< class TInputImage, class TOutputImage >
GPUShrinkImageFilter< TInputImage, TOutputImage >::GPUShrinkImageFilter()
{
  std::ostringstream defines;

  if(TInputImage::ImageDimension > 3 || TInputImage::ImageDimension < 1)
  {
    itkExceptionMacro("GPUShrinkImageFilter supports 1/2/3D image.");
  }
  defines << "#define DIM_" << int(TInputImage::ImageDimension) << "\n";

  defines << "#define INPIXELTYPE ";
  GetTypenameInString( typeid ( typename TInputImage::PixelType ), defines );
  defines << "#define OUTPIXELTYPE ";
  GetTypenameInString( typeid ( typename TOutputImage::PixelType ), defines );

  // OpenCL source path
  const std::string oclSrcPath(oclGPUShrinkImageFilter);
  // Load and create kernel
  const bool loaded = this->m_GPUKernelManager->LoadProgramFromFile(oclSrcPath.c_str(), defines.str().c_str());
  if(loaded)
  {
    m_FilterGPUKernelHandle = this->m_GPUKernelManager->CreateKernel("ShrinkImageFilter");
  }
  else
  {
    itkExceptionMacro( << "Kernel has not been loaded from: " << oclSrcPath );
  }
}

//------------------------------------------------------------------------------
template< class TInputImage, class TOutputImage >
void GPUShrinkImageFilter< TInputImage, TOutputImage >::GPUGenerateData()
{
  typedef typename GPUTraits<TInputImage>::Type  GPUInputImage;
  typedef typename GPUTraits<TOutputImage>::Type GPUOutputImage;

  typename GPUInputImage::Pointer  inPtr = dynamic_cast<GPUInputImage *>( this->ProcessObject::GetInput(0) );
  typename GPUOutputImage::Pointer otPtr = dynamic_cast<GPUOutputImage *>( this->ProcessObject::GetOutput(0) );

  // Perform the safe check
  if(inPtr.IsNull())
  {
    itkExceptionMacro(<< "The GPU InputImage is NULL. Filter unable to perform.");
    return;
  }
  if(otPtr.IsNull())
  {
    itkExceptionMacro(<< "The GPU OutputImage is NULL. Filter unable to perform.");
    return;
  }

  // Convert the factor for convenient multiplication
  typename TOutputImage::SizeType factorSize;
  const ShrinkFactorsType shrinkFactors = this->GetShrinkFactors();
  for(unsigned int i = 0; i < TInputImage::ImageDimension; i++)
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
  for(unsigned int i = 0; i < TInputImage::ImageDimension; i++)
  {
    offsetIndex[i] = inputIndex[i] - outputIndex[i] * shrinkFactors[i];
    // It is plausible that due to small amounts of loss of numerical
    // precision that the offset it negaive, this would cause sampling
    // out of out region, this is insurance against that possibility
    offsetIndex[i] = vnl_math_max(zeroOffset, offsetIndex[i]);
  }

  const unsigned int ImageDim = (unsigned int)(TInputImage::ImageDimension);
  typename GPUOutputImage::SizeType inSize  = inPtr->GetLargestPossibleRegion().GetSize();
  typename GPUOutputImage::SizeType outSize = otPtr->GetLargestPossibleRegion().GetSize();
  size_t localSize[3], globalSize[3];
  localSize[0] = localSize[1] = localSize[2] = OpenCLGetLocalBlockSize(ImageDim);

  for(unsigned int i=0; i<ImageDim; i++)
  {
    // total # of threads
    globalSize[i] = localSize[i]*(unsigned int)ceil( (float)outSize[i]/(float)localSize[i]);
  }

  // arguments set up
  int argidx = 0;
  this->m_GPUKernelManager->SetKernelArgWithImage(m_FilterGPUKernelHandle, argidx++, inPtr->GetGPUDataManager());
  this->m_GPUKernelManager->SetKernelArgWithImage(m_FilterGPUKernelHandle, argidx++, otPtr->GetGPUDataManager());

  // set arguments for image size/offset/shrinkfactors
  unsigned int inImageSize[TInputImage::ImageDimension];
  unsigned int outImageSize[TInputImage::ImageDimension];
  for(unsigned int i=0; i<TInputImage::ImageDimension; i++)
  {
    inImageSize[i]  = inSize[i];
    outImageSize[i] = outSize[i];
  }

  //signed long offset[TInputImage::ImageDimension];
  //unsigned long factorsize[TInputImage::ImageDimension];
  unsigned int offset[TInputImage::ImageDimension];
  unsigned int shrinkfactors[TInputImage::ImageDimension];

  for(unsigned int i = 0; i < TInputImage::ImageDimension; i++)
  {
    offset[i] = offsetIndex[i];
    shrinkfactors[i] = factorSize[i];
  }

  switch (ImageDim)
  {
  case 1:
    this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint), &(inImageSize));
    this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint), &(outImageSize));
    this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint), &(offset));
    this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint), &(shrinkfactors));
    break;
  case 2:
    this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint2), &(inImageSize));
    this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint2), &(outImageSize));
    this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint2), &(offset));
    this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint2), &(shrinkfactors));
    break;
  case 3:
    this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint3), &(inImageSize));
    this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint3), &(outImageSize));
    this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint3), &(offset));
    this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint3), &(shrinkfactors));
    break;
  }

  // launch kernel
  this->m_GPUKernelManager->LaunchKernel(m_FilterGPUKernelHandle, ImageDim, globalSize, localSize);
}

template< class TInputImage, class TOutputImage >
void GPUShrinkImageFilter< TInputImage, TOutputImage >::PrintSelf(std::ostream & os, Indent indent) const
{
  CPUSuperclass::PrintSelf(os, indent);
  GPUSuperclass::PrintSelf(os, indent);
}

} // end namespace itk

#endif

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
#ifndef itkGPUBSplineDecompositionImageFilter_hxx
#define itkGPUBSplineDecompositionImageFilter_hxx

#include "itkGPUBSplineDecompositionImageFilter.h"
#include "itkGPUCastImageFilter.h"
#include "itkOpenCLUtil.h"
#include "itkOpenCLDevice.h"

namespace itk
{
/**
 * ****************** Constructor ***********************
 */

template <typename TInputImage, typename TOutputImage>
GPUBSplineDecompositionImageFilter<TInputImage, TOutputImage>::GPUBSplineDecompositionImageFilter()
{
  std::ostringstream defines;

  if (TInputImage::ImageDimension > 3 || TInputImage::ImageDimension < 1)
  {
    itkExceptionMacro("GPUBSplineDecompositionImageFilter supports 1/2/3D image.");
  }

  // \todo: explain this:
  if (TInputImage::ImageDimension == 1)
  {
    defines << "#define DIM_1\n";
  }
  else
  {
    defines << "#define DIM_" << int(TInputImage::ImageDimension - 1) << "\n";
  }

  // Define m_DeviceLocalMemorySize as:
  // local memory: 16384 bytes / 1 buffer of float = 4096 - offset
  const unsigned long localMemSize = this->m_GPUKernelManager->GetContext()->GetDefaultDevice().GetLocalMemorySize();

  this->m_DeviceLocalMemorySize = (localMemSize / sizeof(float)) - 3 * sizeof(float);

  defines << "#define BUFFSIZE " << this->m_DeviceLocalMemorySize << "\n";
  defines << "#define BUFFPIXELTYPE float\n";
  defines << "#define INPIXELTYPE ";
  GetTypenameInString(typeid(typename TInputImage::PixelType), defines);
  defines << "#define OUTPIXELTYPE ";
  GetTypenameInString(typeid(typename TOutputImage::PixelType), defines);

  // OpenCL kernel source
  const char * GPUSource = GPUBSplineDecompositionImageFilterKernel::GetOpenCLSource();
  // Build and create kernel
  OpenCLProgram program = this->m_GPUKernelManager->BuildProgramFromSourceCode(GPUSource, defines.str());
  if (!program.IsNull())
  {
    this->m_FilterGPUKernelHandle = this->m_GPUKernelManager->CreateKernel(program, "BSplineDecompositionImageFilter");
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
GPUBSplineDecompositionImageFilter<TInputImage, TOutputImage>::GPUGenerateData()
{
  itkDebugMacro(<< "Calling GPUBSplineDecompositionImageFilter::GPUGenerateData()");

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
  const typename GPUInputImage::SizeType  dataLength = inPtr->GetLargestPossibleRegion().GetSize();
  typename GPUOutputImage::SizeValueType  maxLength = 0;

  for (std::size_t n = 0; n < InputImageDimension; ++n)
  {
    if (dataLength[n] > maxLength)
    {
      maxLength = dataLength[n];
    }
  }

  // Check if GPU filter are able to perform for this image
  if (maxLength > this->m_DeviceLocalMemorySize)
  {
    itkExceptionMacro(<< "GPUBSplineDecompositionImageFilter unable to perform.");
    return;
  }

  // Cast here, see the same call in this->CopyImageToImage() of
  // BSplineDecompositionImageFilter::DataToCoefficientsND()
  using CasterType = GPUCastImageFilter<GPUInputImage, GPUOutputImage>;
  auto caster = CasterType::New();
  caster->SetInput(inPtr);
  caster->GraftOutput(otPtr);
  caster->Update();

  typename GPUInputImage::SizeType localSize, globalSize;
  for (std::size_t i = 0; i < InputImageDimension; ++i)
  {
    localSize[i] = OpenCLGetLocalBlockSize(InputImageDimension);
    // total # of threads
    globalSize[i] =
      localSize[i] *
      (static_cast<unsigned int>(std::ceil(static_cast<float>(outSize[i]) / static_cast<float>(localSize[i]))));
  }

  // Make GPU buffer not dirty
  otPtr->GetGPUDataManager()->SetGPUDirtyFlag(false);

  // arguments set up
  int argidx = 0;
  this->m_GPUKernelManager->SetKernelArgWithImage(this->m_FilterGPUKernelHandle, argidx++, inPtr->GetGPUDataManager());
  this->m_GPUKernelManager->SetKernelArgWithImage(this->m_FilterGPUKernelHandle, argidx++, otPtr->GetGPUDataManager());

  // set image size
  unsigned int imageSize[InputImageDimension];
  for (std::size_t i = 0; i < InputImageDimension; ++i)
  {
    imageSize[i] = outSize[i];
  }

  // Solving warning "case label value exceeds maximum value for type"
  // by making a local copy of the input image dimension.
  // switch( InputImageDimension )
  const unsigned int ImageDim = (unsigned int)(InputImageDimension);
  switch (ImageDim)
  {
    case 1:
      unsigned int imageSize1D[2];
      imageSize1D[0] = imageSize[0];
      imageSize1D[1] = 0;
      this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint2), &imageSize1D);
      break;
    case 2:
      this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint2), &imageSize);
      break;
    case 3:
      this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_uint3), &imageSize);
      break;
  }

  // Set poles calculated for a given spline order
  float                       spline_poles[2];
  const int                   itkNumberOfPoles = this->GetNumberOfPoles();
  const SplinePolesVectorType itkSplinePoles = this->GetSplinePoles();
  if (itkNumberOfPoles == 1)
  {
    spline_poles[0] = static_cast<float>(itkSplinePoles[0]);
    spline_poles[1] = 0.0;
  }
  else if (itkNumberOfPoles == 2)
  {
    spline_poles[0] = static_cast<float>(itkSplinePoles[0]);
    spline_poles[1] = static_cast<float>(itkSplinePoles[1]);
  }
  this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_float2), &spline_poles);

  // Set m_NumberOfPoles
  this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx++, sizeof(cl_int), &itkNumberOfPoles);

  // Loop over directions
  OpenCLEventList eventList;
  for (std::size_t n = 0; n < InputImageDimension; ++n)
  {
    this->m_GPUKernelManager->SetKernelArg(this->m_FilterGPUKernelHandle, argidx, sizeof(cl_uint), &n);

    unsigned int x = 0, y = 2;
    switch (n)
    {
      case 0:
        x = 1;
        break;
      case 2:
        y = 1;
        break;
    }

    switch (ImageDim)
    {
      case 1:
      {
        OpenCLEvent event =
          this->m_GPUKernelManager->LaunchKernel(this->m_FilterGPUKernelHandle, OpenCLSize(1), OpenCLSize(1));
        eventList.Append(event);
      }
      break;
      case 2:
      {
        OpenCLEvent event = this->m_GPUKernelManager->LaunchKernel(
          this->m_FilterGPUKernelHandle, OpenCLSize(globalSize[n]), OpenCLSize(localSize[n]));
        eventList.Append(event);
      }
      break;
      case 3:
      {
        OpenCLEvent event = this->m_GPUKernelManager->LaunchKernel(this->m_FilterGPUKernelHandle,
                                                                   OpenCLSize(globalSize[x], globalSize[y]),
                                                                   OpenCLSize(localSize[n], localSize[n]));
        eventList.Append(event);
      }
      break;
    }
  } // end loop over InputImageDimension

  eventList.WaitForFinished();

  itkDebugMacro(<< "GPUBSplineDecompositionImageFilter::GPUGenerateData() finished");
} // end GPUGenerateData()


/**
 * ****************** PrintSelf ***********************
 */

template <typename TInputImage, typename TOutputImage>
void
GPUBSplineDecompositionImageFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  CPUSuperclass::PrintSelf(os, indent);
  GPUSuperclass::PrintSelf(os, indent);
} // end PrintSelf()


} // end namespace itk

#endif /* itkGPUBSplineDecompositionImageFilter_hxx */

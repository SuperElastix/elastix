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
#ifndef __itkGPUCastImageFilter_hxx
#define __itkGPUCastImageFilter_hxx

#include "itkGPUCastImageFilter.h"
#include "itkGPUKernels.h"

namespace itk
{
/**
* Constructor
*/
template< class TInputImage, class TOutputImage >
GPUCastImageFilter< TInputImage, TOutputImage >
::GPUCastImageFilter()
{
  std::ostringstream defines;

  if(TInputImage::ImageDimension > 3 || TInputImage::ImageDimension < 1)
  {
    itkExceptionMacro("GPUCastImageFilter supports 1/2/3D image.");
  }

  defines << "#define DIM_" << TInputImage::ImageDimension << "\n";
  defines << "#define INPIXELTYPE ";
  GetTypenameInString( typeid ( typename TInputImage::PixelType ), defines );
  defines << "#define OUTPIXELTYPE ";
  GetTypenameInString( typeid ( typename TOutputImage::PixelType ), defines );

  // OpenCL source path
  const std::string oclSrcPath(oclGPUCastImageFilter);
  // Load and create kernel
  bool loaded = this->m_GPUKernelManager->LoadProgramFromFile(oclSrcPath.c_str(), defines.str().c_str());
  if(loaded)
  {
    this->m_UnaryFunctorImageFilterGPUKernelHandle = this->m_GPUKernelManager->CreateKernel("CastImageFilter");
  }
  else
  {
    itkExceptionMacro( << "Kernel has not been loaded from: " << oclSrcPath );
  }
}

template< class TInputImage, class TOutputImage >
void
GPUCastImageFilter< TInputImage, TOutputImage >
::GPUGenerateData()
{
  GPUSuperclass::GPUGenerateData();
}

} // end of namespace itk

#endif

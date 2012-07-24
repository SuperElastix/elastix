/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPURecursiveGaussianImageFilter_hxx
#define __itkGPURecursiveGaussianImageFilter_hxx

#include "itkGPURecursiveGaussianImageFilter.h"
#include "itkGPUKernels.h"

namespace itk
{
template< class TInputImage, class TOutputImage >
GPURecursiveGaussianImageFilter< TInputImage, TOutputImage >::GPURecursiveGaussianImageFilter()
{
  std::ostringstream defines;

  if(TInputImage::ImageDimension > 3 || TInputImage::ImageDimension < 1)
  {
    itkExceptionMacro("GPURecursiveGaussianImageFilter supports 1/2/3D image.");
  }

  if(TInputImage::ImageDimension == 1)
    defines << "#define DIM_1\n";
  else
    defines << "#define DIM_" << int(TInputImage::ImageDimension-1) << "\n";

  // This is hack for now, we don't know the LOCAL_MEM_SIZE in advance usually called with:
  // cl_ulong localMemSize;
  // clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemSize, 0);
  // So, you would like to ask it from m_GPUKernelManager for example:
  // m_DeviceLocalMemorySize = m_GPUKernelManager->GetDeviceLocalMemorySize();

  // local memory: 16384 bytes / 3 buffers of float = 1365
  m_DeviceLocalMemorySize = 1365;
  defines << "#define BUFFSIZE "<< m_DeviceLocalMemorySize << "\n";
  defines << "#define BUFFPIXELTYPE float" << "\n";
  defines << "#define INPIXELTYPE ";
  GetTypenameInString( typeid ( typename TInputImage::PixelType ), defines );
  defines << "#define OUTPIXELTYPE ";
  GetTypenameInString( typeid ( typename TOutputImage::PixelType ), defines );

  // OpenCL source path
  const std::string oclSrcPath(oclGPURecursiveGaussianImageFilter);
  // Load and create kernel
  const bool loaded = this->m_GPUKernelManager->LoadProgramFromFile(oclSrcPath.c_str(), defines.str().c_str());
  if(loaded)
  {
    m_FilterGPUKernelHandle = this->m_GPUKernelManager->CreateKernel("RecursiveGaussianImageFilter");
  }
  else
  {
    itkExceptionMacro( << "Kernel has not been loaded from: " << oclSrcPath );
  }
}

//------------------------------------------------------------------------------
template< class TInputImage, class TOutputImage >
void GPURecursiveGaussianImageFilter< TInputImage, TOutputImage >::GPUGenerateData()
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

  const typename GPUOutputImage::SizeType outSize = otPtr->GetLargestPossibleRegion().GetSize();
  const unsigned int ln = outSize[this->GetDirection()];
  const unsigned int ImageDim = (unsigned int)(TInputImage::ImageDimension);

  // Check if GPU filter are able to perform for this image
  if(ln > m_DeviceLocalMemorySize)
  {
    itkExceptionMacro(<< "GPURecursiveGaussianImageFilter unable to perform.");
    return;
  }

  int imgSize[TInputImage::ImageDimension];
  for(unsigned int i=0; i<ImageDim; i++)
  {
    imgSize[i] = outSize[i];
  }

  size_t globalSize1D = 0, globalSize2D[2];
  size_t localSize1D = 0, localSize2D[2];
  for(unsigned int i=0; i<2; i++)
  {
    globalSize2D[i] = 0;
    localSize2D[i] = 0;
  }

  // Initialize globalSize, localSize here
  if(ImageDim == 3)
  {
    // 0 (direction x) : y/z
    // 1 (direction y) : x/z
    // 2 (direction z) : x/y
    switch(this->GetDirection())
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

    // A bit difficult to set it, lets just keep it safe
    if(ln > static_cast<unsigned int>(OpenCLGetLocalBlockSize(ImageDim)))
      localSize2D[0] = OpenCLGetLocalBlockSize(ImageDim);
    else
      localSize2D[0]  = 1; // Always safe
    
    localSize2D[1] = 1;
  }
  else if(ImageDim == 2)
  {
    globalSize1D = ln;

    // A bit difficult to set it, lets just keep it safe
    if(ln > static_cast<unsigned int>(OpenCLGetLocalBlockSize(ImageDim)))
      localSize1D = OpenCLGetLocalBlockSize(ImageDim);
    else
      localSize1D = 1; // Always safe
  }
  else if(ImageDim == 1)
  {
    globalSize1D = 1;
    localSize1D = 1;
  }

  // Arguments set up
  int argidx = 0;
  this->m_GPUKernelManager->SetKernelArgWithImage(m_FilterGPUKernelHandle, argidx++, inPtr->GetGPUDataManager());
  this->m_GPUKernelManager->SetKernelArgWithImage(m_FilterGPUKernelHandle, argidx++, otPtr->GetGPUDataManager());

  // Set ln
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(ln), &(ln));

  // Set direction
  const int direction = this->GetDirection();
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(int), &(direction));

  // Set causal coefficients
  const float N0 = static_cast<float>(this->m_N0);
  const float N1 = static_cast<float>(this->m_N1);
  const float N2 = static_cast<float>(this->m_N2);
  const float N3 = static_cast<float>(this->m_N3);
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(float), &(N0));
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(float), &(N1));
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(float), &(N2));
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(float), &(N3));

  // Set recursive coefficients
  const float D1 = static_cast<float>(this->m_D1);
  const float D2 = static_cast<float>(this->m_D2);
  const float D3 = static_cast<float>(this->m_D3);
  const float D4 = static_cast<float>(this->m_D4);
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(float), &(D1));
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(float), &(D2));
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(float), &(D3));
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(float), &(D4));

  // Set anti-causal coefficients
  const float M1 = static_cast<float>(this->m_M1);
  const float M2 = static_cast<float>(this->m_M2);
  const float M3 = static_cast<float>(this->m_M3);
  const float M4 = static_cast<float>(this->m_M4);
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(float), &(M1));
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(float), &(M2));
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(float), &(M3));
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(float), &(M4));

  // Set recursive coefficients to be used at the boundaries
  const float BN1 = static_cast<float>(this->m_BN1);
  const float BN2 = static_cast<float>(this->m_BN2);
  const float BN3 = static_cast<float>(this->m_BN3);
  const float BN4 = static_cast<float>(this->m_BN4);
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(float), &(BN1));
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(float), &(BN2));
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(float), &(BN3));
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(float), &(BN4));

  const float BM1 = static_cast<float>(this->m_BM1);
  const float BM2 = static_cast<float>(this->m_BM2);
  const float BM3 = static_cast<float>(this->m_BM3);
  const float BM4 = static_cast<float>(this->m_BM4);
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(float), &(BM1));
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(float), &(BM2));
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(float), &(BM3));
  this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(float), &(BM4));

  // Set image size
  for(unsigned int i=0; i<ImageDim; i++)
  {
    this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(int), &(imgSize[i]));
  }
  if(ImageDim == 1)
  {
    const int height = 0;
    this->m_GPUKernelManager->SetKernelArg(m_FilterGPUKernelHandle, argidx++, sizeof(int), &(height));
  }

  // Launch kernel
  switch (ImageDim)
  {
  case 1:
  case 2:
    //this->m_GPUKernelManager->LaunchKernel1D(m_FilterGPUKernelHandle,
    //  globalSize1D, localSize1D);
    this->m_GPUKernelManager->LaunchKernel1D(m_FilterGPUKernelHandle,
      globalSize1D);
    break;
  case 3:
    //this->m_GPUKernelManager->LaunchKernel2D(m_FilterGPUKernelHandle,
    //  globalSize2D[0], globalSize2D[1], localSize2D[0], localSize2D[1]);
    this->m_GPUKernelManager->LaunchKernel2D(m_FilterGPUKernelHandle,
      globalSize2D[0], globalSize2D[1]);
    break;
  }
}

template< class TInputImage, class TOutputImage >
void GPURecursiveGaussianImageFilter< TInputImage, TOutputImage >::PrintSelf(std::ostream & os, Indent indent) const
{
  CPUSuperclass::PrintSelf(os, indent);
  GPUSuperclass::PrintSelf(os, indent);
}

} // end namespace itk

#endif

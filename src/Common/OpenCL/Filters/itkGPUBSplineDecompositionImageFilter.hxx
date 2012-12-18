/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUBSplineDecompositionImageFilter_hxx
#define __itkGPUBSplineDecompositionImageFilter_hxx

#include "itkGPUBSplineDecompositionImageFilter.h"
#include "itkGPUCastImageFilter.h"

namespace itk
{
/**
 * ****************** Constructor ***********************
 */

template< class TInputImage, class TOutputImage >
GPUBSplineDecompositionImageFilter< TInputImage, TOutputImage >
::GPUBSplineDecompositionImageFilter()
{
  std::ostringstream defines;

  if ( TInputImage::ImageDimension > 3 || TInputImage::ImageDimension < 1 )
  {
    itkExceptionMacro( "ERROR: GPUBSplineDecompositionImageFilter supports 1/2/3D image." );
  }

  // \todo: explain this:
  if ( TInputImage::ImageDimension == 1 )
  {
    defines << "#define DIM_1\n";
  }
  else
  {
    defines << "#define DIM_" << int(TInputImage::ImageDimension - 1) << "\n";
  }

  // This is hack for now, we don't know the LOCAL_MEM_SIZE in advance usually
  // called with:
  // cl_ulong localMemSize;
  // clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong),
  // &localMemSize, 0);
  // So, you would like to ask it from m_GPUKernelManager for example:
  // m_DeviceLocalMemorySize = m_GPUKernelManager->GetDeviceLocalMemorySize();

  // local memory: 16384 bytes / 1 buffer of float = 4096
  this->m_DeviceLocalMemorySize = 4084;
  defines << "#define BUFFSIZE " << m_DeviceLocalMemorySize << "\n";
  defines << "#define BUFFPIXELTYPE float" << "\n";
  defines << "#define INPIXELTYPE ";
  GetTypenameInString( typeid( typename TInputImage::PixelType ), defines );
  defines << "#define OUTPIXELTYPE ";
  GetTypenameInString( typeid( typename TOutputImage::PixelType ), defines );

  // OpenCL kernel source
  const char *GPUSource = GPUBSplineDecompositionImageFilterKernel::GetOpenCLSource();
  // Load and create kernel
  const bool loaded = this->m_GPUKernelManager->LoadProgramFromString( GPUSource, defines.str().c_str() );
  if ( loaded )
  {
    this->m_FilterGPUKernelHandle =
      this->m_GPUKernelManager->CreateKernel( "BSplineDecompositionImageFilter" );
  }
  else
  {
    itkExceptionMacro( << "Kernel has not been loaded from:\n" << GPUSource );
  }
} // end Constructor()

/**
 * ****************** GPUGenerateData ***********************
 */

template< class TInputImage, class TOutputImage >
void
GPUBSplineDecompositionImageFilter< TInputImage, TOutputImage >
::GPUGenerateData( void )
{
  itkDebugMacro(<< "Calling GPUBSplineDecompositionImageFilter::GPUGenerateData()");

  typedef typename GPUTraits< TInputImage >::Type  GPUInputImage;
  typedef typename GPUTraits< TOutputImage >::Type GPUOutputImage;

  typename GPUInputImage::Pointer inPtr =
    dynamic_cast< GPUInputImage * >( this->ProcessObject::GetInput( 0 ) );
  typename GPUOutputImage::Pointer otPtr =
    dynamic_cast< GPUOutputImage * >( this->ProcessObject::GetOutput( 0 ) );

  // Perform the safe check
  if ( inPtr.IsNull() )
  {
    itkExceptionMacro( << "ERROR: The GPU InputImage is NULL. Filter unable to perform." );
    return;
  }
  if ( otPtr.IsNull() )
  {
    itkExceptionMacro( << "ERROR: The GPU OutputImage is NULL. Filter unable to perform." );
    return;
  }

  const typename GPUOutputImage::SizeType outSize =
    otPtr->GetLargestPossibleRegion().GetSize();
  const typename GPUInputImage::SizeType dataLength =
    inPtr->GetLargestPossibleRegion().GetSize();
  typename GPUOutputImage::SizeValueType maxLength = 0;

  for ( std::size_t n = 0; n < InputImageDimension; n++ )
  {
    if ( dataLength[n] > maxLength )
    {
      maxLength = dataLength[n];
    }
  }

  // Check if GPU filter are able to perform for this image
  if ( maxLength > this->m_DeviceLocalMemorySize )
  {
    itkExceptionMacro( << "ERROR: GPUBSplineDecompositionImageFilter unable to perform." );
    return;
  }

  // Cast here, see the same call in this->CopyImageToImage() of
  // BSplineDecompositionImageFilter::DataToCoefficientsND()
  typedef GPUCastImageFilter< GPUInputImage, GPUOutputImage > CasterType;
  typename CasterType::Pointer caster = CasterType::New();
  caster->SetInput( inPtr );
  caster->GraftOutput( otPtr );
  caster->Update();

  typename GPUInputImage::SizeType localSize, globalSize;
  for ( std::size_t i = 0; i < InputImageDimension; i++ )
  {
    localSize[i] = OpenCLGetLocalBlockSize( InputImageDimension );
    // total # of threads
    globalSize[i] = localSize[i] * ( static_cast< unsigned int >(
                                       vcl_ceil( static_cast< float >( outSize[i] )
                                                 / static_cast< float >( localSize[i] ) ) ) );
  }

  // Make GPU buffer not dirty
  otPtr->GetGPUDataManager()->SetGPUDirtyFlag( false );

  // arguments set up
  int argidx = 0;
  this->m_GPUKernelManager->SetKernelArgWithImage(
    this->m_FilterGPUKernelHandle, argidx++, inPtr->GetGPUDataManager() );
  this->m_GPUKernelManager->SetKernelArgWithImage(
    this->m_FilterGPUKernelHandle, argidx++, otPtr->GetGPUDataManager() );

  // set image size
  unsigned int imageSize[InputImageDimension];
  for ( std::size_t i = 0; i < InputImageDimension; i++ )
  {
    imageSize[i] = outSize[i];
  }

  // Solving warning "case label value exceeds maximum value for type"
  // by making a local copy of the input image dimension.
  //switch( InputImageDimension )
  const unsigned int ImageDim = (unsigned int)( InputImageDimension );
  switch ( ImageDim )
  {
    case 1:
      unsigned int imageSize1D[2];
      imageSize1D[0] = imageSize[0];
      imageSize1D[1] = 0;
      this->m_GPUKernelManager->SetKernelArg(
        this->m_FilterGPUKernelHandle, argidx++, sizeof( cl_uint2 ), &imageSize1D );
      break;
    case 2:
      this->m_GPUKernelManager->SetKernelArg(
        this->m_FilterGPUKernelHandle, argidx++, sizeof( cl_uint2 ), &imageSize );
      break;
    case 3:
      this->m_GPUKernelManager->SetKernelArg(
        this->m_FilterGPUKernelHandle, argidx++, sizeof( cl_uint3 ), &imageSize );
      break;
  }

  // Set poles calculated for a given spline order
  float spline_poles[2];
  if ( this->m_NumberOfPoles == 1 )
  {
    spline_poles[0] = static_cast< float >( this->m_SplinePoles[0] );
    spline_poles[1] = 0.0;
  }
  else if ( this->m_NumberOfPoles == 2 )
  {
    spline_poles[0] = static_cast< float >( this->m_SplinePoles[0] );
    spline_poles[1] = static_cast< float >( this->m_SplinePoles[1] );
  }
  this->m_GPUKernelManager->SetKernelArg(
    this->m_FilterGPUKernelHandle, argidx++, sizeof( cl_float2 ), &spline_poles );

  // Set m_NumberOfPoles
  this->m_GPUKernelManager->SetKernelArg(
    this->m_FilterGPUKernelHandle, argidx++, sizeof( cl_int ), &this->m_NumberOfPoles );

  // Loop over directions
  for ( std::size_t n = 0; n < InputImageDimension; n++ )
  {
    this->m_GPUKernelManager->SetKernelArg(
      this->m_FilterGPUKernelHandle, argidx, sizeof( cl_uint ), &n );

    unsigned int x, y;
    switch ( n )
    {
      case 0:
        x = 1; y = 2;
        break;
      case 1:
        x = 0; y = 2;
        break;
      case 2:
        x = 0; y = 1;
        break;
    }

    switch ( ImageDim )
    {
      case 1:
      case 2:
        this->m_GPUKernelManager->LaunchKernel(
          this->m_FilterGPUKernelHandle, OpenCLSize( globalSize[n] ), OpenCLSize( localSize[n] ) );
        break;
      case 3:
        this->m_GPUKernelManager->LaunchKernel(
          this->m_FilterGPUKernelHandle,
          OpenCLSize( globalSize[x], globalSize[y] ), OpenCLSize( localSize[n], localSize[n] ) );
        break;
    }
  } // end loop over InputImageDimension

  itkDebugMacro(<< "GPUBSplineDecompositionImageFilter::GPUGenerateData() finished");
}   // end GPUGenerateData()

/**
 * ****************** PrintSelf ***********************
 */

template< class TInputImage, class TOutputImage >
void
GPUBSplineDecompositionImageFilter< TInputImage, TOutputImage >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  CPUSuperclass::PrintSelf( os, indent );
  GPUSuperclass::PrintSelf( os, indent );
} // end PrintSelf()
} // end namespace itk

#endif /* __itkGPUBSplineDecompositionImageFilter_hxx */

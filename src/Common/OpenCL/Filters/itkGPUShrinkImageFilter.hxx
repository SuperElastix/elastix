/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUShrinkImageFilter_hxx
#define __itkGPUShrinkImageFilter_hxx

#include "itkGPUShrinkImageFilter.h"

namespace itk
{
/**
 * ****************** Constructor ***********************
 */

template< class TInputImage, class TOutputImage >
GPUShrinkImageFilter< TInputImage, TOutputImage >
::GPUShrinkImageFilter()
{
  std::ostringstream defines;
  if ( TInputImage::ImageDimension > 3 || TInputImage::ImageDimension < 1 )
  {
    itkExceptionMacro( "GPUShrinkImageFilter supports 1/2/3D image." );
  }
  defines << "#define DIM_" << int(TInputImage::ImageDimension) << "\n";

  defines << "#define INPIXELTYPE ";
  GetTypenameInString( typeid( typename TInputImage::PixelType ), defines );
  defines << "#define OUTPIXELTYPE ";
  GetTypenameInString( typeid( typename TOutputImage::PixelType ), defines );

  // OpenCL kernel source
  const char *GPUSource = GPUShrinkImageFilterKernel::GetOpenCLSource();
  // Load and create kernel
  bool loaded = this->m_GPUKernelManager->LoadProgramFromString( GPUSource, defines.str().c_str() );
  if ( loaded )
  {
    m_FilterGPUKernelHandle = this->m_GPUKernelManager->CreateKernel( "ShrinkImageFilter" );
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
GPUShrinkImageFilter< TInputImage, TOutputImage >
::GPUGenerateData( void )
{
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

  // Convert the factor for convenient multiplication
  typename TOutputImage::SizeType factorSize;
  const ShrinkFactorsType shrinkFactors = this->GetShrinkFactors();
  for ( unsigned int i = 0; i < InputImageDimension; i++ )
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
  otPtr->TransformIndexToPhysicalPoint( outputIndex, tempPoint );
  inPtr->TransformPhysicalPointToIndex( tempPoint, inputIndex );

  // Given that the size is scaled by a constant factor eq:
  // inputIndex = outputIndex * factorSize
  // is equivalent up to a fixed offset which we now compute
  OffsetValueType zeroOffset = 0;
  for ( unsigned int i = 0; i < InputImageDimension; i++ )
  {
    offsetIndex[i] = inputIndex[i] - outputIndex[i] * shrinkFactors[i];
    // It is plausible that due to small amounts of loss of numerical
    // precision that the offset is negative, this would cause sampling
    // out of out region, this is insurance against that possibility
    offsetIndex[i] = vnl_math_max( zeroOffset, offsetIndex[i] );
  }

  typename GPUOutputImage::SizeType inSize  = inPtr->GetLargestPossibleRegion().GetSize();
  typename GPUOutputImage::SizeType outSize = otPtr->GetLargestPossibleRegion().GetSize();
  size_t localSize[3], globalSize[3];
  localSize[0] = localSize[1] = localSize[2] = OpenCLGetLocalBlockSize( InputImageDimension );

  for ( unsigned int i = 0; i < InputImageDimension; ++i )
  {
    // total # of threads
    globalSize[i] = localSize[i] * ( static_cast< unsigned int >(
                                       vcl_ceil( static_cast< float >( outSize[i] )
                                                 / static_cast< float >( localSize[i] ) ) ) );
  }

  // arguments set up
  int argidx = 0;
  this->m_GPUKernelManager->SetKernelArgWithImage(
    this->m_FilterGPUKernelHandle, argidx++, inPtr->GetGPUDataManager() );
  this->m_GPUKernelManager->SetKernelArgWithImage(
    this->m_FilterGPUKernelHandle, argidx++, otPtr->GetGPUDataManager() );

  // set arguments for image size/offset/shrinkfactors
  unsigned int inImageSize[InputImageDimension];
  unsigned int outImageSize[InputImageDimension];
  for ( unsigned int i = 0; i < InputImageDimension; i++ )
  {
    inImageSize[i]  = inSize[i];
    outImageSize[i] = outSize[i];
  }

  //signed long offset[TInputImage::ImageDimension];
  //unsigned long factorsize[TInputImage::ImageDimension];
  unsigned int offset[InputImageDimension];
  unsigned int shrinkfactors[InputImageDimension];

  for ( unsigned int i = 0; i < InputImageDimension; i++ )
  {
    offset[i] = offsetIndex[i];
    shrinkfactors[i] = factorSize[i];
  }

  const unsigned int ImageDim = static_cast< unsigned int >( InputImageDimension );
  switch ( ImageDim )
  {
    case 1:
      this->m_GPUKernelManager->SetKernelArg(
        this->m_FilterGPUKernelHandle, argidx++, sizeof( cl_uint ), &inImageSize );
      this->m_GPUKernelManager->SetKernelArg(
        this->m_FilterGPUKernelHandle, argidx++, sizeof( cl_uint ), &outImageSize );
      this->m_GPUKernelManager->SetKernelArg(
        this->m_FilterGPUKernelHandle, argidx++, sizeof( cl_uint ), &offset );
      this->m_GPUKernelManager->SetKernelArg(
        this->m_FilterGPUKernelHandle, argidx++, sizeof( cl_uint ), &shrinkfactors );
      break;
    case 2:
      this->m_GPUKernelManager->SetKernelArg(
        this->m_FilterGPUKernelHandle, argidx++, sizeof( cl_uint2 ), &inImageSize );
      this->m_GPUKernelManager->SetKernelArg(
        this->m_FilterGPUKernelHandle, argidx++, sizeof( cl_uint2 ), &outImageSize );
      this->m_GPUKernelManager->SetKernelArg(
        this->m_FilterGPUKernelHandle, argidx++, sizeof( cl_uint2 ), &offset );
      this->m_GPUKernelManager->SetKernelArg(
        this->m_FilterGPUKernelHandle, argidx++, sizeof( cl_uint2 ), &shrinkfactors );
      break;
    case 3:
      this->m_GPUKernelManager->SetKernelArg(
        this->m_FilterGPUKernelHandle, argidx++, sizeof( cl_uint3 ), &inImageSize );
      this->m_GPUKernelManager->SetKernelArg(
        this->m_FilterGPUKernelHandle, argidx++, sizeof( cl_uint3 ), &outImageSize );
      this->m_GPUKernelManager->SetKernelArg(
        this->m_FilterGPUKernelHandle, argidx++, sizeof( cl_uint3 ), &offset );
      this->m_GPUKernelManager->SetKernelArg(
        this->m_FilterGPUKernelHandle, argidx++, sizeof( cl_uint3 ), &shrinkfactors );
      break;
  }

  // launch kernel
  this->m_GPUKernelManager->LaunchKernel(
    this->m_FilterGPUKernelHandle,
    InputImageDimension, globalSize, localSize );
} // end GPUGenerateData()

/**
 * ****************** PrintSelf ***********************
 */

template< class TInputImage, class TOutputImage >
void
GPUShrinkImageFilter< TInputImage, TOutputImage >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  CPUSuperclass::PrintSelf( os, indent );
  GPUSuperclass::PrintSelf( os, indent );
} // end PrintSelf()

} // end namespace itk

#endif /* __itkGPUShrinkImageFilter_hxx */

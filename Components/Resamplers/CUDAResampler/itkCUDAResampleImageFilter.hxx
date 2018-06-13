/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkCUDAResamplerImageFilter_txx
#define __itkCUDAResamplerImageFilter_txx

#include <cuda_runtime.h>
#include "itkCUDAResampleImageFilter.h"
#include "itkBSplineInterpolateImageFunction.h"

namespace itk
{

/**
 * ******************* Constructor ***********************
 */

template< typename TInputImage, typename TOutputImage, typename TInterpolatorPrecisionType >
itkCUDAResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
::itkCUDAResampleImageFilter()
{
  this->m_UseCuda           = true;
  this->m_UseGPUToCastData  = false;
  this->m_UseFastCUDAKernel = false; // accurate by default

} // end Constructor


/**
 * ******************* Destructor ***********************
 */

template< typename TInputImage, typename TOutputImage, typename TInterpolatorPrecisionType >
itkCUDAResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
::~itkCUDAResampleImageFilter()
{
  if( this->m_UseCuda )
  {
    this->m_CudaResampleImageFilter.cudaUnInit();
  }
}


/**
 * ******************* CopyParameters ***********************
 */

template< typename TInputImage, typename TOutputImage, typename TInterpolatorPrecisionType >
void
itkCUDAResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
::CopyParameters( ValidTransformPointer bSplineTransform )
{
  /* Copy parameters to the GPU memory space. */
  const SizeType        itkOutputSize    = this->GetSize();
  const SizeType        itkInputSize     = this->GetInput()->GetLargestPossibleRegion().GetSize();
  const SpacingType     itkOutputSpacing = this->GetOutputSpacing();
  const OriginPointType itkOutputOrigin  = this->GetOutputOrigin();
  const SpacingType     itkInputSpacing  = this->GetInput()->GetSpacing();
  const OriginPointType itkInputOrigin   = this->GetInput()->GetOrigin();

  /** Copy the input image data. */
  uint3 inputSize = make_uint3(
    itkInputSize[ 0 ],  itkInputSize[ 1 ],  itkInputSize[ 2 ] );
  uint3 outputSize = make_uint3(
    itkOutputSize[ 0 ], itkOutputSize[ 1 ], itkOutputSize[ 2 ] );
  const InputPixelType * data = this->GetInput()->GetBufferPointer();
  this->m_CudaResampleImageFilter.cudaMallocImageData( inputSize, outputSize, data );

  /** Copy output image information. */
  float3 outputImageSpacing = make_float3(
    itkOutputSpacing[ 0 ], itkOutputSpacing[ 1 ], itkOutputSpacing[ 2 ] );
  float3 outputImageOrigin = make_float3(
    itkOutputOrigin[ 0 ],  itkOutputOrigin[ 1 ],  itkOutputOrigin[ 2 ] );
  float3 inputImageSpacing = make_float3(
    itkInputSpacing[ 0 ],  itkInputSpacing[ 1 ],  itkInputSpacing[ 2 ] );
  float3 inputImageOrigin = make_float3(
    itkInputOrigin[ 0 ],   itkInputOrigin[ 1 ],   itkInputOrigin[ 2 ] );
  float defaultPixelValue = this->GetDefaultPixelValue();
  this->m_CudaResampleImageFilter.cudaCopyImageSymbols(
    inputImageSpacing, inputImageOrigin,
    outputImageSpacing, outputImageOrigin, defaultPixelValue );

  /** Copy B-spline grid data. */
  const typename InternalBSplineTransformType::OriginType itkGridOrigin
    = bSplineTransform->GetGridOrigin();
  const typename InternalBSplineTransformType::SpacingType itkGridSpacing
    = bSplineTransform->GetGridSpacing();
  const typename InternalBSplineTransformType::SizeType itkGridSize
    = bSplineTransform->GetGridRegion().GetSize();
  float3 gridSpacing = make_float3( itkGridSpacing[ 0 ], itkGridSpacing[ 1 ], itkGridSpacing[ 2 ] );
  float3 gridOrigin  = make_float3( itkGridOrigin[ 0 ],  itkGridOrigin[ 1 ],  itkGridOrigin[ 2 ] );
  uint3  gridSize    = make_uint3( itkGridSize[ 0 ],    itkGridSize[ 1 ],    itkGridSize[ 2 ] );
  this->m_CudaResampleImageFilter.cudaCopyGridSymbols( gridSpacing, gridOrigin, gridSize );

  /** Copy B-spline parameters. */
  const typename InternalBSplineTransformType::ParametersType params
    = bSplineTransform->GetParameters();
  this->m_CudaResampleImageFilter.cudaMallocTransformationData(
    gridSize, params.data_block() );

} // end CopyParameters()


/**
 * ******************* CheckForValidTransform ***********************
 */

template< typename TInputImage, typename TOutputImage, typename TInterpolatorPrecisionType >
bool
itkCUDAResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
::CheckForValidTransform( ValidTransformPointer & bSplineTransform ) const
{
  /** First check if the Transform is valid for CUDA. */
  typename InternalBSplineTransformType::Pointer testPtr1a
    = const_cast< InternalBSplineTransformType * >(
    dynamic_cast< const InternalBSplineTransformType * >( this->GetTransform() ) );
  typename InternalAdvancedBSplineTransformType::Pointer testPtr1b
    = const_cast< InternalAdvancedBSplineTransformType * >(
    dynamic_cast< const InternalAdvancedBSplineTransformType * >( this->GetTransform() ) );
  typename InternalComboTransformType::Pointer testPtr2a
    = const_cast< InternalComboTransformType * >(
    dynamic_cast< const InternalComboTransformType * >( this->GetTransform() ) );

  bool transformIsValid = false;
  if( testPtr1a )
  {
    /** The transform is of type BSplineDeformableTransform. */
    //transformIsValid = true;
    transformIsValid = false; // \todo: not yet supported
    //bSplineTransform = testPtr1a;
  }
  else if( testPtr1b )
  {
    /** The transform is of type AdvancedBSplineDeformableTransform. */
    transformIsValid = true;
    bSplineTransform = testPtr1b;
  }
  else if( testPtr2a )
  {
    // Check that the comboT has no initial transform and that current = B-spline
    // and that B-spline = 3rd order

    /** The transform is of type AdvancedCombinationTransform. */
    if( !testPtr2a->GetInitialTransform() )
    {
      typename InternalAdvancedBSplineTransformType::Pointer testPtr2b
        = dynamic_cast< InternalAdvancedBSplineTransformType * >(
        testPtr2a->GetCurrentTransform() );
      if( testPtr2b )
      {
        /** The current transform is of type AdvancedBSplineDeformableTransform. */
        transformIsValid = true;
        bSplineTransform = testPtr2b;
      }
    }
  } // end if combo transform

  return transformIsValid;

} // end CheckForValidTransform()


/**
 * ******************* CheckForValidInterpolator ***********************
 */

template< typename TInputImage, typename TOutputImage, typename TInterpolatorPrecisionType >
bool
itkCUDAResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
::CheckForValidInterpolator( void ) const
{
  /** Check if the interpolator is valid for CUDA. */
  // ImageType = ElastixType::MovingImageType = InputImageType
  // CoordRepType = ElastixType::CoordRepType = TInterpolatorPrecisionType
  // CoefficientType = float or double, does not matter
  typedef BSplineInterpolateImageFunction<
    InputImageType, TInterpolatorPrecisionType, float >   ValidInterpolatorFloatType;
  typedef BSplineInterpolateImageFunction<
    InputImageType, TInterpolatorPrecisionType, double >  ValidInterpolatorDoubleType;

  typename ValidInterpolatorFloatType::Pointer testPtr1
    = const_cast< ValidInterpolatorFloatType * >(
    dynamic_cast< const ValidInterpolatorFloatType * >( this->GetInterpolator() ) );
  typename ValidInterpolatorDoubleType::Pointer testPtr2
    = const_cast< ValidInterpolatorDoubleType * >(
    dynamic_cast< const ValidInterpolatorDoubleType * >( this->GetInterpolator() ) );

  bool interpolatorIsValid = false;
  if( testPtr1 )
  {
    if( testPtr1->GetSplineOrder() == 3 )
    {
      interpolatorIsValid = true;
    }
  }
  else if( testPtr2 )
  {
    if( testPtr2->GetSplineOrder() == 3 )
    {
      interpolatorIsValid = true;
    }
  }

  return interpolatorIsValid;

} // end CheckForValidInterpolator()


/**
 * ******************* CheckForValidDirectionCosines ***********************
 */

template< typename TInputImage, typename TOutputImage, typename TInterpolatorPrecisionType >
bool
itkCUDAResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
::CheckForValidDirectionCosines( ValidTransformPointer bSplineTransform ) // const
{
  /** Check if the direction cosines are valid for CUDA. */
  bool          directionCosinesAreValid = true;
  DirectionType identityDC; identityDC.SetIdentity();
  typedef typename InternalAdvancedBSplineTransformType::DirectionType GridDirectionType;
  GridDirectionType identityDCGrid; identityDCGrid.SetIdentity();

  /** Check input image direction cosines. */
  DirectionType inputImageDC = this->GetInput()->GetDirection();
  if( inputImageDC != identityDC )
  {
    directionCosinesAreValid = false;
  }

  /** Check output image direction cosines. */
  DirectionType outputImageDC = this->GetOutputDirection();
  if( outputImageDC != identityDC )
  {
    directionCosinesAreValid = false;
  }

  /** Check B-spline grid direction cosines. */
  GridDirectionType bsplineGridDC = bSplineTransform->GetGridDirection();
  if( bsplineGridDC != identityDCGrid )
  {
    directionCosinesAreValid = false;
  }

  return directionCosinesAreValid;

} // end CheckForValidDirectionCosines()


/**
 * ******************* CheckForValidConfiguration ***********************
 */

template< typename TInputImage, typename TOutputImage, typename TInterpolatorPrecisionType >
void
itkCUDAResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
::CheckForValidConfiguration( ValidTransformPointer & bSplineTransform ) // const
{
  try // why try/catch?
  {
    this->m_WarningReport.ResetWarningReport();

    /** Check for valid transform: 3rd order B-spline, no initial transform, dimension 3. */
    bool transformIsValid = this->CheckForValidTransform( bSplineTransform );
    if( !transformIsValid )
    {
      this->m_UseCuda = false;
      std::string message = "WARNING: No valid transform set:\n"
        + std::string( "The transform should be 3rd order B-spline, 3D image, no initial transform.\n" )
        + std::string( "Falling back to CPU implementation." );
      this->m_WarningReport.m_Warnings.push_back( message );
    }

    /** Check for valid interpolator: 3rd order B-spline. */
    bool interpolatorIsValid = this->CheckForValidInterpolator();
    if( !interpolatorIsValid )
    {
      this->m_UseCuda = false;
      std::string message = "WARNING: No valid interpolator set:\n"
        + std::string( "The interpolator should be 3rd order B-spline, 3D image\n" )
        + std::string( "Falling back to CPU implementation." );
      this->m_WarningReport.m_Warnings.push_back( message );
    }

    /** Check for identity cosines. */
    if( transformIsValid )
    {
      bool directionCosinesAreValid = this->CheckForValidDirectionCosines( bSplineTransform );
      if( !directionCosinesAreValid )
      {
        this->m_UseCuda = false;
        std::string message = "WARNING: No valid direction cosines:\n"
          + std::string( "The input image, output image, and B-spline grid direction should all be the identity.\n" )
          + std::string( "Falling back to CPU implementation." );
        this->m_WarningReport.m_Warnings.push_back( message );
      }
    }

    /** Check if proper CUDA device. */
    bool cuda_device = ( CudaResampleImageFilterType::checkExecutionParameters() == 0 );
    if( !cuda_device )
    {
      this->m_UseCuda = false;
      std::string message = "WARNING: No valid GPU found:\n"
        + std::string( "The GPU should support CUDA, and the driver should be up-to-date.\n" )
        + std::string( "Falling back to CPU implementation." );
      this->m_WarningReport.m_Warnings.push_back( message );
    }
  }
  catch( itk::ExceptionObject & excep )
  {
    // FIXME: no printing
    std::cerr << excep << std::endl;
    this->m_UseCuda = false;
  }

} // end CheckForValidConfiguration()


/**
 * ******************* GenerateData ***********************
 */

template< typename TInputImage, typename TOutputImage, typename TInterpolatorPrecisionType >
void
itkCUDAResampleImageFilter< TInputImage, TOutputImage, TInterpolatorPrecisionType >
::GenerateData( void )
{
  /** If we are not using CUDA simply use the CPU implementation. */
  if( !this->m_UseCuda )
  {
    return Superclass::GenerateData();
  }

  /** Checks! */
  ValidTransformPointer tempTransform = NULL;
  this->CheckForValidConfiguration( tempTransform );

  /** The GPU can't be used. Use CPU instead. */
  if( !this->m_UseCuda )
  {
    return this->Superclass::GenerateData();
  }

  /** Initialize CUDA device. */
  this->m_CudaResampleImageFilter.cudaInit();
  this->m_CudaResampleImageFilter.SetCastOnGPU(
    this->m_UseGPUToCastData );
  this->m_CudaResampleImageFilter.SetUseFastCUDAKernel(
    this->m_UseFastCUDAKernel );

  /** Copy the parameters to the GPU. */
  this->CopyParameters( tempTransform );

  /** Allocate host memory for the output and copy/cast the result back to the host. */
  this->AllocateOutputs();
  InputPixelType * data = this->GetOutput()->GetBufferPointer();

  /** Run the CUDA resampler. */
  this->m_CudaResampleImageFilter.GenerateData( data );

  /** Release the GPU memory. */
  this->m_CudaResampleImageFilter.cudaUnInit();

} // end GenerateData()


}  // end namespace itk

#endif // end #ifndef __itkCUDAResamplerImageFilter_txx

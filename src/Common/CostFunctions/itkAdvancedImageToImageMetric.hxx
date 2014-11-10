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
#ifndef _itkAdvancedImageToImageMetric_hxx
#define _itkAdvancedImageToImageMetric_hxx

#include "itkAdvancedImageToImageMetric.h"

#include "itkImageRegionConstIterator.h"          // used for extrema computation
#include "itkImageRegionConstIteratorWithIndex.h" // used for extrema computation
#include "itkAdvancedRayCastInterpolateImageFunction.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{

/**
 * ********************* Constructor ****************************
 */

template< class TFixedImage, class TMovingImage >
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::AdvancedImageToImageMetric()
{
  /** don't use the default gradient image as implemented by ITK.
   * It uses a Gaussian derivative, which introduces extra smoothing,
   * which may not always be desired. Also, when the derivatives are
   * computed using Gaussian filtering, the gray-values should also be
   * blurred, to have a consistent 'image model'.
   */
  this->SetComputeGradient( false );

  this->m_ImageSampler                = 0;
  this->m_UseImageSampler             = false;
  this->m_RequiredRatioOfValidSamples = 0.25;

  this->m_BSplineInterpolator             = 0;
  this->m_BSplineInterpolatorFloat        = 0;
  this->m_ReducedBSplineInterpolator      = 0;
  this->m_LinearInterpolator              = 0;
  this->m_InterpolatorIsBSpline           = false;
  this->m_InterpolatorIsBSplineFloat      = false;
  this->m_InterpolatorIsReducedBSpline    = false;
  this->m_InterpolatorIsLinear            = false;
  this->m_CentralDifferenceGradientFilter = 0;

  this->m_AdvancedTransform              = 0;
  this->m_TransformIsAdvanced            = false;
  this->m_TransformIsBSpline             = false;
  this->m_UseMovingImageDerivativeScales = false;
  this->m_MovingImageDerivativeScales.Fill( 1.0 );

  this->m_FixedImageLimiter     = 0;
  this->m_MovingImageLimiter    = 0;
  this->m_UseFixedImageLimiter  = false;
  this->m_UseMovingImageLimiter = false;
  this->m_FixedLimitRangeRatio  = 0.01;
  this->m_MovingLimitRangeRatio = 0.01;
  this->m_FixedImageTrueMin     = NumericTraits< FixedImagePixelType  >::Zero;
  this->m_FixedImageTrueMax     = NumericTraits< FixedImagePixelType  >::One;
  this->m_MovingImageTrueMin    = NumericTraits< MovingImagePixelType >::Zero;
  this->m_MovingImageTrueMax    = NumericTraits< MovingImagePixelType >::One;
  this->m_FixedImageMinLimit    = NumericTraits< FixedImageLimiterOutputType  >::Zero;
  this->m_FixedImageMaxLimit    = NumericTraits< FixedImageLimiterOutputType  >::One;
  this->m_MovingImageMinLimit   = NumericTraits< MovingImageLimiterOutputType >::Zero;
  this->m_MovingImageMaxLimit   = NumericTraits< MovingImageLimiterOutputType >::One;

  /** Threading related variables. */
  this->m_UseMetricSingleThreaded = true;

  /** OpenMP related. Switch to on when available */
#ifdef ELASTIX_USE_OPENMP
  this->m_UseOpenMP = true;

  const int nthreads = static_cast< int >( this->m_NumberOfThreads );
  omp_set_num_threads( nthreads );
#else
  this->m_UseOpenMP = false;
#endif

  /** Initialize the m_ThreaderMetricParameters. */
  this->m_ThreaderMetricParameters.st_Metric = this;

  // Multi-threading structs
  this->m_GetValueAndDerivativePerThreadVariables     = NULL;
  this->m_GetValueAndDerivativePerThreadVariablesSize = 0;

} // end Constructor


/**
 * ********************* Destructor ****************************
 */

template< class TFixedImage, class TMovingImage >
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::~AdvancedImageToImageMetric()
{
  delete[] this->m_GetValueAndDerivativePerThreadVariables;
} // end Destructor


/**
 * ********************* SetNumberOfThreads ****************************
 */

template< class TFixedImage, class TMovingImage >
void
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::SetNumberOfThreads( ThreadIdType numberOfThreads )
{
  Superclass::SetNumberOfThreads( numberOfThreads );

#ifdef ELASTIX_USE_OPENMP
  const int nthreads = static_cast< int >( this->m_NumberOfThreads );
  omp_set_num_threads( nthreads );
#endif
} // end SetNumberOfThreads()


/**
 * ********************* Initialize ****************************
 */

template< class TFixedImage, class TMovingImage >
void
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::Initialize( void ) throw ( ExceptionObject )
{
  /** Initialize transform, interpolator, etc. */
  Superclass::Initialize();

  /** Setup the parameters for the gray value limiters. */
  this->InitializeLimiters();

  /** Connect the image sampler */
  this->InitializeImageSampler();

  /** Check if the interpolator is a B-spline interpolator. */
  this->CheckForBSplineInterpolator();

  /** Check if the transform is an advanced transform. */
  this->CheckForAdvancedTransform();

  /** Check if the transform is a B-spline transform. */
  this->CheckForBSplineTransform();

  /** Initialize some threading related parameters. */
  if( this->m_UseMultiThread )
  {
    this->InitializeThreadingParameters();
  }

} // end Initialize()


/**
 * ********************* InitializeThreadingParameters ****************************
 */

template< class TFixedImage, class TMovingImage >
void
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::InitializeThreadingParameters( void ) const
{
  /** Resize and initialize the threading related parameters.
   * The SetSize() functions do not resize the data when this is not
   * needed, which saves valuable re-allocation time.
   *
   * This function is only to be called at the start of each resolution.
   * Re-initialization of the potentially large vectors is performed after
   * each iteration, in the accumulate functions, in a multi-threaded fashion.
   * This has performance benefits for larger vector sizes.
   */

  /** Only resize the array of structs when needed. */
  if( this->m_GetValueAndDerivativePerThreadVariablesSize != this->m_NumberOfThreads )
  {
    delete[] this->m_GetValueAndDerivativePerThreadVariables;
    this->m_GetValueAndDerivativePerThreadVariables     = new AlignedGetValueAndDerivativePerThreadStruct[ this->m_NumberOfThreads ];
    this->m_GetValueAndDerivativePerThreadVariablesSize = this->m_NumberOfThreads;
  }

  /** Some initialization. */
  for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
  {
    this->m_GetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCounted = NumericTraits< SizeValueType >::Zero;
    this->m_GetValueAndDerivativePerThreadVariables[ i ].st_Value                 = NumericTraits< MeasureType >::Zero;
    this->m_GetValueAndDerivativePerThreadVariables[ i ].st_Derivative.SetSize( this->GetNumberOfParameters() );
    this->m_GetValueAndDerivativePerThreadVariables[ i ].st_Derivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );
  }

} // end InitializeThreadingParameters()


/**
 * ****************** ComputeFixedImageExtrema ***************************
 */

template< class TFixedImage, class TMovingImage >
void
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::ComputeFixedImageExtrema(
  const FixedImageType * image,
  const FixedImageRegionType & region )
{
  /** NB: We can't use StatisticsImageFilterWithMask to do this because
   * the filter computes the min/max for the largest possible region.
   * This filter is multi-threaded though.
   */
  FixedImagePixelType trueMinTemp = NumericTraits< FixedImagePixelType >::max();
  FixedImagePixelType trueMaxTemp = NumericTraits< FixedImagePixelType >::NonpositiveMin();

  /** If no mask. */
  if( this->m_FixedImageMask.IsNull() )
  {
    typedef ImageRegionConstIterator< FixedImageType > IteratorType;
    IteratorType it( image, region );
    for( it.GoToBegin(); !it.IsAtEnd(); ++it )
    {
      const FixedImagePixelType sample = it.Get();
      trueMinTemp = vnl_math_min( trueMinTemp, sample );
      trueMaxTemp = vnl_math_max( trueMaxTemp, sample );
    }
  }
  /** Excluded extrema outside the mask.
   * Because we have to call TransformIndexToPhysicalPoint() and
   * check IsInside() this way is much (!) slower.
   */
  else
  {
    typedef ImageRegionConstIteratorWithIndex< FixedImageType > IteratorType;
    IteratorType it( image, region );

    for( it.GoToBegin(); !it.IsAtEnd(); ++it )
    {
      OutputPointType point;
      image->TransformIndexToPhysicalPoint( it.GetIndex(), point );
      if( this->m_FixedImageMask->IsInside( point ) )
      {
        const FixedImagePixelType sample = it.Get();
        trueMinTemp = vnl_math_min( trueMinTemp, sample );
        trueMaxTemp = vnl_math_max( trueMaxTemp, sample );
      }
    }
  }

  /** Update member variables. */
  this->m_FixedImageTrueMin = trueMinTemp;
  this->m_FixedImageTrueMax = trueMaxTemp;

  this->m_FixedImageMinLimit = static_cast< FixedImageLimiterOutputType >(
    trueMinTemp - this->m_FixedLimitRangeRatio * ( trueMaxTemp - trueMinTemp ) );
  this->m_FixedImageMaxLimit = static_cast< FixedImageLimiterOutputType >(
    trueMaxTemp + this->m_FixedLimitRangeRatio * ( trueMaxTemp - trueMinTemp ) );

} // end ComputeFixedImageExtrema()


/**
 * ****************** ComputeMovingImageExtrema ***************************
 */

template< class TFixedImage, class TMovingImage >
void
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::ComputeMovingImageExtrema(
  const MovingImageType * image,
  const MovingImageRegionType & region )
{
  /** NB: We can't use StatisticsImageFilter to do this because
   * the filter computes the min/max for the largest possible region.
   */
  MovingImagePixelType trueMinTemp = NumericTraits< MovingImagePixelType >::max();
  MovingImagePixelType trueMaxTemp = NumericTraits< MovingImagePixelType >::NonpositiveMin();

  /** If no mask. */
  if( this->m_MovingImageMask.IsNull() )
  {
    typedef ImageRegionConstIterator< MovingImageType > IteratorType;
    IteratorType iterator( image, region );
    for( iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator )
    {
      const MovingImagePixelType sample = iterator.Get();
      trueMinTemp = vnl_math_min( trueMinTemp, sample );
      trueMaxTemp = vnl_math_max( trueMaxTemp, sample );
    }
  }
  /** Excluded extrema outside the mask.
   * Because we have to call TransformIndexToPhysicalPoint() and
   * check IsInside() this way is much (!) slower.
   */
  else
  {
    typedef ImageRegionConstIteratorWithIndex< MovingImageType > IteratorType;
    IteratorType it( image, region );

    for( it.GoToBegin(); !it.IsAtEnd(); ++it )
    {
      OutputPointType point;
      image->TransformIndexToPhysicalPoint( it.GetIndex(), point );
      if( this->m_MovingImageMask->IsInside( point ) )
      {
        const MovingImagePixelType sample = it.Get();
        trueMinTemp = vnl_math_min( trueMinTemp, sample );
        trueMaxTemp = vnl_math_max( trueMaxTemp, sample );
      }
    }
  }

  /** Update member variables. */
  this->m_MovingImageTrueMin = trueMinTemp;
  this->m_MovingImageTrueMax = trueMaxTemp;

  this->m_MovingImageMinLimit = static_cast< MovingImageLimiterOutputType >(
    trueMinTemp - this->m_MovingLimitRangeRatio * ( trueMaxTemp - trueMinTemp ) );
  this->m_MovingImageMaxLimit = static_cast< MovingImageLimiterOutputType >(
    trueMaxTemp + this->m_MovingLimitRangeRatio * ( trueMaxTemp - trueMinTemp ) );

} // end ComputeMovingImageExtrema()


/**
 * ****************** InitializeLimiter *****************************
 */

template< class TFixedImage, class TMovingImage >
void
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::InitializeLimiters( void )
{
  /** Set up fixed limiter. */
  if( this->GetUseFixedImageLimiter() )
  {
    if( this->GetFixedImageLimiter() == 0 )
    {
      itkExceptionMacro( << "No fixed image limiter has been set!" );
    }

    this->ComputeFixedImageExtrema(
      this->GetFixedImage(),
      this->GetFixedImageRegion() );

    this->m_FixedImageLimiter->SetLowerThreshold(
      static_cast< RealType >( this->m_FixedImageTrueMin ) );
    this->m_FixedImageLimiter->SetUpperThreshold(
      static_cast< RealType >( this->m_FixedImageTrueMax ) );
    this->m_FixedImageLimiter->SetLowerBound( this->m_FixedImageMinLimit );
    this->m_FixedImageLimiter->SetUpperBound( this->m_FixedImageMaxLimit );

    this->m_FixedImageLimiter->Initialize();
  }

  /** Set up moving limiter. */
  if( this->GetUseMovingImageLimiter() )
  {
    if( this->GetMovingImageLimiter() == 0 )
    {
      itkExceptionMacro( << "No moving image limiter has been set!" );
    }

    this->ComputeMovingImageExtrema(
      this->GetMovingImage(),
      this->GetMovingImage()->GetBufferedRegion() );

    this->m_MovingImageLimiter->SetLowerThreshold(
      static_cast< RealType >( this->m_MovingImageTrueMin ) );
    this->m_MovingImageLimiter->SetUpperThreshold(
      static_cast< RealType >( this->m_MovingImageTrueMax ) );
    this->m_MovingImageLimiter->SetLowerBound( this->m_MovingImageMinLimit );
    this->m_MovingImageLimiter->SetUpperBound( this->m_MovingImageMaxLimit );

    this->m_MovingImageLimiter->Initialize();
  }

} // end InitializeLimiter()


/**
 * ********************* InitializeImageSampler ****************************
 */

template< class TFixedImage, class TMovingImage >
void
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::InitializeImageSampler( void ) throw ( ExceptionObject )
{
  if( this->GetUseImageSampler() )
  {
    /** Check if the ImageSampler is set. */
    if( !this->m_ImageSampler )
    {
      itkExceptionMacro( << "ImageSampler is not present" );
    }

    /** Initialize the Image Sampler. */
    this->m_ImageSampler->SetInput( this->m_FixedImage );
    this->m_ImageSampler->SetMask( this->m_FixedImageMask );
    this->m_ImageSampler->SetInputImageRegion( this->GetFixedImageRegion() );
  }

} // end InitializeImageSampler()


/**
 * ****************** CheckForBSplineInterpolator **********************
 */

template< class TFixedImage, class TMovingImage >
void
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::CheckForBSplineInterpolator( void )
{
  /** Check if the interpolator is of type BSplineInterpolateImageFunction,
   * or of type AdvancedLinearInterpolateImageFunction.
   * If so, we can make use of their EvaluateDerivatives methods.
   * Otherwise, we precompute the gradients using a central difference scheme,
   * and do evaluate the gradient using nearest neighbor interpolation.
   */
  this->m_InterpolatorIsBSpline = false;
  BSplineInterpolatorType * testPtr
    = dynamic_cast< BSplineInterpolatorType * >( this->m_Interpolator.GetPointer() );
  if( testPtr )
  {
    this->m_InterpolatorIsBSpline = true;
    this->m_BSplineInterpolator   = testPtr;
    itkDebugMacro( "Interpolator is B-spline" );
  }
  else
  {
    this->m_BSplineInterpolator = 0;
    itkDebugMacro( "Interpolator is not B-spline" );
  }

  this->m_InterpolatorIsBSplineFloat = false;
  BSplineInterpolatorFloatType * testPtr2
    = dynamic_cast< BSplineInterpolatorFloatType * >( this->m_Interpolator.GetPointer() );
  if( testPtr2 )
  {
    this->m_InterpolatorIsBSplineFloat = true;
    this->m_BSplineInterpolatorFloat   = testPtr2;
    itkDebugMacro( "Interpolator is BSplineFloat" );
  }
  else
  {
    this->m_BSplineInterpolatorFloat = 0;
    itkDebugMacro( "Interpolator is not BSplineFloat" );
  }

  this->m_InterpolatorIsReducedBSpline = false;
  ReducedBSplineInterpolatorType * testPtr3
    = dynamic_cast< ReducedBSplineInterpolatorType * >( this->m_Interpolator.GetPointer() );
  if( testPtr3 )
  {
    this->m_InterpolatorIsReducedBSpline = true;
    this->m_ReducedBSplineInterpolator   = testPtr3;
    itkDebugMacro( "Interpolator is ReducedBSpline" );
  }
  else
  {
    this->m_ReducedBSplineInterpolator = 0;
    itkDebugMacro( "Interpolator is not ReducedBSpline" );
  }

  this->m_InterpolatorIsLinear = false;
  LinearInterpolatorType * testPtr4
    = dynamic_cast< LinearInterpolatorType * >( this->m_Interpolator.GetPointer() );
  if( testPtr4 )
  {
    this->m_InterpolatorIsLinear = true;
    this->m_LinearInterpolator   = testPtr4;
  }
  else
  {
    this->m_LinearInterpolator = 0;
  }

  /** Don't overwrite the gradient image if GetComputeGradient() == true.
   * Otherwise we can use a forward difference derivative, or the derivative
   * provided by the B-spline interpolator.
   */
  if( !this->GetComputeGradient() )
  {
    /** In addition, don't compute the moving image gradient for 2D/3D registration,
     * i.e. whenever the interpolator is a ray cast interpolator.
     * This is a bit of a hack that does not respect the setting of the boolean
     * m_ComputeGradient. By doing this, there is no way to ask no gradient
     * computation at all (to save memory).
     * The best solution would be to remove everything below this point, and to
     * override the ComputeGradient() function of ITK by computing a central
     * difference derivative. This way SetComputeGradient will enable or disable
     * the gradient computation and let derived classes choose if it needs the
     * precomputation of the gradient.
     *
     * For more details see the post about "2D/3D registration memory issue" in
     * elastix's mailing list (2 July 2012).
     */
    typedef itk::AdvancedRayCastInterpolateImageFunction<
      MovingImageType, CoordinateRepresentationType >       RayCastInterpolatorType;
    const bool interpolatorIsRayCast
      = dynamic_cast< RayCastInterpolatorType * >( this->m_Interpolator.GetPointer() ) != 0;

    if( !this->m_InterpolatorIsBSpline && !this->m_InterpolatorIsBSplineFloat
      && !this->m_InterpolatorIsReducedBSpline
      && !this->m_InterpolatorIsLinear
      && !interpolatorIsRayCast )
    {
      this->m_CentralDifferenceGradientFilter = CentralDifferenceGradientFilterType::New();
      this->m_CentralDifferenceGradientFilter->SetUseImageSpacing( true );
      this->m_CentralDifferenceGradientFilter->SetInput( this->m_MovingImage );
      this->m_CentralDifferenceGradientFilter->Update();
      this->m_GradientImage = this->m_CentralDifferenceGradientFilter->GetOutput();
    }
    else
    {
      this->m_CentralDifferenceGradientFilter = 0;
      this->m_GradientImage                   = 0;
    }
  }

} // end CheckForBSplineInterpolator()


/**
 * ****************** CheckForAdvancedTransform **********************
 */

template< class TFixedImage, class TMovingImage >
void
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::CheckForAdvancedTransform( void )
{
  /** Check if the transform is of type AdvancedTransform. */
  this->m_TransformIsAdvanced = false;
  AdvancedTransformType * testPtr
    = dynamic_cast< AdvancedTransformType * >(
    this->m_Transform.GetPointer() );
  if( !testPtr )
  {
    this->m_AdvancedTransform = 0;
    itkDebugMacro( "Transform is not Advanced" );
    itkExceptionMacro( << "The AdvancedImageToImageMetric requires an AdvancedTransform" );
  }
  else
  {
    this->m_TransformIsAdvanced = true;
    this->m_AdvancedTransform   = testPtr;
    itkDebugMacro( "Transform is Advanced" );
  }

} // end CheckForAdvancedTransform()


/**
 * ****************** CheckForBSplineTransform **********************
 */

template< class TFixedImage, class TMovingImage >
void
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::CheckForBSplineTransform( void )
{
  /** Check if this transform is a combo transform. */
  CombinationTransformType * testPtr_combo
    = dynamic_cast< CombinationTransformType * >( this->m_AdvancedTransform.GetPointer() );

  /** Check if this transform is a B-spline transform. */
  BSplineOrder1TransformType * testPtr_1
    = dynamic_cast< BSplineOrder1TransformType * >( this->m_AdvancedTransform.GetPointer() );
  BSplineOrder2TransformType * testPtr_2
    = dynamic_cast< BSplineOrder2TransformType * >( this->m_AdvancedTransform.GetPointer() );
  BSplineOrder3TransformType * testPtr_3
    = dynamic_cast< BSplineOrder3TransformType * >( this->m_AdvancedTransform.GetPointer() );

  bool transformIsBSpline = false;
  if( testPtr_1 || testPtr_2 || testPtr_3 )
  {
    transformIsBSpline = true;
  }
  else if( testPtr_combo )
  {
    /** Check if the current transform is a B-spline transform. */
    BSplineOrder1TransformType * testPtr_1b = dynamic_cast< BSplineOrder1TransformType * >(
      testPtr_combo->GetCurrentTransform() );
    BSplineOrder2TransformType * testPtr_2b = dynamic_cast< BSplineOrder2TransformType * >(
      testPtr_combo->GetCurrentTransform() );
    BSplineOrder3TransformType * testPtr_3b = dynamic_cast< BSplineOrder3TransformType * >(
      testPtr_combo->GetCurrentTransform() );
    if( testPtr_1b || testPtr_2b || testPtr_3b )
    {
      transformIsBSpline = true;
    }
  }

  /** Store the result. */
  this->m_TransformIsBSpline = transformIsBSpline;

} // end CheckForBSplineTransform()


/**
 * ******************* EvaluateMovingImageValueAndDerivative ******************
 */

template< class TFixedImage, class TMovingImage >
bool
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::EvaluateMovingImageValueAndDerivative(
  const MovingImagePointType & mappedPoint,
  RealType & movingImageValue,
  MovingImageDerivativeType * gradient ) const
{
  /** Check if mapped point inside image buffer. */
  MovingImageContinuousIndexType cindex;
  this->m_Interpolator->ConvertPointToContinuousIndex( mappedPoint, cindex );
  bool sampleOk = this->m_Interpolator->IsInsideBuffer( cindex );
  if( sampleOk )
  {
    /** Compute value and possibly derivative. */
    if( gradient )
    {
      if( this->m_InterpolatorIsBSpline && !this->GetComputeGradient() )
      {
        /** Compute moving image value and gradient using the B-spline kernel. */
        this->m_BSplineInterpolator->EvaluateValueAndDerivativeAtContinuousIndex(
          cindex, movingImageValue, *gradient );
      }
      else if( this->m_InterpolatorIsBSplineFloat && !this->GetComputeGradient() )
      {
        /** Compute moving image value and gradient using the B-spline kernel. */
        this->m_BSplineInterpolatorFloat->EvaluateValueAndDerivativeAtContinuousIndex(
          cindex, movingImageValue, *gradient );
      }
      else if( this->m_InterpolatorIsReducedBSpline && !this->GetComputeGradient() )
      {
        /** Compute moving image value and gradient using the B-spline kernel. */
        movingImageValue = this->m_Interpolator->EvaluateAtContinuousIndex( cindex );
        ( *gradient )
          = this->m_ReducedBSplineInterpolator->EvaluateDerivativeAtContinuousIndex( cindex );
        //this->m_ReducedBSplineInterpolator->EvaluateValueAndDerivativeAtContinuousIndex(
        //  cindex, movingImageValue, *gradient );
      }
      else if( this->m_InterpolatorIsLinear && !this->GetComputeGradient() )
      {
        /** Compute moving image value and gradient using the linear interpolator. */
        this->m_LinearInterpolator->EvaluateValueAndDerivativeAtContinuousIndex(
          cindex, movingImageValue, *gradient );
      }
      else
      {
        /** Get the gradient by NearestNeighboorInterpolation of the gradient image.
         * It is assumed that the gradient image is computed.
         */
        movingImageValue = this->m_Interpolator->EvaluateAtContinuousIndex( cindex );
        MovingImageIndexType index;
        for( unsigned int j = 0; j < MovingImageDimension; j++ )
        {
          index[ j ] = static_cast< long >( Math::Round< double >( cindex[ j ] ) );
        }
        ( *gradient ) = this->m_GradientImage->GetPixel( index );
      }
      if( this->m_UseMovingImageDerivativeScales )
      {
        for( unsigned int i = 0; i < MovingImageDimension; ++i )
        {
          ( *gradient )[ i ] *= this->m_MovingImageDerivativeScales[ i ];
        }
      }
    } // end if gradient
    else
    {
      movingImageValue = this->m_Interpolator->EvaluateAtContinuousIndex( cindex );
    }
  } // end if sampleOk

  return sampleOk;

} // end EvaluateMovingImageValueAndDerivative()


/**
 * *************** EvaluateTransformJacobianInnerProduct ****************
 */

template< class TFixedImage, class TMovingImage >
void
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::EvaluateTransformJacobianInnerProduct(
  const TransformJacobianType & jacobian,
  const MovingImageDerivativeType & movingImageDerivative,
  DerivativeType & imageJacobian ) const
{
  typedef typename TransformJacobianType::const_iterator JacobianIteratorType;
  typedef typename DerivativeType::iterator              DerivativeIteratorType;

  /** Multiple the 1-by-dim vector movingImageDerivative with the
   * dim-by-length matrix jacobian, to get a 1-by-length vector imageJacobian.
   * An optimized route can be taken for B-spline transforms.
   */
  if( this->m_TransformIsBSpline )
  {
    // For the B-spline we know that the Jacobian is mostly empty.
    //       [ j ... j 0 ... 0 0 ... 0 ]
    // jac = [ 0 ... 0 j ... j 0 ... 0 ]
    //       [ 0 ... 0 0 ... 0 j ... j ]
    const unsigned int sizeImageJacobian              = imageJacobian.GetSize();
    const unsigned int numberOfParametersPerDimension = sizeImageJacobian / FixedImageDimension;
    unsigned int       counter                        = 0;
    for( unsigned int dim = 0; dim < FixedImageDimension; ++dim )
    {
      const double imDeriv = movingImageDerivative[ dim ];
      for( unsigned int mu = 0; mu < numberOfParametersPerDimension; ++mu )
      {
        imageJacobian( counter )
          = jacobian( dim, counter ) * imDeriv;
        ++counter;
      }
    }
  }
  else
  {
    /** Otherwise perform a full multiplication. */
    JacobianIteratorType jac = jacobian.begin();
    imageJacobian.Fill( 0.0 );
    const unsigned int sizeImageJacobian = imageJacobian.GetSize();

    for( unsigned int dim = 0; dim < FixedImageDimension; ++dim )
    {
      const double           imDeriv = movingImageDerivative[ dim ];
      DerivativeIteratorType imjac   = imageJacobian.begin();

      for( unsigned int mu = 0; mu < sizeImageJacobian; ++mu )
      {
        ( *imjac ) += ( *jac ) * imDeriv;
        ++imjac;
        ++jac;
      }
    }
  }

} // end EvaluateTransformJacobianInnerProduct()


/**
 * ********************** TransformPoint ************************
 */

template< class TFixedImage, class TMovingImage >
bool
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::TransformPoint(
  const FixedImagePointType & fixedImagePoint,
  MovingImagePointType & mappedPoint ) const
{
  mappedPoint = this->m_Transform->TransformPoint( fixedImagePoint );

  /** For future use: return whether the sample is valid */
  const bool valid = true;
  return valid;

} // end TransformPoint()


/**
 * *************** EvaluateTransformJacobian ****************
 */

template< class TFixedImage, class TMovingImage >
bool
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::EvaluateTransformJacobian(
  const FixedImagePointType & fixedImagePoint,
  TransformJacobianType & jacobian,
  NonZeroJacobianIndicesType & nzji ) const
{
  /** Advanced transform: generic sparse Jacobian support */
  this->m_AdvancedTransform->GetJacobian(
    fixedImagePoint, jacobian, nzji );

  /** For future use: return whether the sample is valid */
  const bool valid = true;
  return valid;

} // end EvaluateTransformJacobian()


/**
 * ************************** IsInsideMovingMask *************************
 */

template< class TFixedImage, class TMovingImage >
bool
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::IsInsideMovingMask( const MovingImagePointType & point ) const
{
  /** If a mask has been set: */
  if( this->m_MovingImageMask.IsNotNull() )
  {
    return this->m_MovingImageMask->IsInside( point );
  }

  /** If no mask has been set, just return true. */
  return true;

} // end IsInsideMovingMask()


/**
 * *********************** GetSelfHessian ***********************
 */

template< class TFixedImage, class TMovingImage >
void
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::GetSelfHessian(
  const TransformParametersType & itkNotUsed( parameters ),
  HessianType & H ) const
{
  itkDebugMacro( "GetSelfHessian()" );

  /** Set identity matrix as default implementation. */
  H.set_size( this->GetNumberOfParameters(),
    this->GetNumberOfParameters() );
  //H.Fill(0.0);
  //H.fill_diagonal(1.0);
  for( unsigned int i = 0; i < this->GetNumberOfParameters(); ++i )
  {
    H( i, i ) = 1.0;
  }

} // end GetSelfHessian()


/**
 * *********************** BeforeThreadedGetValueAndDerivative ***********************
 */

template< class TFixedImage, class TMovingImage >
void
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::BeforeThreadedGetValueAndDerivative( const TransformParametersType & parameters ) const
{
  /** In this function do all stuff that cannot be multi-threaded. */
  if( this->m_UseMetricSingleThreaded )
  {
    this->SetTransformParameters( parameters );
    if( this->m_UseImageSampler )
    {
      this->GetImageSampler()->Update();
    }
  }

} // end BeforeThreadedGetValueAndDerivative()


/**
 * **************** GetValueAndDerivativeThreaderCallback *******
 */

template< class TFixedImage, class TMovingImage >
ITK_THREAD_RETURN_TYPE
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::GetValueAndDerivativeThreaderCallback( void * arg )
{
  ThreadInfoType * infoStruct = static_cast< ThreadInfoType * >( arg );
  ThreadIdType     threadID   = infoStruct->ThreadID;

  MultiThreaderParameterType * temp
    = static_cast< MultiThreaderParameterType * >( infoStruct->UserData );

  temp->st_Metric->ThreadedGetValueAndDerivative( threadID );

  return ITK_THREAD_RETURN_VALUE;

} // end GetValueAndDerivativeThreaderCallback()


/**
 * *********************** LaunchGetValueAndDerivativeThreaderCallback***************
 */

template< class TFixedImage, class TMovingImage >
void
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::LaunchGetValueAndDerivativeThreaderCallback( void ) const
{
  /** Setup threader. */
  this->m_Threader->SetSingleMethod( this->GetValueAndDerivativeThreaderCallback,
    const_cast< void * >( static_cast< const void * >( &this->m_ThreaderMetricParameters ) ) );

  /** Launch. */
  this->m_Threader->SingleMethodExecute();

} // end LaunchGetValueAndDerivativeThreaderCallback()


/**
 *********** AccumulateDerivativesThreaderCallback *************
 */

template< class TFixedImage, class TMovingImage >
ITK_THREAD_RETURN_TYPE
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::AccumulateDerivativesThreaderCallback( void * arg )
{
  ThreadInfoType * infoStruct  = static_cast< ThreadInfoType * >( arg );
  ThreadIdType     threadID    = infoStruct->ThreadID;
  ThreadIdType     nrOfThreads = infoStruct->NumberOfThreads;

  MultiThreaderParameterType * temp
    = static_cast< MultiThreaderParameterType * >( infoStruct->UserData );

  const unsigned int numPar  = temp->st_Metric->GetNumberOfParameters();
  const unsigned int subSize = static_cast< unsigned int >(
    vcl_ceil( static_cast< double >( numPar )
    / static_cast< double >( nrOfThreads ) ) );
  const unsigned int jmin = threadID * subSize;
  unsigned int       jmax = ( threadID + 1 ) * subSize;
  jmax = ( jmax > numPar ) ? numPar : jmax;

  /** This thread accumulates all sub-derivatives into a single one, for the
   * range [ jmin, jmax [. Additionally, the sub-derivatives are reset.
   */
  const DerivativeValueType zero = NumericTraits< DerivativeValueType >::Zero;
  const DerivativeValueType normalization = 1.0 / temp->st_NormalizationFactor;
  for( unsigned int j = jmin; j < jmax; ++j )
  {
    DerivativeValueType tmp = zero;
    for( ThreadIdType i = 0; i < nrOfThreads; ++i )
    {
      tmp += temp->st_Metric->m_GetValueAndDerivativePerThreadVariables[ i ].st_Derivative[ j ];

      /** Reset this variable for the next iteration. */
      temp->st_Metric->m_GetValueAndDerivativePerThreadVariables[ i ].st_Derivative[ j ] = zero;
    }
    temp->st_DerivativePointer[ j ] = tmp * normalization;
  }

  return ITK_THREAD_RETURN_VALUE;

} // end AccumulateDerivativesThreaderCallback()


/**
 * *********************** CheckNumberOfSamples ***********************
 */

template< class TFixedImage, class TMovingImage >
void
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::CheckNumberOfSamples(
  unsigned long wanted, unsigned long found ) const
{
  this->m_NumberOfPixelsCounted = found;
  if( found < wanted * this->GetRequiredRatioOfValidSamples() )
  {
    itkExceptionMacro( "Too many samples map outside moving image buffer: "
        << found << " / " << wanted << std::endl );
  }

} // end CheckNumberOfSamples()


/**
 * ********************* PrintSelf ****************************
 */

template< class TFixedImage, class TMovingImage >
void
AdvancedImageToImageMetric< TFixedImage, TMovingImage >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  /** Variables related to the Sampler. */
  os << indent << "Variables related to the Sampler: " << std::endl;
  os << indent.GetNextIndent() << "ImageSampler: "
     << this->m_ImageSampler.GetPointer() << std::endl;
  os << indent.GetNextIndent() << "UseImageSampler: "
     << this->m_UseImageSampler << std::endl;

  /** Variables for the Limiters. */
  os << indent << "Variables related to the Limiters: " << std::endl;
  os << indent.GetNextIndent() << "FixedLimitRangeRatio: "
     << this->m_FixedLimitRangeRatio << std::endl;
  os << indent.GetNextIndent() << "MovingLimitRangeRatio: "
     << this->m_MovingLimitRangeRatio << std::endl;
  os << indent.GetNextIndent() << "UseFixedImageLimiter: "
     << this->m_UseFixedImageLimiter << std::endl;
  os << indent.GetNextIndent() << "UseMovingImageLimiter: "
     << this->m_UseMovingImageLimiter << std::endl;
  os << indent.GetNextIndent() << "FixedImageLimiter: "
     << this->m_FixedImageLimiter.GetPointer() << std::endl;
  os << indent.GetNextIndent() << "MovingImageLimiter: "
     << this->m_MovingImageLimiter.GetPointer() << std::endl;
  os << indent.GetNextIndent() << "FixedImageTrueMin: "
     << this->m_FixedImageTrueMin << std::endl;
  os << indent.GetNextIndent() << "MovingImageTrueMin: "
     << this->m_MovingImageTrueMin << std::endl;
  os << indent.GetNextIndent() << "FixedImageTrueMax: "
     << this->m_FixedImageTrueMax << std::endl;
  os << indent.GetNextIndent() << "MovingImageTrueMax: "
     << this->m_MovingImageTrueMax << std::endl;
  os << indent.GetNextIndent() << "FixedImageMinLimit: "
     << this->m_FixedImageMinLimit << std::endl;
  os << indent.GetNextIndent() << "MovingImageMinLimit: "
     << this->m_MovingImageMinLimit << std::endl;
  os << indent.GetNextIndent() << "FixedImageMaxLimit: "
     << this->m_FixedImageMaxLimit << std::endl;
  os << indent.GetNextIndent() << "MovingImageMaxLimit: "
     << this->m_MovingImageMaxLimit << std::endl;

  /** Variables related to image derivative computation. */
  os << indent << "Variables related to image derivative computation: " << std::endl;
  os << indent.GetNextIndent() << "InterpolatorIsBSpline: "
     << this->m_InterpolatorIsBSpline << std::endl;
  os << indent.GetNextIndent() << "BSplineInterpolator: "
     << this->m_BSplineInterpolator.GetPointer() << std::endl;
  os << indent.GetNextIndent() << "InterpolatorIsBSplineFloat: "
     << this->m_InterpolatorIsBSplineFloat << std::endl;
  os << indent.GetNextIndent() << "BSplineInterpolatorFloat: "
     << this->m_BSplineInterpolatorFloat.GetPointer() << std::endl;
  os << indent.GetNextIndent() << "CentralDifferenceGradientFilter: "
     << this->m_CentralDifferenceGradientFilter.GetPointer() << std::endl;

  /** Variables used when the transform is a B-spline transform. */
  os << indent << "Variables store the transform as an AdvancedTransform: " << std::endl;
  os << indent.GetNextIndent() << "TransformIsAdvanced: "
     << this->m_TransformIsAdvanced << std::endl;
  os << indent.GetNextIndent() << "AdvancedTransform: "
     << this->m_AdvancedTransform.GetPointer() << std::endl;

  /** Other variables. */
  os << indent << "Other variables of the AdvancedImageToImageMetric: " << std::endl;
  os << indent.GetNextIndent() << "RequiredRatioOfValidSamples: "
     << this->m_RequiredRatioOfValidSamples << std::endl;
  os << indent.GetNextIndent() << "UseMovingImageDerivativeScales: "
     << this->m_UseMovingImageDerivativeScales << std::endl;
  os << indent.GetNextIndent() << "MovingImageDerivativeScales: "
     << this->m_MovingImageDerivativeScales << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef _itkAdvancedImageToImageMetric_hxx

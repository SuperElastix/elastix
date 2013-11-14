/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef _itkAdvancedMeanSquaresImageToImageMetric_hxx
#define _itkAdvancedMeanSquaresImageToImageMetric_hxx

#include "itkAdvancedMeanSquaresImageToImageMetric.h"
#include "vnl/algo/vnl_matrix_update.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif


namespace itk
{

/**
 * ******************* Constructor *******************
 */

template <class TFixedImage, class TMovingImage>
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::AdvancedMeanSquaresImageToImageMetric()
{
  this->SetUseImageSampler( true );
  this->SetUseFixedImageLimiter( false );
  this->SetUseMovingImageLimiter( false );

  this->m_UseNormalization = false;
  this->m_NormalizationFactor = 1.0;

  /** SelfHessian related variables, experimental feature. */
  this->m_SelfHessianSmoothingSigma = 1.0;
  this->m_SelfHessianNoiseRange = 1.0;
  this->m_NumberOfSamplesForSelfHessian = 100000;

  this->m_SelfHessianNoiseRange = 1.0;

} // end Constructor


/**
 * ********************* Initialize ****************************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::Initialize( void ) throw ( ExceptionObject )
{
  /** Initialize transform, interpolator, etc. */
  Superclass::Initialize();

  if ( this->GetUseNormalization() )
  {
    /** Try to guess a normalization factor. */
    this->ComputeFixedImageExtrema(
      this->GetFixedImage(),
      this->GetFixedImageRegion() );

    this->ComputeMovingImageExtrema(
      this->GetMovingImage(),
      this->GetMovingImage()->GetBufferedRegion() );

    const double diff1 = this->m_FixedImageTrueMax - this->m_MovingImageTrueMin;
    const double diff2 = this->m_MovingImageTrueMax - this->m_FixedImageTrueMin;
    const double maxdiff = vnl_math_max( diff1, diff2 );

    /** We guess that maxdiff/10 is the maximum average difference that will
     * be observed.
     * \todo We may involve the standard derivation of the image into this estimate.
     */
    this->m_NormalizationFactor = 1.0;
    if ( maxdiff > 1e-10 )
    {
      this->m_NormalizationFactor = 100.0 / maxdiff / maxdiff;
    }
  }
  else
  {
    this->m_NormalizationFactor = 1.0;
  }

} // end Initialize()


/**
 * ******************* PrintSelf *******************
 */

template < class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << "UseNormalization: "
    << this->m_UseNormalization << std::endl;
  os << "SelfHessianSmoothingSigma: "
    << this->m_SelfHessianSmoothingSigma << std::endl;
  os << "NumberOfSamplesForSelfHessian: "
    << this->m_NumberOfSamplesForSelfHessian << std::endl;

} // end PrintSelf()


/**
 * ******************* GetValue *******************
 */

template <class TFixedImage, class TMovingImage>
typename AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::GetValue( const TransformParametersType & parameters ) const
{
  itkDebugMacro( "GetValue( " << parameters << " ) " );

  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  MeasureType measure = NumericTraits< MeasureType >::Zero;

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative( parameters );

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Loop over the fixed image samples to calculate the mean squares. */
  for ( fiter = fbegin; fiter != fend; ++fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;
    RealType movingImageValue;
    MovingImagePointType mappedPoint;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside mask. */
    if ( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
    }

    /** Compute the moving image value and check if the point is
     * inside the moving image buffer.
     */
    if ( sampleOk )
    {
      sampleOk = this->EvaluateMovingImageValueAndDerivative(
        mappedPoint, movingImageValue, 0 );
    }

    if ( sampleOk )
    {
      this->m_NumberOfPixelsCounted++;

      /** Get the fixed image value. */
      const RealType & fixedImageValue = static_cast<double>( (*fiter).Value().m_ImageValue );

      /** The difference squared. */
      const RealType diff = movingImageValue - fixedImageValue;
      measure += diff * diff;

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** Update measure value. */
  double normal_sum = 0.0;
  if ( this->m_NumberOfPixelsCounted > 0 )
  {
    normal_sum = this->m_NormalizationFactor /
      static_cast<double>( this->m_NumberOfPixelsCounted );
  }
  measure *= normal_sum;

  /** Return the mean squares measure value. */
  return measure;

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template < class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::GetDerivative(
  const TransformParametersType & parameters,
  DerivativeType & derivative ) const
{
  /** When the derivative is calculated, all information for calculating
   * the metric value is available. It does not cost anything to calculate
   * the metric value now. Therefore, we have chosen to only implement the
   * GetValueAndDerivative(), supplying it with a dummy value variable.
   */
  MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;
  this->GetValueAndDerivative( parameters, dummyvalue, derivative );

} // end GetDerivative()


/**
 * ******************* GetValueAndDerivativeSingleThreaded *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::GetValueAndDerivativeSingleThreaded(
  const TransformParametersType & parameters,
  MeasureType & value, DerivativeType & derivative ) const
{
  itkDebugMacro( "GetValueAndDerivative( " << parameters << " ) " );

  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  MeasureType measure = NumericTraits< MeasureType >::Zero;
  derivative = DerivativeType( this->GetNumberOfParameters() );
  derivative.Fill( NumericTraits< DerivativeValueType >::Zero );

  /** Array that stores dM(x)/dmu, and the sparse jacobian+indices. */
  NonZeroJacobianIndicesType nzji(
    this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
  DerivativeType imageJacobian( nzji.size() );
  TransformJacobianType jacobian;

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative( parameters );

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Loop over the fixed image to calculate the mean squares. */
  for ( fiter = fbegin; fiter != fend; ++fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;
    RealType movingImageValue;
    MovingImagePointType mappedPoint;
    MovingImageDerivativeType movingImageDerivative;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside mask. */
    if ( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
    }

    /** Compute the moving image value M(T(x)) and derivative dM/dx and check if
     * the point is inside the moving image buffer.
     */
    if ( sampleOk )
    {
      sampleOk = this->EvaluateMovingImageValueAndDerivative(
        mappedPoint, movingImageValue, &movingImageDerivative );
    }

    if ( sampleOk )
    {
      this->m_NumberOfPixelsCounted++;

      /** Get the fixed image value. */
      const RealType & fixedImageValue
        = static_cast<RealType>( (*fiter).Value().m_ImageValue );

      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );

      /** Compute the inner products (dM/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(
        jacobian, movingImageDerivative, imageJacobian );

      /** Compute this pixel's contribution to the measure and derivatives. */
      this->UpdateValueAndDerivativeTerms(
        fixedImageValue, movingImageValue,
        imageJacobian, nzji,
        measure, derivative );

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** Compute the measure value and derivative. */
  double normal_sum = 0.0;
  if ( this->m_NumberOfPixelsCounted > 0 )
  {
    normal_sum = this->m_NormalizationFactor /
      static_cast<double>( this->m_NumberOfPixelsCounted );
  }
  measure *= normal_sum;
  derivative *= normal_sum;

  /** The return value. */
  value = measure;

} // end GetValueAndDerivativeSingleThreaded()


/**
 * ******************* GetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::GetValueAndDerivative(
  const TransformParametersType & parameters,
  MeasureType & value, DerivativeType & derivative ) const
{
  /** Option for now to still use the single threaded code. */
  if( !this->m_UseMultiThread )
  {
    return this->GetValueAndDerivativeSingleThreaded(
      parameters, value, derivative );
  }

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative( parameters );

  /** Initialize some threading related parameters. */
  this->InitializeThreadingParameters();

  /** Launch multi-threading metric */
  this->LaunchGetValueAndDerivativeThreaderCallback();

  /** Gather the metric values and derivatives from all threads. */
  this->AfterThreadedGetValueAndDerivative( value, derivative );

} // end GetValueAndDerivative()


/**
 * ******************* ThreadedGetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::ThreadedGetValueAndDerivative( ThreadIdType threadId )
{
  /** Initialize array that stores dM(x)/dmu, and the sparse Jacobian + indices. */
  const NumberOfParametersType nnzji = this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices();
  NonZeroJacobianIndicesType nzji = NonZeroJacobianIndicesType( nnzji );
  DerivativeType imageJacobian( nzji.size() );

  /** Get a handle to a pre-allocated transform Jacobian. */
  TransformJacobianType & jacobian = this->m_GetValueAndDerivativePerThreadVariables[ threadId ].st_TransformJacobian;

  /** Get a handle to the pre-allocated derivative for the current thread.
   * Also initialize per thread, instead of sequentially in InitializeThreadingParameters().
   */
  DerivativeType & derivative = this->m_GetValueAndDerivativePerThreadVariables[ threadId ].st_Derivative;
  derivative.Fill( NumericTraits<DerivativeValueType>::Zero ); // needed?

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const unsigned long sampleContainerSize = sampleContainer->Size();

  /** Get the samples for this thread. */
  const unsigned long nrOfSamplesPerThreads
    = static_cast<unsigned long>( vcl_ceil( static_cast<double>( sampleContainerSize )
      / static_cast<double>( this->m_NumberOfThreads ) ) );

  unsigned long pos_begin = nrOfSamplesPerThreads * threadId;
  unsigned long pos_end = nrOfSamplesPerThreads * ( threadId + 1 );
  pos_begin = ( pos_begin > sampleContainerSize ) ? sampleContainerSize : pos_begin;
  pos_end = ( pos_end > sampleContainerSize ) ? sampleContainerSize : pos_end;

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator threader_fiter;
  typename ImageSampleContainerType::ConstIterator threader_fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator threader_fend = sampleContainer->Begin();

  threader_fbegin += (int)pos_begin;
  threader_fend += (int)pos_end;

  /** Create variables to store intermediate results. circumvent false sharing */
  unsigned long numberOfPixelsCounted = 0;
  MeasureType measure = NumericTraits<MeasureType>::Zero;

  /** Loop over the fixed image to calculate the mean squares. */
  for( threader_fiter = threader_fbegin; threader_fiter != threader_fend; ++threader_fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = (*threader_fiter).Value().m_ImageCoordinates;
    RealType movingImageValue;
    MovingImagePointType mappedPoint;
    MovingImageDerivativeType movingImageDerivative;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside mask. */
    if( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint ); // thread-safe?
    }

    /** Compute the moving image value M(T(x)) and derivative dM/dx and check if
     * the point is inside the moving image buffer.
     */
    if( sampleOk )
    {
      sampleOk = this->EvaluateMovingImageValueAndDerivative(
        mappedPoint, movingImageValue, &movingImageDerivative );
    }

    if( sampleOk )
    {
      numberOfPixelsCounted++;

      /** Get the fixed image value. */
      const RealType & fixedImageValue
        = static_cast<RealType>( (*threader_fiter).Value().m_ImageValue );

      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );

      /** Compute the inner products (dM/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(
        jacobian, movingImageDerivative, imageJacobian );

      /** Compute this pixel's contribution to the measure and derivatives. */
      this->UpdateValueAndDerivativeTerms(
        fixedImageValue, movingImageValue,
        imageJacobian, nzji,
        measure, derivative );

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Only update these variables at the end to prevent unnecessary "false sharing". */
  this->m_GetValueAndDerivativePerThreadVariables[ threadId ].st_NumberOfPixelsCounted = numberOfPixelsCounted;
  this->m_GetValueAndDerivativePerThreadVariables[ threadId ].st_Value = measure;

} // end ThreadedGetValueAndDerivative()


/**
 * ******************* AfterThreadedGetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::AfterThreadedGetValueAndDerivative(
  MeasureType & value, DerivativeType & derivative ) const
{
  /** Accumulate the number of pixels. */
  this->m_NumberOfPixelsCounted = this->m_GetValueAndDerivativePerThreadVariables[ 0 ].st_NumberOfPixelsCounted;
  for( ThreadIdType i = 1; i < this->m_NumberOfThreads; ++i )
  {
    this->m_NumberOfPixelsCounted += this->m_GetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCounted;
  }

  /** Check if enough samples were valid. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** The normalization factor. */
  DerivativeValueType normal_sum = this->m_NormalizationFactor /
    static_cast<DerivativeValueType>( this->m_NumberOfPixelsCounted );

  /** Accumulate values. */
  value = NumericTraits<MeasureType>::Zero;
  for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
  {
    value += this->m_GetValueAndDerivativePerThreadVariables[ i ].st_Value;
  }
  value *= normal_sum;

  /** Accumulate derivatives. */
  // compute single-threadedly
  if( !this->m_UseMultiThread && false ) // force multi-threaded
  {
    derivative = this->m_GetValueAndDerivativePerThreadVariables[ 0 ].st_Derivative * normal_sum;
    for( ThreadIdType i = 1; i < this->m_NumberOfThreads; i++ )
    {
      derivative += this->m_GetValueAndDerivativePerThreadVariables[ i ].st_Derivative * normal_sum;
    }
  }
  // compute multi-threadedly with itk threads
  else if( true ) // force ITK threads !this->m_UseOpenMP )
  {
    this->m_ThreaderMetricParameters.st_DerivativePointer   = derivative.begin();
    this->m_ThreaderMetricParameters.st_NormalizationFactor = 1.0 / normal_sum;

    typename ThreaderType::Pointer local_threader = ThreaderType::New();
    local_threader->SetNumberOfThreads( this->m_NumberOfThreads );
    local_threader->SetSingleMethod( this->AccumulateDerivativesThreaderCallback,
      const_cast<void *>( static_cast<const void *>( &this->m_ThreaderMetricParameters ) ) );
    local_threader->SingleMethodExecute();
  }
#ifdef ELASTIX_USE_OPENMP
  // compute multi-threadedly with openmp
  else
  {
    const int spaceDimension = static_cast<int>( this->GetNumberOfParameters() );

    #pragma omp parallel for
    for( int j = 0; j < spaceDimension; ++j )
    {
      DerivativeValueType tmp = NumericTraits<DerivativeValueType>::Zero;
      for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
      {
        tmp += this->m_GetValueAndDerivativePerThreadVariables[ i ].st_Derivative[ j ];
      }
      derivative[ j ] = tmp * normal_sum;
    }
  }
#endif

} // end AfterThreadedGetValueAndDerivative()


/**
 * *************** UpdateValueAndDerivativeTerms ***************************
 */

template < class TFixedImage, class TMovingImage >
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::UpdateValueAndDerivativeTerms(
  const RealType fixedImageValue,
  const RealType movingImageValue,
  const DerivativeType & imageJacobian,
  const NonZeroJacobianIndicesType & nzji,
  MeasureType & measure,
  DerivativeType & deriv ) const
{
  /** The difference squared. */
  const RealType diff = movingImageValue - fixedImageValue;
  const RealType diffdiff = diff * diff;
  measure += diffdiff;

  /** Calculate the contributions to the derivatives with respect to each parameter. */
  const RealType diff_2 = diff * 2.0;
  if( nzji.size() == this->GetNumberOfParameters() )
  {
    /** Loop over all Jacobians. */
    typename DerivativeType::const_iterator imjacit = imageJacobian.begin();
    typename DerivativeType::iterator derivit = deriv.begin();
    for( unsigned int mu = 0; mu < this->GetNumberOfParameters(); ++mu )
    {
      (*derivit) += diff_2 * (*imjacit);
      ++imjacit;
      ++derivit;
    }
  }
  else
  {
    /** Only pick the nonzero Jacobians. */
    for( unsigned int i = 0; i < imageJacobian.GetSize(); ++i )
    {
      const unsigned int index = nzji[ i ];
      deriv[ index ] += diff_2 * imageJacobian[ i ];
    }
  }
} // end UpdateValueAndDerivativeTerms()


/**
 * ******************* GetSelfHessian *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::GetSelfHessian( const TransformParametersType & parameters, HessianType & H ) const
{
  itkDebugMacro("GetSelfHessian()");
  typedef Statistics::MersenneTwisterRandomVariateGenerator RandomGeneratorType;

  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  RandomGeneratorType::Pointer randomGenerator = RandomGeneratorType::GetInstance();
  randomGenerator->Initialize();

  /** Array that stores dM(x)/dmu, and the sparse jacobian+indices. */
  NonZeroJacobianIndicesType nzji(
    this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
  DerivativeType imageJacobian( nzji.size() );
  TransformJacobianType jacobian;

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  /** Prepare Hessian */
  H.set_size( this->GetNumberOfParameters(),
    this->GetNumberOfParameters() );
  //H.Fill(0.0); // done by set_size if sparse matrix

  /** Smooth fixed image */
  typename SmootherType::Pointer smoother = SmootherType::New();
  smoother->SetInput( this->GetFixedImage() );
  smoother->SetSigma( this->GetSelfHessianSmoothingSigma() );
  smoother->Update();

  /** Set up interpolator for fixed image */
  typename FixedImageInterpolatorType::Pointer fixedInterpolator = FixedImageInterpolatorType::New();
  if ( this->m_BSplineInterpolator.IsNotNull() )
  {
    fixedInterpolator->SetSplineOrder( this->m_BSplineInterpolator->GetSplineOrder() );
  }
  else
  {
    fixedInterpolator->SetSplineOrder( 1 );
  }
  fixedInterpolator->SetInputImage( smoother->GetOutput() );

  /** Set up random coordinate sampler
   * Actually we could do without a sampler, but it's easy like this.
   */
  typename SelfHessianSamplerType::Pointer sampler = SelfHessianSamplerType::New();
  //typename DummyFixedImageInterpolatorType::Pointer dummyInterpolator =
  //  DummyFixedImageInterpolatorType::New();
  sampler->SetInputImageRegion( this->GetImageSampler()->GetInputImageRegion() );
  sampler->SetMask( this->GetImageSampler()->GetMask() );
  sampler->SetInput( smoother->GetInput() );
  sampler->SetNumberOfSamples( this->m_NumberOfSamplesForSelfHessian );
  //sampler->SetInterpolator( dummyInterpolator );

  /** Update the imageSampler and get a handle to the sample container. */
  sampler->Update();
  ImageSampleContainerPointer sampleContainer = sampler->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Loop over the fixed image to calculate the mean squares. */
  for ( fiter = fbegin; fiter != fend; ++fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;
    MovingImagePointType mappedPoint;
    MovingImageDerivativeType movingImageDerivative;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint);

    /** Check if point is inside mask. NB: we assume here that the
     * initial transformation is approximately ok.
     */
    if ( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
    }

    /** Check if point is inside moving image. NB: we assume here that the
     * initial transformation is approximately ok.
     */
    if ( sampleOk )
    {
      sampleOk = this->m_Interpolator->IsInsideBuffer( mappedPoint );
    }

    if ( sampleOk )
    {
      this->m_NumberOfPixelsCounted++;

      /** Use the derivative of the fixed image for the self Hessian!
       * \todo: we can do this more efficient without the interpolation,
       * without the sampler, and with a precomputed gradient image,
       * but is this the bottleneck?
       */
      movingImageDerivative = fixedInterpolator->EvaluateDerivative( fixedPoint );
      for ( unsigned int d = 0; d < FixedImageDimension; ++d )
      {
        movingImageDerivative[d] += randomGenerator->GetVariateWithClosedRange(
          this->m_SelfHessianNoiseRange ) - this->m_SelfHessianNoiseRange / 2.0;
      }

      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );

      /** Compute the innerproducts (dM/dx)^T (dT/dmu) */
      this->EvaluateTransformJacobianInnerProduct(
        jacobian, movingImageDerivative, imageJacobian );

      /** Compute this pixel's contribution to the SelfHessian. */
      this->UpdateSelfHessianTerms( imageJacobian, nzji, H );

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** Compute the measure value and derivative. */
  if ( this->m_NumberOfPixelsCounted > 0 )
  {
    const double normal_sum = 2.0 * this->m_NormalizationFactor /
      static_cast<double>( this->m_NumberOfPixelsCounted );
    for (unsigned int i = 0; i < this->GetNumberOfParameters(); ++i )
    {
      H.scale_row(i, normal_sum);
    }
  }
  else
  {
    //H.fill_diagonal(1.0);
    for (unsigned int i = 0; i < this->GetNumberOfParameters(); ++i )
    {
      H(i,i) = 1.0;
    }
  }

} // end GetSelfHessian()


/**
 * *************** UpdateSelfHessianTerms ***************************
 */

template < class TFixedImage, class TMovingImage >
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::UpdateSelfHessianTerms(
  const DerivativeType & imageJacobian,
  const NonZeroJacobianIndicesType & nzji,
  HessianType & H ) const
{
  typedef typename HessianType::row RowType;
  typedef typename RowType::iterator RowIteratorType;
  typedef typename HessianType::pair_t ElementType;

  // does not work for sparse matrix. \todo: distinguish between sparse and nonsparse
  ///** Do rank-1 update of H */
  //if ( nzji.size() == this->GetNumberOfParameters() )
  //{
  //  /** Loop over all Jacobians. */
  //  vnl_matrix_update( H, imageJacobian, imageJacobian );
  //}
  //else
  //{
    /** Only pick the nonzero Jacobians.
    * Save only upper triangular part of the matrix */
    const unsigned int imjacsize = imageJacobian.GetSize();
    for ( unsigned int i = 0; i < imjacsize; ++i )
    {
      const unsigned int row = nzji[ i ];
      const double imjacrow = imageJacobian[ i ];

      RowType & rowVector = H.get_row( row );
      RowIteratorType rowIt = rowVector.begin();

      for ( unsigned int j = i; j < imjacsize; ++j )
      {
        const unsigned int col = nzji[ j ];
        const double val = imjacrow * imageJacobian[ j ];
        if ( ( val < 1e-14 ) && ( val > -1e-14 ) )
        {
          continue;
        }

        /** The following implements:
         * H(row,col) += imjacrow * imageJacobian[ j ];
         * But more efficient.
         */

        /** Go to next element */
        for (; (rowIt != rowVector.end()) && ((*rowIt).first < col); ++rowIt );

        if ( (rowIt == rowVector.end()) || ((*rowIt).first != col) )
        {
          /** Add new column to the row and set iterator to that column. */
          rowIt = rowVector.insert( rowIt, ElementType( col, val ) );
        }
        else
        {
          /** Add to existing value */
          (*rowIt).second += val;
        }
      }
    }

  //} // end else

} // end UpdateSelfHessianTerms()


} // end namespace itk


#endif // end #ifndef _itkAdvancedMeanSquaresImageToImageMetric_hxx

/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef _itkAdvancedNormalizedCorrelationImageToImageMetric_txx
#define _itkAdvancedNormalizedCorrelationImageToImageMetric_txx

#include "itkAdvancedNormalizedCorrelationImageToImageMetric.h"


namespace itk
{

/**
 * ******************* Constructor *******************
 */

template <class TFixedImage, class TMovingImage>
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::AdvancedNormalizedCorrelationImageToImageMetric()
{
  this->m_SubtractMean = false;

  this->SetUseImageSampler( true );
  this->SetUseFixedImageLimiter( false );
  this->SetUseMovingImageLimiter( false );

  this->m_ThreaderSff.resize( 0 );
  this->m_ThreaderSmm.resize( 0 );
  this->m_ThreaderSfm.resize( 0 );
  this->m_ThreaderSf.resize( 0 );
  this->m_ThreaderSm.resize( 0 );
  this->m_ThreaderDerivativeF.resize( 0 );
  this->m_ThreaderDerivativeM.resize( 0 );
  this->m_ThreaderDifferential.resize( 0 );

} // end Constructor


/**
 * ******************* Destructor *******************
 */

template <class TFixedImage, class TMovingImage>
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::~AdvancedNormalizedCorrelationImageToImageMetric()
{
} // end Destructor


/**
 * ******************* InitializeThreadingParameters *******************
 */

template < class TFixedImage, class TMovingImage>
void
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::InitializeThreadingParameters( void ) const
{
  // tmp: time this:
  typedef tmr::Timer          TimerType; typedef TimerType::Pointer  TimerPointer;
  TimerPointer timer = TimerType::New();
  timer->StartTimer();

  /** Resize and initialize the threading related parameters. */
  this->m_ThreaderNumberOfPixelsCounted.resize(
    this->m_NumberOfThreads, NumericTraits<SizeValueType>::Zero );
  const AccumulateType zero = NumericTraits<AccumulateType>::Zero;
  this->m_ThreaderSff.resize( this->m_NumberOfThreads, zero );
  this->m_ThreaderSmm.resize( this->m_NumberOfThreads, zero );
  this->m_ThreaderSfm.resize( this->m_NumberOfThreads, zero );
  this->m_ThreaderSf.resize(  this->m_NumberOfThreads, zero );
  this->m_ThreaderSm.resize(  this->m_NumberOfThreads, zero );

  this->m_ThreaderDerivativeF.resize(  this->m_NumberOfThreads );
  this->m_ThreaderDerivativeM.resize(  this->m_NumberOfThreads );
  this->m_ThreaderDifferential.resize( this->m_NumberOfThreads );

  /** Initialize the derivatives. */
  for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
  {
    this->m_ThreaderDerivativeF[ i ].SetSize( this->GetNumberOfParameters() );
    this->m_ThreaderDerivativeF[ i ].Fill( NumericTraits<DerivativeValueType>::Zero );
    this->m_ThreaderDerivativeM[ i ].SetSize( this->GetNumberOfParameters() );
    this->m_ThreaderDerivativeM[ i ].Fill( NumericTraits<DerivativeValueType>::Zero );
    this->m_ThreaderDifferential[ i ].SetSize( this->GetNumberOfParameters() );
    this->m_ThreaderDifferential[ i ].Fill( NumericTraits<DerivativeValueType>::Zero );
  }

  // end timer and store
  timer->StopTimer();
  this->m_FillDerivativesTimings.push_back( timer->GetElapsedClockSec() * 1000.0 );

} // end InitializeThreadingParameters()


/**
 * ******************* PrintSelf *******************
 */

template < class TFixedImage, class TMovingImage>
void
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "SubtractMean: " << this->m_SubtractMean << std::endl;

} // end PrintSelf()


/**
 * *************** UpdateDerivativeTerms ***************************
 */

template < class TFixedImage, class TMovingImage >
void
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::UpdateDerivativeTerms(
  const RealType & fixedImageValue,
  const RealType & movingImageValue,
  const DerivativeType & imageJacobian,
  const NonZeroJacobianIndicesType & nzji,
  DerivativeType & derivativeF,
  DerivativeType & derivativeM,
  DerivativeType & differential ) const
{
  typedef typename DerivativeType::ValueType        DerivativeValueType;

  /** Calculate the contributions to the derivatives with respect to each parameter. */
  if ( nzji.size() == this->GetNumberOfParameters() )
  {
    /** Loop over all Jacobians. */
    typename DerivativeType::const_iterator imjacit = imageJacobian.begin();
    typename DerivativeType::iterator derivativeFit = derivativeF.begin();
    typename DerivativeType::iterator derivativeMit = derivativeM.begin();
    typename DerivativeType::iterator differentialit = differential.begin();

    for ( unsigned int mu = 0; mu < this->GetNumberOfParameters(); ++mu )
    {
      (*derivativeFit) += fixedImageValue * (*imjacit);
      (*derivativeMit) += movingImageValue * (*imjacit);
      (*differentialit) += (*imjacit);
      ++imjacit;
      ++derivativeFit;
      ++derivativeMit;
      ++differentialit;
    }
  }
  else
  {
    /** Only pick the nonzero Jacobians. */
    for ( unsigned int i = 0; i < imageJacobian.GetSize(); ++i )
    {
      const unsigned int index = nzji[ i ];
      const RealType differentialtmp = imageJacobian[ i ];
      derivativeF[ index ] += fixedImageValue  * differentialtmp;
      derivativeM[ index ] += movingImageValue * differentialtmp;
      differential[ index ] += differentialtmp;
    }
  }

} // end UpdateValueAndDerivativeTerms()


/**
 * ******************* GetValue *******************
 */

template <class TFixedImage, class TMovingImage>
typename AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::GetValue( const TransformParametersType & parameters ) const
{
  itkDebugMacro( "GetValue( " << parameters << " ) " );

  /** Initialize some variables */
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

  /** Create variables to store intermediate results. */
  AccumulateType sff = NumericTraits< AccumulateType >::Zero;
  AccumulateType smm = NumericTraits< AccumulateType >::Zero;
  AccumulateType sfm = NumericTraits< AccumulateType >::Zero;
  AccumulateType sf  = NumericTraits< AccumulateType >::Zero;
  AccumulateType sm  = NumericTraits< AccumulateType >::Zero;

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
    * inside the moving image buffer. */
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

      /** Update some sums needed to calculate NC. */
      sff += fixedImageValue  * fixedImageValue;
      smm += movingImageValue * movingImageValue;
      sfm += fixedImageValue  * movingImageValue;
      if ( this->m_SubtractMean )
      {
        sf += fixedImageValue;
        sm += movingImageValue;
      }

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** If SubtractMean, then subtract things from sff, smm and sfm. */
  const RealType N = static_cast<RealType>( this->m_NumberOfPixelsCounted );
  if ( this->m_SubtractMean && this->m_NumberOfPixelsCounted > 0 )
  {
    sff -= ( sf * sf / N );
    smm -= ( sm * sm / N );
    sfm -= ( sf * sm / N );
  }

  /** The denominator of the NC. */
  const RealType denom = -1.0 * vcl_sqrt( sff * smm );

  /** Calculate the measure value. */
  if ( this->m_NumberOfPixelsCounted > 0 && denom < -1e-14 )
  {
    measure = sfm / denom;
  }
  else
  {
    measure = NumericTraits< MeasureType >::Zero;
  }

  /** Return the NC measure value. */
  return measure;

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template < class TFixedImage, class TMovingImage>
void
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::GetDerivative( const TransformParametersType & parameters,
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
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::GetValueAndDerivativeSingleThreaded( const TransformParametersType & parameters,
  MeasureType & value, DerivativeType & derivative ) const
{
  itkDebugMacro( << "GetValueAndDerivative( " << parameters << " ) " );

  typedef typename DerivativeType::ValueType        DerivativeValueType;
  typedef typename TransformJacobianType::ValueType TransformJacobianValueType;

  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  derivative = DerivativeType( this->GetNumberOfParameters() );
  derivative.Fill( NumericTraits< DerivativeValueType >::Zero );
  DerivativeType derivativeF = DerivativeType( this->GetNumberOfParameters() );
  derivativeF.Fill( NumericTraits< DerivativeValueType >::Zero );
  DerivativeType derivativeM = DerivativeType( this->GetNumberOfParameters() );
  derivativeM.Fill( NumericTraits< DerivativeValueType >::Zero );
  DerivativeType differential = DerivativeType( this->GetNumberOfParameters() );
  differential.Fill( NumericTraits< DerivativeValueType >::Zero );

  /** Array that stores dM(x)/dmu, and the sparse Jacobian + indices. */
  NonZeroJacobianIndicesType nzji( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
  DerivativeType imageJacobian( nzji.size() );
  TransformJacobianType jacobian;

  /** Initialize some variables for intermediate results. */
  AccumulateType sff = NumericTraits< AccumulateType >::Zero;
  AccumulateType smm = NumericTraits< AccumulateType >::Zero;
  AccumulateType sfm = NumericTraits< AccumulateType >::Zero;
  AccumulateType sf  = NumericTraits< AccumulateType >::Zero;
  AccumulateType sm  = NumericTraits< AccumulateType >::Zero;

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

  /** Loop over the fixed image to calculate the correlation. */
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
      const RealType & fixedImageValue = static_cast<RealType>( (*fiter).Value().m_ImageValue );

      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );

      /** Compute the innerproducts (dM/dx)^T (dT/dmu) and (dMask/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(
        jacobian, movingImageDerivative, imageJacobian );

      /** Update some sums needed to calculate the value of NC. */
      sff += fixedImageValue  * fixedImageValue;
      smm += movingImageValue * movingImageValue;
      sfm += fixedImageValue  * movingImageValue;
      sf  += fixedImageValue;  // Only needed when m_SubtractMean == true
      sm  += movingImageValue; // Only needed when m_SubtractMean == true

      /** Compute this pixel's contribution to the derivative terms. */
      this->UpdateDerivativeTerms(
        fixedImageValue, movingImageValue, imageJacobian, nzji,
        derivativeF, derivativeM, differential );

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** If SubtractMean, then subtract things from sff, smm, sfm,
   * derivativeF and derivativeM.
   */
  const RealType N = static_cast<RealType>( this->m_NumberOfPixelsCounted );
  if ( this->m_SubtractMean && this->m_NumberOfPixelsCounted > 0 )
  {
    sff -= ( sf * sf / N );
    smm -= ( sm * sm / N );
    sfm -= ( sf * sm / N );

    for( unsigned int i = 0; i < this->GetNumberOfParameters(); i++ )
    {
      derivativeF[ i ] -= sf * differential[ i ] / N;
      derivativeM[ i ] -= sm * differential[ i ] / N;
    }
  }

  /** The denominator of the value and the derivative. */
  const RealType denom = -1.0 * vcl_sqrt( sff * smm );

  /** Calculate the value and the derivative. */
  if ( this->m_NumberOfPixelsCounted > 0 && denom < -1e-14 )
  {
    value = sfm / denom;
    for ( unsigned int i = 0; i < this->GetNumberOfParameters(); i++ )
    {
      derivative[ i ] = ( derivativeF[ i ] - ( sfm / smm ) * derivativeM[ i ] )
        / denom;
    }
  }
  else
  {
    value = NumericTraits< MeasureType >::Zero;
    derivative.Fill( NumericTraits< DerivativeValueType >::Zero );
  }

} // end GetValueAndDerivativeSingleThreaded()


/**
 * ******************* GetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::GetValueAndDerivative(
  const TransformParametersType & parameters,
  MeasureType & value, DerivativeType & derivative ) const
{
  /** Option for now to still use the single threaded code. */
  if ( !this->m_UseMultiThread )
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

  /** launch multithreading metric */
  this->LaunchGetValueAndDerivativeThreaderCallback();

  /** Gather the metric values and derivatives from all threads. */
  this->AfterThreadedGetValueAndDerivative( value, derivative );

} // end GetValueAndDerivative()


/**
 * ******************* ThreadedGetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::ThreadedGetValueAndDerivative( ThreadIdType threadId )
{
  /** Initialize array that stores dM(x)/dmu, and the sparse Jacobian + indices. */
  const unsigned int nnzji = this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices();
  NonZeroJacobianIndicesType nzji = NonZeroJacobianIndicesType( nnzji );
  DerivativeType imageJacobian( nzji.size() );
  TransformJacobianType jacobian( FixedImageDimension, nnzji );
  jacobian.Fill( 0.0 ); // needed?

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const unsigned long sampleContainerSize = sampleContainer->Size();

  /** Get the samples for this thread. */
  const unsigned long nrOfSamplesPerThreads
    = static_cast<unsigned long>( vcl_ceil( static_cast<double>( sampleContainerSize )
      / static_cast<double>( this->m_NumberOfThreads ) ) );

  const unsigned long pos_begin = nrOfSamplesPerThreads * threadId;
  unsigned long pos_end = nrOfSamplesPerThreads * ( threadId + 1 );
  pos_end = ( pos_end > sampleContainerSize ) ? sampleContainerSize : pos_end;

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator threader_fiter;
  typename ImageSampleContainerType::ConstIterator threader_fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator threader_fend = sampleContainer->Begin();

  threader_fbegin += (int)pos_begin;
  threader_fend += (int)pos_end;

  /** Create variables to store intermediate results. */
  AccumulateType sff = NumericTraits< AccumulateType >::Zero;
  AccumulateType smm = NumericTraits< AccumulateType >::Zero;
  AccumulateType sfm = NumericTraits< AccumulateType >::Zero;
  AccumulateType sf  = NumericTraits< AccumulateType >::Zero;
  AccumulateType sm  = NumericTraits< AccumulateType >::Zero;
  unsigned long numberOfPixelsCounted = 0;

  // circumvent false sharing?
  DerivativeType & derivativeF = this->m_ThreaderDerivativeF[ threadId ];
  DerivativeType & derivativeM = this->m_ThreaderDerivativeM[ threadId ];
  DerivativeType & differential = this->m_ThreaderDifferential[ threadId ];

  /** Loop over the fixed image to calculate the mean squares. */
  for ( threader_fiter = threader_fbegin; threader_fiter != threader_fend; ++threader_fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = (*threader_fiter).Value().m_ImageCoordinates;
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
      numberOfPixelsCounted++;

      /** Get the fixed image value. */
      const RealType & fixedImageValue
        = static_cast<RealType>( (*threader_fiter).Value().m_ImageValue );

      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );

      /** Compute the inner products (dM/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(
        jacobian, movingImageDerivative, imageJacobian );

      /** Update some sums needed to calculate the value of NC. */
      sff += fixedImageValue  * fixedImageValue;
      smm += movingImageValue * movingImageValue;
      sfm += fixedImageValue  * movingImageValue;
      sf  += fixedImageValue;  // Only needed when m_SubtractMean == true
      sm  += movingImageValue; // Only needed when m_SubtractMean == true

      /** Compute this pixel's contribution to the derivative terms. */
      this->UpdateDerivativeTerms(
        fixedImageValue, movingImageValue, imageJacobian, nzji,
        derivativeF, derivativeM, differential );
        //this->m_ThreaderDerivativeF[ threadId ],
        //this->m_ThreaderDerivativeM[ threadId ],
        //this->m_ThreaderDifferential[ threadId ] );

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Only update these variables at the end to prevent unnessary "false sharing". */
  this->m_ThreaderSff[ threadId ] = sff;
  this->m_ThreaderSmm[ threadId ] = smm;
  this->m_ThreaderSfm[ threadId ] = sfm;
  this->m_ThreaderSf[  threadId ] = sf;
  this->m_ThreaderSm[  threadId ] = sm;
  this->m_ThreaderNumberOfPixelsCounted[ threadId ] = numberOfPixelsCounted;

} // end ThreadedGetValueAndDerivative()


/**
 * ******************* AfterThreadedGetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::AfterThreadedGetValueAndDerivative(
  MeasureType & value, DerivativeType & derivative ) const
{
  /** Accumulate the number of pixels. */
  this->m_NumberOfPixelsCounted = this->m_ThreaderNumberOfPixelsCounted[ 0 ];
  for( ThreadIdType i = 1; i < this->m_NumberOfThreads; ++i )
  {
    this->m_NumberOfPixelsCounted += this->m_ThreaderNumberOfPixelsCounted[ i ];
  }

  /** Check if enough samples were valid. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** Accumulate values. */
  AccumulateType sff = this->m_ThreaderSff[ 0 ];
  AccumulateType smm = this->m_ThreaderSmm[ 0 ];
  AccumulateType sfm = this->m_ThreaderSfm[ 0 ];
  AccumulateType sf  = this->m_ThreaderSf[ 0 ];
  AccumulateType sm  = this->m_ThreaderSm[ 0 ];
  for( ThreadIdType i = 1; i < this->m_NumberOfThreads; ++i )
  {
    sff += this->m_ThreaderSff[ i ];
    smm += this->m_ThreaderSmm[ i ];
    sfm += this->m_ThreaderSfm[ i ];
    sf  += this->m_ThreaderSf[ i ];
    sm  += this->m_ThreaderSm[ i ];
  }

  /** If SubtractMean, then subtract things from sff, smm and sfm. */
  const RealType N = static_cast<RealType>( this->m_NumberOfPixelsCounted );
  if ( this->m_SubtractMean )
  {
    sff -= ( sf * sf / N );
    smm -= ( sm * sm / N );
    sfm -= ( sf * sm / N );
  }

  /** The denominator of the value and the derivative. */
  const RealType denom = -1.0 * vcl_sqrt( sff * smm );

  /** Check for sufficiently large denominator. */
  if ( denom > -1e-14 )
  {
    value = NumericTraits< MeasureType >::Zero;
    derivative.Fill( NumericTraits< DerivativeValueType >::Zero );
    return;
  }

  /** Calculate the metric value. */
  value = sfm / denom;

  /** Calculate the metric derivative. */
#if 0 // single-threaded
  DerivativeType & derivativeF = this->m_ThreaderDerivativeF[0];
  DerivativeType & derivativeM = this->m_ThreaderDerivativeM[0];
  DerivativeType & differential= this->m_ThreaderDifferential[0];

  for( ThreadIdType i = 1; i < this->m_NumberOfThreads; ++i )
  {
    derivativeF  += this->m_ThreaderDerivativeF[ i ];
    derivativeM  += this->m_ThreaderDerivativeM[ i ];
    differential += this->m_ThreaderDifferential[ i ];
  }

  /** If SubtractMean, then subtract things from  derivativeF and derivativeM. */
  const RealType N = static_cast<RealType>( this->m_NumberOfPixelsCounted );
  if ( this->m_SubtractMean )
  {
    derivativeF -= (sf/N) * differential;
    derivativeM -= (sm/N) * differential;
  }
  derivative = ( derivativeF - ( sfm / smm )*derivativeM ) / denom;
#else // multi-threaded
  MultiThreaderComputeDerivativeType * temp = new MultiThreaderComputeDerivativeType;

  temp->sf_N = sf / N;
  temp->sm_N = sm / N;
  temp->sfm_smm = sfm / smm;
  temp->invDenom = 1.0 / denom;
  temp->subtractMean = this->m_SubtractMean;
  temp->derivativeIterator = derivative.begin();
  temp->m_ThreaderDerivativeFIterator = this->m_ThreaderDerivativeF.begin();
  temp->m_ThreaderDerivativeMIterator = this->m_ThreaderDerivativeM.begin();
  temp->m_ThreaderDifferentialIterator = this->m_ThreaderDifferential.begin();
  temp->numberOfParameters = this->GetNumberOfParameters();

  typename ThreaderType::Pointer local_threader = ThreaderType::New();
  local_threader->SetNumberOfThreads( this->m_NumberOfThreads );
  local_threader->SetSingleMethod( ComputeDerivativesThreaderCallback, temp );
  local_threader->SingleMethodExecute();

  delete temp;

#endif

} // end AfterThreadedGetValueAndDerivative()


/**
 *********** ComputeDerivativesThreaderCallback *************
 */

template <class TFixedImage, class TMovingImage>
ITK_THREAD_RETURN_TYPE
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::ComputeDerivativesThreaderCallback( void * arg )
{
  typedef typename DerivativeType::ValueType        DerivativeValueType;
  ThreadInfoType * infoStruct = static_cast< ThreadInfoType * >( arg );
  ThreadIdType threadId = infoStruct->ThreadID;
  ThreadIdType nrOfThreads = infoStruct->NumberOfThreads;

  MultiThreaderComputeDerivativeType * temp
    = static_cast<MultiThreaderComputeDerivativeType * >( infoStruct->UserData );

  const unsigned int subSize = static_cast<unsigned int>(
    vcl_ceil( static_cast<double>( temp->numberOfParameters ) / static_cast<double>( nrOfThreads ) ) );

  unsigned int jmin = threadId * subSize;
  unsigned int jmax = (threadId+1) * subSize;
  jmax = ( jmax > temp->numberOfParameters ) ? temp->numberOfParameters : jmax;

  DerivativeValueType derivativeF, derivativeM, differential;

  for( unsigned int j = jmin; j < jmax; j++ )
  {
    derivativeF = temp->m_ThreaderDerivativeFIterator [0][j];
    derivativeM = temp->m_ThreaderDerivativeMIterator [0][j];
    differential = temp->m_ThreaderDifferentialIterator[0][j];

    for( ThreadIdType i = 1; i < nrOfThreads; i++ )
    {
      derivativeF += temp->m_ThreaderDerivativeFIterator [i][j];
      derivativeM += temp->m_ThreaderDerivativeMIterator [i][j];
      differential+= temp->m_ThreaderDifferentialIterator[i][j];
    }

    if ( temp->subtractMean )
    {
      derivativeF -= temp->sf_N * differential;
      derivativeM -= temp->sm_N * differential;
    }

    temp->derivativeIterator[j] = ( derivativeF - temp->sfm_smm * derivativeM ) * temp->invDenom;
  }

  return ITK_THREAD_RETURN_VALUE;

} // end ComputeDerivativesThreaderCallback()


} // end namespace itk


#endif // end #ifndef _itkAdvancedNormalizedCorrelationImageToImageMetric_txx

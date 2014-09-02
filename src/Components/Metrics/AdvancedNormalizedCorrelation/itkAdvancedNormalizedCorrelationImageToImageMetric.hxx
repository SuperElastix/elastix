/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef _itkAdvancedNormalizedCorrelationImageToImageMetric_hxx
#define _itkAdvancedNormalizedCorrelationImageToImageMetric_hxx

#include "itkAdvancedNormalizedCorrelationImageToImageMetric.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template< class TFixedImage, class TMovingImage >
AdvancedNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::AdvancedNormalizedCorrelationImageToImageMetric()
{
  this->m_SubtractMean = false;

  this->SetUseImageSampler( true );
  this->SetUseFixedImageLimiter( false );
  this->SetUseMovingImageLimiter( false );

  // Multi-threading structs
  this->m_CorrelationGetValueAndDerivativePerThreadVariables     = NULL;
  this->m_CorrelationGetValueAndDerivativePerThreadVariablesSize = 0;

} // end Constructor


/**
 * ******************* Destructor *******************
 */

template< class TFixedImage, class TMovingImage >
AdvancedNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::~AdvancedNormalizedCorrelationImageToImageMetric()
{
  delete[] this->m_CorrelationGetValueAndDerivativePerThreadVariables;
} // end Destructor


/**
 * ******************* InitializeThreadingParameters *******************
 */

template< class TFixedImage, class TMovingImage >
void
AdvancedNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::InitializeThreadingParameters( void ) const
{
  /** Resize and initialize the threading related parameters.
   * The SetSize() functions do not resize the data when this is not
   * needed, which saves valuable re-allocation time.
   * Filling the potentially large vectors is performed later, in each thread,
   * which has performance benefits for larger vector sizes.
   */

  /** Only resize the array of structs when needed. */
  if( this->m_CorrelationGetValueAndDerivativePerThreadVariablesSize != this->m_NumberOfThreads )
  {
    delete[] this->m_CorrelationGetValueAndDerivativePerThreadVariables;
    this->m_CorrelationGetValueAndDerivativePerThreadVariables
      = new AlignedCorrelationGetValueAndDerivativePerThreadStruct[ this->
      m_NumberOfThreads ];
    this->m_CorrelationGetValueAndDerivativePerThreadVariablesSize = this->m_NumberOfThreads;
  }

  /** Some initialization. */
  const AccumulateType      zero1 = NumericTraits< AccumulateType >::Zero;
  const DerivativeValueType zero2 = NumericTraits< DerivativeValueType >::Zero;
  for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
  {
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCounted = NumericTraits< SizeValueType >::Zero;
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_Sff                   = zero1;
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_Smm                   = zero1;
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_Sfm                   = zero1;
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_Sf                    = zero1;
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_Sm                    = zero1;
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_DerivativeF.SetSize( this->GetNumberOfParameters() );
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_DerivativeM.SetSize( this->GetNumberOfParameters() );
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_Differential.SetSize( this->GetNumberOfParameters() );
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_DerivativeF.Fill( zero2 );
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_DerivativeM.Fill( zero2 );
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_Differential.Fill( zero2 );
  }

} // end InitializeThreadingParameters()


/**
 * ******************* PrintSelf *******************
 */

template< class TFixedImage, class TMovingImage >
void
AdvancedNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "SubtractMean: " << this->m_SubtractMean << std::endl;

} // end PrintSelf()


/**
 * *************** UpdateDerivativeTerms ***************************
 */

template< class TFixedImage, class TMovingImage >
void
AdvancedNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::UpdateDerivativeTerms(
  const RealType & fixedImageValue,
  const RealType & movingImageValue,
  const DerivativeType & imageJacobian,
  const NonZeroJacobianIndicesType & nzji,
  DerivativeType & derivativeF,
  DerivativeType & derivativeM,
  DerivativeType & differential ) const
{
  /** Calculate the contributions to the derivatives with respect to each parameter. */
  if( nzji.size() == this->GetNumberOfParameters() )
  {
    /** Loop over all Jacobians. */
    typename DerivativeType::const_iterator imjacit  = imageJacobian.begin();
    typename DerivativeType::iterator derivativeFit  = derivativeF.begin();
    typename DerivativeType::iterator derivativeMit  = derivativeM.begin();
    typename DerivativeType::iterator differentialit = differential.begin();

    for( unsigned int mu = 0; mu < this->GetNumberOfParameters(); ++mu )
    {
      ( *derivativeFit )  += fixedImageValue * ( *imjacit );
      ( *derivativeMit )  += movingImageValue * ( *imjacit );
      ( *differentialit ) += ( *imjacit );
      ++imjacit;
      ++derivativeFit;
      ++derivativeMit;
      ++differentialit;
    }
  }
  else
  {
    /** Only pick the nonzero Jacobians. */
    for( unsigned int i = 0; i < imageJacobian.GetSize(); ++i )
    {
      const unsigned int index           = nzji[ i ];
      const RealType     differentialtmp = imageJacobian[ i ];
      derivativeF[ index ]  += fixedImageValue  * differentialtmp;
      derivativeM[ index ]  += movingImageValue * differentialtmp;
      differential[ index ] += differentialtmp;
    }
  }

} // end UpdateValueAndDerivativeTerms()


/**
 * ******************* GetValue *******************
 */

template< class TFixedImage, class TMovingImage >
typename AdvancedNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >::MeasureType
AdvancedNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
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
  typename ImageSampleContainerType::ConstIterator fend   = sampleContainer->End();

  /** Create variables to store intermediate results. */
  AccumulateType sff = NumericTraits< AccumulateType >::Zero;
  AccumulateType smm = NumericTraits< AccumulateType >::Zero;
  AccumulateType sfm = NumericTraits< AccumulateType >::Zero;
  AccumulateType sf  = NumericTraits< AccumulateType >::Zero;
  AccumulateType sm  = NumericTraits< AccumulateType >::Zero;

  /** Loop over the fixed image samples to calculate the mean squares. */
  for( fiter = fbegin; fiter != fend; ++fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
    RealType                    movingImageValue;
    MovingImagePointType        mappedPoint;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside mask. */
    if( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
    }

    /** Compute the moving image value and check if the point is
    * inside the moving image buffer. */
    if( sampleOk )
    {
      sampleOk = this->EvaluateMovingImageValueAndDerivative(
        mappedPoint, movingImageValue, 0 );
    }

    if( sampleOk )
    {
      this->m_NumberOfPixelsCounted++;

      /** Get the fixed image value. */
      const RealType & fixedImageValue = static_cast< double >( ( *fiter ).Value().m_ImageValue );

      /** Update some sums needed to calculate NC. */
      sff += fixedImageValue  * fixedImageValue;
      smm += movingImageValue * movingImageValue;
      sfm += fixedImageValue  * movingImageValue;
      if( this->m_SubtractMean )
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
  const RealType N = static_cast< RealType >( this->m_NumberOfPixelsCounted );
  if( this->m_SubtractMean && this->m_NumberOfPixelsCounted > 0 )
  {
    sff -= ( sf * sf / N );
    smm -= ( sm * sm / N );
    sfm -= ( sf * sm / N );
  }

  /** The denominator of the NC. */
  const RealType denom = -1.0 * vcl_sqrt( sff * smm );

  /** Calculate the measure value. */
  if( this->m_NumberOfPixelsCounted > 0 && denom < -1e-14 )
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

template< class TFixedImage, class TMovingImage >
void
AdvancedNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
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

template< class TFixedImage, class TMovingImage >
void
AdvancedNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::GetValueAndDerivativeSingleThreaded( const TransformParametersType & parameters,
  MeasureType & value, DerivativeType & derivative ) const
{
  itkDebugMacro( << "GetValueAndDerivative( " << parameters << " ) " );

  typedef typename DerivativeType::ValueType        DerivativeValueType;

  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  derivative                    = DerivativeType( this->GetNumberOfParameters() );
  derivative.Fill( NumericTraits< DerivativeValueType >::Zero );
  DerivativeType derivativeF = DerivativeType( this->GetNumberOfParameters() );
  derivativeF.Fill( NumericTraits< DerivativeValueType >::Zero );
  DerivativeType derivativeM = DerivativeType( this->GetNumberOfParameters() );
  derivativeM.Fill( NumericTraits< DerivativeValueType >::Zero );
  DerivativeType differential = DerivativeType( this->GetNumberOfParameters() );
  differential.Fill( NumericTraits< DerivativeValueType >::Zero );

  /** Array that stores dM(x)/dmu, and the sparse Jacobian + indices. */
  NonZeroJacobianIndicesType nzji( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
  DerivativeType             imageJacobian( nzji.size() );
  TransformJacobianType      jacobian;

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
  typename ImageSampleContainerType::ConstIterator fend   = sampleContainer->End();

  /** Loop over the fixed image to calculate the correlation. */
  for( fiter = fbegin; fiter != fend; ++fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
    RealType                    movingImageValue;
    MovingImagePointType        mappedPoint;
    MovingImageDerivativeType   movingImageDerivative;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside mask. */
    if( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
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
      this->m_NumberOfPixelsCounted++;

      /** Get the fixed image value. */
      const RealType & fixedImageValue = static_cast< RealType >( ( *fiter ).Value().m_ImageValue );

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
  const RealType N = static_cast< RealType >( this->m_NumberOfPixelsCounted );
  if( this->m_SubtractMean && this->m_NumberOfPixelsCounted > 0 )
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
  if( this->m_NumberOfPixelsCounted > 0 && denom < -1e-14 )
  {
    value = sfm / denom;
    for( unsigned int i = 0; i < this->GetNumberOfParameters(); i++ )
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

template< class TFixedImage, class TMovingImage >
void
AdvancedNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
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

  /** launch multithreading metric */
  this->LaunchGetValueAndDerivativeThreaderCallback();

  /** Gather the metric values and derivatives from all threads. */
  this->AfterThreadedGetValueAndDerivative( value, derivative );

} // end GetValueAndDerivative()


/**
 * ******************* ThreadedGetValueAndDerivative *******************
 */

template< class TFixedImage, class TMovingImage >
void
AdvancedNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::ThreadedGetValueAndDerivative( ThreadIdType threadId )
{
  /** Initialize array that stores dM(x)/dmu, and the sparse Jacobian + indices. */
  const NumberOfParametersType nnzji = this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices();
  NonZeroJacobianIndicesType   nzji  = NonZeroJacobianIndicesType( nnzji );
  DerivativeType               imageJacobian( nzji.size() );

  /** Get handles to the pre-allocated derivatives for the current thread.
   * The initialization is performed at the beginning of each resolution in
   * InitializeThreadingParameters(), and at the end of each iteration in
   * AfterThreadedGetValueAndDerivative() and the accumulate functions.
   */
  DerivativeType & derivativeF  = this->m_CorrelationGetValueAndDerivativePerThreadVariables[ threadId ].st_DerivativeF;
  DerivativeType & derivativeM  = this->m_CorrelationGetValueAndDerivativePerThreadVariables[ threadId ].st_DerivativeM;
  DerivativeType & differential = this->m_CorrelationGetValueAndDerivativePerThreadVariables[ threadId ].st_Differential;

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer     = this->GetImageSampler()->GetOutput();
  const unsigned long         sampleContainerSize = sampleContainer->Size();

  /** Get the samples for this thread. */
  const unsigned long nrOfSamplesPerThreads
    = static_cast< unsigned long >( vcl_ceil( static_cast< double >( sampleContainerSize )
    / static_cast< double >( this->m_NumberOfThreads ) ) );

  unsigned long pos_begin = nrOfSamplesPerThreads * threadId;
  unsigned long pos_end   = nrOfSamplesPerThreads * ( threadId + 1 );
  pos_begin = ( pos_begin > sampleContainerSize ) ? sampleContainerSize : pos_begin;
  pos_end   = ( pos_end > sampleContainerSize ) ? sampleContainerSize : pos_end;

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator threader_fiter;
  typename ImageSampleContainerType::ConstIterator threader_fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator threader_fend   = sampleContainer->Begin();

  threader_fbegin += (int)pos_begin;
  threader_fend   += (int)pos_end;

  /** Create variables to store intermediate results. */
  AccumulateType sff                   = NumericTraits< AccumulateType >::Zero;
  AccumulateType smm                   = NumericTraits< AccumulateType >::Zero;
  AccumulateType sfm                   = NumericTraits< AccumulateType >::Zero;
  AccumulateType sf                    = NumericTraits< AccumulateType >::Zero;
  AccumulateType sm                    = NumericTraits< AccumulateType >::Zero;
  unsigned long  numberOfPixelsCounted = 0;

  /** Loop over the fixed image to calculate the mean squares. */
  for( threader_fiter = threader_fbegin; threader_fiter != threader_fend; ++threader_fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = ( *threader_fiter ).Value().m_ImageCoordinates;
    RealType                    movingImageValue;
    MovingImagePointType        mappedPoint;
    MovingImageDerivativeType   movingImageDerivative;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside mask. */
    if( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
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
        = static_cast< RealType >( ( *threader_fiter ).Value().m_ImageValue );

#if 0
      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );

      /** Compute the inner products (dM/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(
        jacobian, movingImageDerivative, imageJacobian );
#else
      /** Compute the inner product of the transform Jacobian dT/dmu and the moving image gradient dM/dx. */
      this->m_AdvancedTransform->EvaluateJacobianWithImageGradientProduct(
        fixedPoint, movingImageDerivative, imageJacobian, nzji );
#endif

      /** Update some sums needed to calculate the value of NC. */
      sff += fixedImageValue  * fixedImageValue;
      smm += movingImageValue * movingImageValue;
      sfm += fixedImageValue  * movingImageValue;
      sf  += fixedImageValue;  // Only needed when m_SubtractMean == true
      sm  += movingImageValue; // Only needed when m_SubtractMean == true

      /** Compute this voxel's contribution to the derivative terms. */
      this->UpdateDerivativeTerms(
        fixedImageValue, movingImageValue, imageJacobian, nzji,
        derivativeF, derivativeM, differential );

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Only update these variables at the end to prevent unnecessary "false sharing". */
  this->m_CorrelationGetValueAndDerivativePerThreadVariables[ threadId ].st_NumberOfPixelsCounted = numberOfPixelsCounted;
  this->m_CorrelationGetValueAndDerivativePerThreadVariables[ threadId ].st_Sff                   = sff;
  this->m_CorrelationGetValueAndDerivativePerThreadVariables[ threadId ].st_Smm                   = smm;
  this->m_CorrelationGetValueAndDerivativePerThreadVariables[ threadId ].st_Sfm                   = sfm;
  this->m_CorrelationGetValueAndDerivativePerThreadVariables[ threadId ].st_Sf                    = sf;
  this->m_CorrelationGetValueAndDerivativePerThreadVariables[ threadId ].st_Sm                    = sm;

} // end ThreadedGetValueAndDerivative()


/**
 * ******************* AfterThreadedGetValueAndDerivative *******************
 */

template< class TFixedImage, class TMovingImage >
void
AdvancedNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::AfterThreadedGetValueAndDerivative(
  MeasureType & value, DerivativeType & derivative ) const
{
  /** Accumulate the number of pixels. */
  this->m_NumberOfPixelsCounted
    = this->m_CorrelationGetValueAndDerivativePerThreadVariables[ 0 ].st_NumberOfPixelsCounted;
  for( ThreadIdType i = 1; i < this->m_NumberOfThreads; ++i )
  {
    this->m_NumberOfPixelsCounted
      += this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCounted;

    /** Reset this variable for the next iteration. */
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCounted = 0;
  }

  /** Check if enough samples were valid. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** Accumulate values. */
  const AccumulateType zero = NumericTraits< AccumulateType >::Zero;
  AccumulateType sff = this->m_CorrelationGetValueAndDerivativePerThreadVariables[ 0 ].st_Sff;
  AccumulateType smm = this->m_CorrelationGetValueAndDerivativePerThreadVariables[ 0 ].st_Smm;
  AccumulateType sfm = this->m_CorrelationGetValueAndDerivativePerThreadVariables[ 0 ].st_Sfm;
  AccumulateType sf  = this->m_CorrelationGetValueAndDerivativePerThreadVariables[ 0 ].st_Sf;
  AccumulateType sm  = this->m_CorrelationGetValueAndDerivativePerThreadVariables[ 0 ].st_Sm;
  for( ThreadIdType i = 1; i < this->m_NumberOfThreads; ++i )
  {
    sff += this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_Sff;
    smm += this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_Smm;
    sfm += this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_Sfm;
    sf  += this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_Sf;
    sm  += this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_Sm;

    /** Reset these variables for the next iteration. */
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_Sff = zero;
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_Smm = zero;
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_Sfm = zero;
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_Sf  = zero;
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_Sm  = zero;
  }

  /** If SubtractMean, then subtract things from sff, smm and sfm. */
  const RealType N = static_cast< RealType >( this->m_NumberOfPixelsCounted );
  if( this->m_SubtractMean )
  {
    sff -= ( sf * sf / N );
    smm -= ( sm * sm / N );
    sfm -= ( sf * sm / N );
  }

  /** The denominator of the value and the derivative. */
  const RealType denom = -1.0 * vcl_sqrt( sff * smm );

  /** Check for sufficiently large denominator. */
  if( denom > -1e-14 )
  {
    value = NumericTraits< MeasureType >::Zero;
    derivative.Fill( NumericTraits< DerivativeValueType >::Zero );
    return;
  }

  /** Calculate the metric value. */
  value = sfm / denom;

  /** Calculate the metric derivative. */
  // single-threaded
  if( !this->m_UseMultiThread && false ) // force multi-threaded
  {
    DerivativeType & derivativeF  = this->m_CorrelationGetValueAndDerivativePerThreadVariables[ 0 ].st_DerivativeF;
    DerivativeType & derivativeM  = this->m_CorrelationGetValueAndDerivativePerThreadVariables[ 0 ].st_DerivativeM;
    DerivativeType & differential = this->m_CorrelationGetValueAndDerivativePerThreadVariables[ 0 ].st_Differential;

    for( ThreadIdType i = 1; i < this->m_NumberOfThreads; ++i )
    {
      derivativeF  += this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_DerivativeF;
      derivativeM  += this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_DerivativeM;
      differential += this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_Differential;
    }

    /** If SubtractMean, then subtract things from  derivativeF and derivativeM. */
    if( this->m_SubtractMean )
    {
      double diff, derF, derM;
      for( unsigned int i = 0; i < this->GetNumberOfParameters(); ++i )
      {
        diff            = differential[ i ];
        derF            = derivativeF[ i ] - ( sf / N ) * diff;
        derM            = derivativeM[ i ] - ( sm / N ) * diff;
        derivative[ i ] = ( derF - ( sfm / smm ) * derM ) / denom;
      }
    }
    else
    {
      for( unsigned int i = 0; i < this->GetNumberOfParameters(); ++i )
      {
        derivative[ i ] = ( derivativeF[ i ] - ( sfm / smm ) * derivativeM[ i ] ) / denom;
      }
    }
  }
  else if( true ) // force !this->m_UseOpenMP ) // multi-threaded using ITK threads
  {
    MultiThreaderAccumulateDerivativeType * temp = new MultiThreaderAccumulateDerivativeType;

    temp->st_Metric              = const_cast< Self * >( this );
    temp->st_sf_N                = sf / N;
    temp->st_sm_N                = sm / N;
    temp->st_sfm_smm             = sfm / smm;
    temp->st_InvertedDenominator = 1.0 / denom;
    temp->st_DerivativePointer   = derivative.begin();

    this->m_Threader->SetSingleMethod( AccumulateDerivativesThreaderCallback, temp );
    this->m_Threader->SingleMethodExecute();

    delete temp;
  }
#ifdef ELASTIX_USE_OPENMP
  // compute multi-threadedly with openmp
  else
  {
    const int            spaceDimension = static_cast< int >( this->GetNumberOfParameters() );
    const AccumulateType sf_N           = sf / N;
    const AccumulateType sm_N           = sm / N;
    const AccumulateType sfm_smm        = sfm / smm;

    #pragma omp parallel for
    for( int j = 0; j < spaceDimension; ++j )
    {
      DerivativeValueType derivativeF
        = this->m_CorrelationGetValueAndDerivativePerThreadVariables[ 0 ].st_DerivativeF[ j ];
      DerivativeValueType derivativeM
        = this->m_CorrelationGetValueAndDerivativePerThreadVariables[ 0 ].st_DerivativeM[ j ];
      DerivativeValueType differential
        = this->m_CorrelationGetValueAndDerivativePerThreadVariables[ 0 ].st_Differential[ j ];

      for( ThreadIdType i = 1; i < this->m_NumberOfThreads; ++i )
      {
        derivativeF  += this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_DerivativeF[ j ];
        derivativeM  += this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_DerivativeM[ j ];
        differential += this->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_Differential[ j ];
      }

      if( this->m_SubtractMean )
      {
        derivativeF -= sf_N * differential;
        derivativeM -= sm_N * differential;
      }

      derivative[ j ] = ( derivativeF - sfm_smm * derivativeM ) / denom;
    }
  } // end OpenMP
#endif

} // end AfterThreadedGetValueAndDerivative()


/**
 *********** AccumulateDerivativesThreaderCallback *************
 */

template< class TFixedImage, class TMovingImage >
ITK_THREAD_RETURN_TYPE
AdvancedNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::AccumulateDerivativesThreaderCallback( void * arg )
{
  ThreadInfoType * infoStruct  = static_cast< ThreadInfoType * >( arg );
  ThreadIdType     threadId    = infoStruct->ThreadID;
  ThreadIdType     nrOfThreads = infoStruct->NumberOfThreads;

  MultiThreaderAccumulateDerivativeType * temp
    = static_cast< MultiThreaderAccumulateDerivativeType * >( infoStruct->UserData );

  const AccumulateType sf_N                = temp->st_sf_N;
  const AccumulateType sm_N                = temp->st_sm_N;
  const AccumulateType sfm_smm             = temp->st_sfm_smm;
  const RealType       invertedDenominator = temp->st_InvertedDenominator;
  const bool           subtractMean        = temp->st_Metric->m_SubtractMean;

  const unsigned int numPar  = temp->st_Metric->GetNumberOfParameters();
  const unsigned int subSize = static_cast< unsigned int >(
    vcl_ceil( static_cast< double >( numPar ) / static_cast< double >( nrOfThreads ) ) );
  unsigned int jmin = threadId * subSize;
  unsigned int jmax = ( threadId + 1 ) * subSize;
  jmax = ( jmax > numPar ) ? numPar : jmax;

  const DerivativeValueType zero = NumericTraits< DerivativeValueType >::Zero;
  DerivativeValueType derivativeF, derivativeM, differential;
  for( unsigned int j = jmin; j < jmax; ++j )
  {
    derivativeF = derivativeM = differential = zero;
    for( ThreadIdType i = 0; i < nrOfThreads; ++i )
    {
      derivativeF  += temp->st_Metric->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_DerivativeF[ j ];
      derivativeM  += temp->st_Metric->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_DerivativeM[ j ];
      differential += temp->st_Metric->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_Differential[ j ];

      /** Reset these variables for the next iteration. */
      temp->st_Metric->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_DerivativeF[ j ] = zero;
      temp->st_Metric->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_DerivativeM[ j ] = zero;
      temp->st_Metric->m_CorrelationGetValueAndDerivativePerThreadVariables[ i ].st_Differential[ j ] = zero;
    }

    if( subtractMean )
    {
      derivativeF -= sf_N * differential;
      derivativeM -= sm_N * differential;
    }

    temp->st_DerivativePointer[ j ]
      = ( derivativeF - sfm_smm * derivativeM ) * invertedDenominator;
  }

  return ITK_THREAD_RETURN_VALUE;

} // end AccumulateDerivativesThreaderCallback()


} // end namespace itk

#endif // end #ifndef _itkAdvancedNormalizedCorrelationImageToImageMetric_hxx

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

  this->m_ThreaderSff.resize(0);
  this->m_ThreaderSmm.resize(0);
  this->m_ThreaderSfm.resize(0);
  this->m_ThreaderSf.resize(0);
  this->m_ThreaderSm.resize(0);
  this->m_ThreaderDerivativeF.resize(0);
  this->m_ThreaderDerivativeM.resize(0);
  this->m_ThreaderDifferential.resize(0);
  this->m_ThreaderNbOfPixelCounted.resize(0);
  this->m_SampleContainerSize = 0;

} // end constructor

/**
 * ******************* Destructor *******************
 */

template <class TFixedImage, class TMovingImage>
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::~AdvancedNormalizedCorrelationImageToImageMetric()
{
  this->m_ThreaderSff.resize(0);
  this->m_ThreaderSmm.resize(0);
  this->m_ThreaderSfm.resize(0);
  this->m_ThreaderSf.resize(0);
  this->m_ThreaderSm.resize(0);
  this->m_ThreaderDerivativeF.resize(0);
  this->m_ThreaderDerivativeM.resize(0);
  this->m_ThreaderDifferential.resize(0);
  this->m_ThreaderNbOfPixelCounted.resize(0);

} // end destructor

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
 * *************** EvaluateTransformJacobianInnerProduct ****************
 */

template < class TFixedImage, class TMovingImage >
void
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::EvaluateTransformJacobianInnerProduct(
  const TransformJacobianType & jacobian,
  const MovingImageDerivativeType & movingImageDerivative,
  DerivativeType & imageJacobian ) const
{
  typedef typename TransformJacobianType::const_iterator JacobianIteratorType;
  typedef typename DerivativeType::iterator              DerivativeIteratorType;
  JacobianIteratorType jac = jacobian.begin();
  imageJacobian.Fill( 0.0 );

  const unsigned int sizeImageJacobian = imageJacobian.GetSize();
  for ( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
  {
    const double imDeriv = movingImageDerivative[ dim ];
    DerivativeIteratorType imjac = imageJacobian.begin();

    for ( unsigned int mu = 0; mu < sizeImageJacobian; mu++ )
    {
      (*imjac) += (*jac) * imDeriv;
      ++imjac;
      ++jac;
    }
  }
} // end EvaluateTransformJacobianInnerProduct()


/**
 * *************** UpdateDerivativeTerms ***************************
 */

template < class TFixedImage, class TMovingImage >
void
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::UpdateDerivativeTerms(
  const RealType fixedImageValue,
  const RealType movingImageValue,
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

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  /** Update the imageSampler and get a handle to the sample container. */
  this->GetImageSampler()->Update();
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Create variables to store intermediate results. */
  typedef typename NumericTraits< MeasureType >::AccumulateType   AccumulateType;
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
 * ******************* GetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::GetValueAndDerivative(
  const TransformParametersType & parameters,
  MeasureType & value, DerivativeType & derivative ) const
{
  itkDebugMacro( "GetValueAndDerivative( " << parameters << " ) " );

  typedef typename DerivativeType::ValueType        DerivativeValueType;
  typedef typename TransformJacobianType::ValueType TransformJacobianValueType;

  derivative = DerivativeType( this->GetNumberOfParameters() );
  derivative.Fill( NumericTraits< DerivativeValueType >::Zero );
  value = NumericTraits< DerivativeValueType >::Zero;

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

  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  this->m_ThreaderSff.resize(this->m_NumberOfThreadsPerMetric,NumericTraits< AccumulateType >::Zero);
  this->m_ThreaderSmm.resize(this->m_NumberOfThreadsPerMetric,NumericTraits< AccumulateType >::Zero);
  this->m_ThreaderSfm.resize(this->m_NumberOfThreadsPerMetric,NumericTraits< AccumulateType >::Zero);
  this->m_ThreaderSf.resize(this->m_NumberOfThreadsPerMetric,NumericTraits< AccumulateType >::Zero);
  this->m_ThreaderSm.resize(this->m_NumberOfThreadsPerMetric,NumericTraits< AccumulateType >::Zero);

  this->m_ThreaderDerivativeF.resize(this->m_NumberOfThreadsPerMetric);
  this->m_ThreaderDerivativeM.resize(this->m_NumberOfThreadsPerMetric);
  this->m_ThreaderDifferential.resize(this->m_NumberOfThreadsPerMetric);

  this->m_ThreaderNbOfPixelCounted.resize(this->m_NumberOfThreadsPerMetric,0);

  for(unsigned int i = 0; i < this->m_NumberOfThreadsPerMetric;i++)
  {
    this->m_ThreaderSff[i] = NumericTraits< AccumulateType >::Zero;
    this->m_ThreaderSmm[i] = NumericTraits< AccumulateType >::Zero;
    this->m_ThreaderSfm[i] = NumericTraits< AccumulateType >::Zero;
    this->m_ThreaderSf[i]  = NumericTraits< AccumulateType >::Zero;
    this->m_ThreaderSm[i]  = NumericTraits< AccumulateType >::Zero;

    this->m_ThreaderDerivativeF[i].SetSize( this->GetNumberOfParameters() );
    this->m_ThreaderDerivativeF[i].Fill( NumericTraits< DerivativeValueType >::Zero );
    this->m_ThreaderDerivativeM[i].SetSize( this->GetNumberOfParameters() );
    this->m_ThreaderDerivativeM[i].Fill( NumericTraits< DerivativeValueType >::Zero );
    this->m_ThreaderDifferential[i].SetSize( this->GetNumberOfParameters() );
    this->m_ThreaderDifferential[i].Fill( NumericTraits< DerivativeValueType >::Zero );

    this->m_ThreaderNbOfPixelCounted[i] = 0;
  }

  /** Get a handle to the sample container. */
  this->m_SampleContainer = this->GetImageSampler()->GetOutput();
  this->m_SampleContainerSize = this->m_SampleContainer->Size();

  /** launch multithreading metric */
  this->LaunchGetValueAndDerivativeThreaderCallback();

  /** gather the metric value and derivatives from value vectors and derivative vectors */
  this->AfterThreadedGetValueAndDerivative(value,derivative);

} // end GetValueAndDerivative()

#if 0
/**
 * ******************* GetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::GetValueAndDerivative( const TransformParametersType & parameters,
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
  typedef typename NumericTraits< MeasureType >::AccumulateType   AccumulateType;
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

} // end GetValueAndDerivative()

#else

/**
 * ******************* ThreadedGetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::ThreadedGetValueAndDerivative(unsigned int threadID)
{

  /** Array that stores dM(x)/dmu, and the sparse jacobian+indices. */
  NonZeroJacobianIndicesType nzji = NonZeroJacobianIndicesType(this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
  DerivativeType imageJacobian = DerivativeType( nzji.size() );
  TransformJacobianType jacobian;

  unsigned long nrOfSamplerPerThreads = (unsigned long)ceil(double(this->m_SampleContainerSize)
                                                            / double(this->m_NumberOfThreadsPerMetric));
  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator threader_fiter;
  typename ImageSampleContainerType::ConstIterator threader_fbegin = this->m_SampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator threader_fend = this->m_SampleContainer->Begin();

  unsigned long pos_begin = nrOfSamplerPerThreads*threadID;
  unsigned long pos_end = nrOfSamplerPerThreads*(threadID+1);

  pos_end = ( pos_end > this->m_SampleContainerSize ) ? this->m_SampleContainerSize : pos_end;

  threader_fbegin+= (int)pos_begin;
  threader_fend+= (int)pos_end;

  /** Loop over the fixed image to calculate the mean squares. */
  for ( threader_fiter = threader_fbegin; threader_fiter != threader_fend;
        ++threader_fiter)
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
      this->m_ThreaderNbOfPixelCounted[threadID]++;

      /** Get the fixed image value. */
      const RealType & fixedImageValue
        = static_cast<RealType>( (*threader_fiter).Value().m_ImageValue );

      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );

      /** Compute the inner products (dM/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(
        jacobian, movingImageDerivative, imageJacobian );

      /** Update some sums needed to calculate the value of NC. */
      this->m_ThreaderSff[threadID] += fixedImageValue  * fixedImageValue;
      this->m_ThreaderSmm[threadID] += movingImageValue * movingImageValue;
      this->m_ThreaderSfm[threadID] += fixedImageValue  * movingImageValue;
      this->m_ThreaderSf[threadID]  += fixedImageValue;  // Only needed when m_SubtractMean == true
      this->m_ThreaderSm[threadID]  += movingImageValue; // Only needed when m_SubtractMean == true

      /** Compute this pixel's contribution to the derivative terms. */
      this->UpdateDerivativeTerms(
        fixedImageValue, movingImageValue, imageJacobian, nzji,
            this->m_ThreaderDerivativeF[threadID], this->m_ThreaderDerivativeM[threadID],
            this->m_ThreaderDifferential[threadID]);

    } // end if sampleOk

  } // end for loop over the image sample container

}

/**
 * ******************* AfterThreadedGetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::AfterThreadedGetValueAndDerivative(MeasureType & value, DerivativeType & derivative ) const
{
#if 0
  typedef typename DerivativeType::ValueType        DerivativeValueType;
  DerivativeType & derivativeF = this->m_ThreaderDerivativeF[0];
  DerivativeType & derivativeM = this->m_ThreaderDerivativeM[0];
  DerivativeType & differential= this->m_ThreaderDifferential[0];

  AccumulateType & sff = this->m_ThreaderSff[0];
  AccumulateType & smm = this->m_ThreaderSmm[0];
  AccumulateType & sfm = this->m_ThreaderSfm[0];
  AccumulateType & sf  = this->m_ThreaderSf[0];
  AccumulateType & sm  = this->m_ThreaderSm[0];

  this->m_NumberOfPixelsCounted = this->m_ThreaderNbOfPixelCounted[0];
  for(unsigned int i=1; i< this->m_NumberOfThreadsPerMetric;i++)
  {
    this->m_NumberOfPixelsCounted += this->m_ThreaderNbOfPixelCounted[i];
    sff += this->m_ThreaderSff[i];
    smm += this->m_ThreaderSmm[i];
    sfm += this->m_ThreaderSfm[i];
    sf  += this->m_ThreaderSf[ i];
    sm  += this->m_ThreaderSm[ i];

    derivativeF += this->m_ThreaderDerivativeF[i];
    derivativeM += this->m_ThreaderDerivativeM[i];
    differential+= this->m_ThreaderDifferential[i];
  }
  /** If SubtractMean, then subtract things from sff, smm, sfm,
   * derivativeF and derivativeM.
   */
  const RealType N = static_cast<RealType>( this->m_NumberOfPixelsCounted );
  if ( this->m_SubtractMean && this->m_NumberOfPixelsCounted > 0 )
  {
    sff -= ( sf * sf / N );
    smm -= ( sm * sm / N );
    sfm -= ( sf * sm / N );

    derivativeF -= (sf/N) * differential;
    derivativeM -= (sm/N) * differential;

  }

  /** The denominator of the value and the derivative. */
  const RealType denom = -1.0 * vcl_sqrt( sff * smm );

  /** Calculate the value and the derivative. */
  if ( this->m_NumberOfPixelsCounted > 0 && denom < -1e-14 )
  {
    value = sfm / denom;
    derivative = ( derivativeF - ( sfm / smm )*derivativeM ) / denom;
  }
  else
  {
    value = NumericTraits< MeasureType >::Zero;
    derivative.Fill( NumericTraits< DerivativeValueType >::Zero );
  }

#else

  typedef typename DerivativeType::ValueType        DerivativeValueType;

  AccumulateType & sff = this->m_ThreaderSff[0];
  AccumulateType & smm = this->m_ThreaderSmm[0];
  AccumulateType & sfm = this->m_ThreaderSfm[0];
  AccumulateType & sf  = this->m_ThreaderSf[0];
  AccumulateType & sm  = this->m_ThreaderSm[0];

  this->m_NumberOfPixelsCounted = this->m_ThreaderNbOfPixelCounted[0];
  for(unsigned int i=1; i< this->m_NumberOfThreadsPerMetric;i++)
  {
    this->m_NumberOfPixelsCounted += this->m_ThreaderNbOfPixelCounted[i];
    sff += this->m_ThreaderSff[i];
    smm += this->m_ThreaderSmm[i];
    sfm += this->m_ThreaderSfm[i];
    sf  += this->m_ThreaderSf[ i];
    sm  += this->m_ThreaderSm[ i];
  }
  /** If SubtractMean, then subtract things from sff, smm, sfm,
   * derivativeF and derivativeM.
   */
  const RealType N = static_cast<RealType>( this->m_NumberOfPixelsCounted );
  if ( this->m_SubtractMean && this->m_NumberOfPixelsCounted > 0 )
  {
    sff -= ( sf * sf / N );
    smm -= ( sm * sm / N );
    sfm -= ( sf * sm / N );
  }

  /** The denominator of the value and the derivative. */
  const RealType denom = -1.0 * vcl_sqrt( sff * smm );

  /** Calculate the value and the derivative. */
  if ( this->m_NumberOfPixelsCounted > 0 && denom < -1e-14 )
  {
    value = sfm / denom;

    MultiThreaderComputeDerivativeType * temp = new  MultiThreaderComputeDerivativeType;

    temp->sf_N = sf / N ;
    temp->sm_N = sm / N ;
    temp->sfm_smm = sfm / smm ;
    temp->invDenom = 1.0 / denom ;
    temp->derivativeIterator = derivative.begin ();
    temp->m_ThreaderDerivativeFIterator = this->m_ThreaderDerivativeF.begin ();
    temp->m_ThreaderDerivativeMIterator = this->m_ThreaderDerivativeM.begin ();
    temp->m_ThreaderDifferentialIterator = this->m_ThreaderDifferential.begin ();
    temp->numberOfParameters = this->GetNumberOfParameters ();

    typename ThreaderType::Pointer local_threader = ThreaderType::New();
    local_threader->SetNumberOfThreads( this->m_NumberOfThreadsPerMetric );
    local_threader->SetSingleMethod( ComputeDerivativesThreaderCallback, temp );
    local_threader->SingleMethodExecute();

    delete[] temp;
  }
  else
  {
    value = NumericTraits< MeasureType >::Zero;
    derivative.Fill( NumericTraits< DerivativeValueType >::Zero );
  }

#endif

}

/**
 *********** ComputeDerivatives threader callback function *************
 */
template <class TFixedImage, class TMovingImage>
ITK_THREAD_RETURN_TYPE
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,TMovingImage>
::ComputeDerivativesThreaderCallback( void * arg )
{
  typedef typename DerivativeType::ValueType        DerivativeValueType;
  ThreadInfoType * infoStruct = static_cast< ThreadInfoType * >( arg );
  unsigned int threadID = (unsigned int)infoStruct->ThreadID;
  unsigned int nrOfThreads = (unsigned int)infoStruct->NumberOfThreads;

  MultiThreaderComputeDerivativeType * temp
    = static_cast<MultiThreaderComputeDerivativeType * >( infoStruct->UserData );

  unsigned int subSize = (unsigned int)ceil(double(temp->numberOfParameters)/double(nrOfThreads));

  unsigned int jmin = threadID * subSize;
  unsigned int jmax = (threadID+1) * subSize;
  jmax = (jmax > temp->numberOfParameters) ? temp->numberOfParameters :jmax ;

  DerivativeValueType derivativeF, derivativeM, differential;

  for(unsigned int j = jmin; j< jmax;j++)
  {
    derivativeF = NumericTraits<DerivativeValueType>::Zero;
    derivativeM = NumericTraits<DerivativeValueType>::Zero;
    differential = NumericTraits<DerivativeValueType>::Zero;

    for(unsigned int i = 0; i< nrOfThreads;i++)
    {
      derivativeF += temp->m_ThreaderDerivativeFIterator [i][j];
      derivativeM += temp->m_ThreaderDerivativeMIterator [i][j];
      differential+= temp->m_ThreaderDifferentialIterator[i][j];
    }

    derivativeF -= temp->sf_N * differential;
    derivativeM -= temp->sm_N * differential;

    temp->derivativeIterator[j] = ( derivativeF - temp->sfm_smm * derivativeM ) * temp->invDenom;

  }

  return ITK_THREAD_RETURN_VALUE;
}// end ComputeDerivativesThreaderCallback()

#endif

} // end namespace itk


#endif // end #ifndef _itkAdvancedNormalizedCorrelationImageToImageMetric_txx

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

} // end constructor


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

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  /** Update the imageSampler and get a handle to the sample container. */
  this->GetImageSampler()->Update();
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


} // end namespace itk


#endif // end #ifndef _itkAdvancedNormalizedCorrelationImageToImageMetric_txx

/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef _itkAdvancedKappaStatisticImageToImageMetric_txx
#define _itkAdvancedKappaStatisticImageToImageMetric_txx

#include "itkAdvancedKappaStatisticImageToImageMetric.h"

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template <class TFixedImage, class TMovingImage>
AdvancedKappaStatisticImageToImageMetric<TFixedImage,TMovingImage>
::AdvancedKappaStatisticImageToImageMetric()
{
  this->SetComputeGradient( true );
  this->SetUseImageSampler( true );
  this->SetUseFixedImageLimiter( false );
  this->SetUseMovingImageLimiter( false );

  this->m_ForegroundValue = 1.0;
  this->m_Epsilon = 1e-3;
  this->m_Complement = true;

} // end Constructor


/**
 * ******************* PrintSelf *******************
 */

template < class TFixedImage, class TMovingImage>
void
AdvancedKappaStatisticImageToImageMetric<TFixedImage,TMovingImage>
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "Complement: "      << ( this->m_Complement ? "On" : "Off" ) << std::endl;
  os << indent << "ForegroundValue: " << this->m_ForegroundValue << std::endl;
  os << indent << "Epsilon: " << this->m_Epsilon << std::endl;

} // end PrintSelf()


/**
 * *************** EvaluateMovingImageAndTransformJacobianInnerProduct ****************
 */

template < class TFixedImage, class TMovingImage >
void
AdvancedKappaStatisticImageToImageMetric<TFixedImage,TMovingImage>
::EvaluateMovingImageAndTransformJacobianInnerProduct(
  const TransformJacobianType & jacobian,
  const MovingImageDerivativeType & movingImageDerivative,
  DerivativeType & innerProduct ) const
{
  typedef typename TransformJacobianType::const_iterator JacobianIteratorType;
  typedef typename DerivativeType::iterator              DerivativeIteratorType;
  JacobianIteratorType jac = jacobian.begin();
  innerProduct.Fill( 0.0 );
  const unsigned int sizeInnerProduct = innerProduct.GetSize();
  for ( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
  {
    const double imDeriv = movingImageDerivative[ dim ];
    DerivativeIteratorType it = innerProduct.begin();
    for ( unsigned int mu = 0; mu < sizeInnerProduct; mu++ )
    {
      (*it) += (*jac) * imDeriv;
      ++it; ++jac;
    }
  }

} // end EvaluateMovingImageAndTransformJacobianInnerProduct()


/**
 * ******************* GetValue *******************
 */

template <class TFixedImage, class TMovingImage>
typename AdvancedKappaStatisticImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
AdvancedKappaStatisticImageToImageMetric<TFixedImage,TMovingImage>
::GetValue( const TransformParametersType & parameters ) const
{
  itkDebugMacro( "GetValue( " << parameters << " ) " );

  /** Initialize some variables. */
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

  /** Some variables. */
  RealType movingImageValue;
  MovingImagePointType mappedPoint;
  std::size_t fixedForegroundArea  = 0; // or unsigned long
  std::size_t movingForegroundArea = 0;
  std::size_t intersection         = 0;

  /** Loop over the fixed image samples to calculate the kappa statistic. */
  for ( fiter = fbegin; fiter != fend; ++fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside moving mask. */
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

    /** Do the actual calculation of the metric value. */
    if ( sampleOk )
    {
      this->m_NumberOfPixelsCounted++;

      /** Get the fixed image value. */
      const RealType & fixedImageValue
        = static_cast<RealType>( (*fiter).Value().m_ImageValue );

      /** Update the intermediate values. */
      const RealType diffFixed = vnl_math_abs( fixedImageValue - this->m_ForegroundValue );
      const RealType diffMoving = vnl_math_abs( movingImageValue - this->m_ForegroundValue );
      if ( diffFixed < this->m_Epsilon ){ fixedForegroundArea++; }
      if ( diffMoving < this->m_Epsilon ){ movingForegroundArea++; }
      if ( diffFixed < this->m_Epsilon
        && diffMoving < this->m_Epsilon ){ intersection++; }

    } // end if samplOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples( sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** Compute the final metric value. */
  std::size_t areaSum = fixedForegroundArea + movingForegroundArea;
  if ( areaSum == 0 )
  {
    measure = NumericTraits< MeasureType >::Zero;
  }
  else
  {
    measure = 1.0 - 2.0 * static_cast<MeasureType>( intersection )
      / static_cast<MeasureType>( areaSum );
  }
  if ( !this->m_Complement ) measure = 1.0 - measure;

  /** Return the mean squares measure value. */
  return measure;

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template < class TFixedImage, class TMovingImage>
void
AdvancedKappaStatisticImageToImageMetric<TFixedImage,TMovingImage>
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
AdvancedKappaStatisticImageToImageMetric<TFixedImage,TMovingImage>
::GetValueAndDerivative( const TransformParametersType & parameters,
  MeasureType & value, DerivativeType & derivative ) const
{
  itkDebugMacro( "GetValueAndDerivative( " << parameters << " ) " );

  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  MeasureType measure = NumericTraits< MeasureType >::Zero;
  derivative = DerivativeType( this->GetNumberOfParameters() );

  /** Array that stores dM(x)/dmu, and the sparse jacobian+indices. */
  NonZeroJacobianIndicesType nzji( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
  DerivativeType imageJacobian( nzji.size() );
  TransformJacobianType jacobian;

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  /** Update the imageSampler and get a handle to the sample container. */
  this->GetImageSampler()->Update();
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Some variables. */
  RealType movingImageValue;
  MovingImagePointType mappedPoint;
  std::size_t fixedForegroundArea  = 0; // or unsigned long
  std::size_t movingForegroundArea = 0;
  std::size_t intersection         = 0;

  DerivativeType vecSum1( this->GetNumberOfParameters() );
  DerivativeType vecSum2( this->GetNumberOfParameters() );
  vecSum1.Fill( NumericTraits< DerivativeValueType >::Zero );
  vecSum2.Fill( NumericTraits< DerivativeValueType >::Zero );

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Loop over the fixed image to calculate the kappa statistic. */
  for ( fiter = fbegin; fiter != fend; ++fiter )
  {
    /** Read fixed coordinates. */
    const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside moving mask. */
    if ( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
    }

    /** Compute the moving image value M(T(x)) and derivative dM/dx and check if
     * the point is inside the moving image buffer.
     */
    MovingImageDerivativeType movingImageDerivative;
    if ( sampleOk )
    {
      sampleOk = this->EvaluateMovingImageValueAndDerivative(
        mappedPoint, movingImageValue, &movingImageDerivative );
    }

    /** Do the actual calculation of the metric value. */
    if ( sampleOk )
    {
      this->m_NumberOfPixelsCounted++;

      /** Get the fixed image value. */
      const RealType & fixedImageValue
        = static_cast<RealType>( (*fiter).Value().m_ImageValue );

      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );

      /** Compute the inner products (dM/dx)^T (dT/dmu). */
      this->EvaluateMovingImageAndTransformJacobianInnerProduct(
        jacobian, movingImageDerivative, imageJacobian );

      /** Compute this pixel's contribution to the measure and derivatives. */
      this->UpdateValueAndDerivativeTerms(
        fixedImageValue, movingImageValue,
        fixedForegroundArea, movingForegroundArea, intersection,
        imageJacobian, nzji,
        vecSum1, vecSum2 );

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** Compute the final metric value. */
  std::size_t areaSum = fixedForegroundArea + movingForegroundArea;
  const MeasureType intersectionFloat = static_cast<MeasureType>( intersection );
  const MeasureType areaSumFloat = static_cast<MeasureType>( areaSum );
  if ( areaSum > 0 )
  {
    measure = 1.0 - 2.0 * intersectionFloat / areaSumFloat;
  }
  if ( !this->m_Complement ){ measure = 1.0 - measure; }
  value = measure;

  /** Calculate the derivative. */
  MeasureType direction = -1.0;
  if ( !this->m_Complement ) direction = 1.0;
  const MeasureType areaSumFloatSquare = direction * areaSumFloat * areaSumFloat;
  const MeasureType tmp1 = areaSumFloat / areaSumFloatSquare;
  const MeasureType tmp2 = 2.0 * intersectionFloat / areaSumFloatSquare;

  if ( areaSum > 0 )
  {
    derivative = tmp1 * vecSum1 - tmp2 * vecSum2;
  }

} // end GetValueAndDerivative()


/**
 * *************** UpdateValueAndDerivativeTerms ***************************
 */

template < class TFixedImage, class TMovingImage >
void
AdvancedKappaStatisticImageToImageMetric<TFixedImage,TMovingImage>
::UpdateValueAndDerivativeTerms(
  const RealType & fixedImageValue,
  const RealType & movingImageValue,
  std::size_t & fixedForegroundArea,
  std::size_t & movingForegroundArea,
  std::size_t & intersection,
  const DerivativeType & imageJacobian,
  const NonZeroJacobianIndicesType & nzji,
  DerivativeType & sum1,
  DerivativeType & sum2 ) const
{
  const RealType diffFixed = vnl_math_abs( fixedImageValue - this->m_ForegroundValue );
  const RealType diffMoving = vnl_math_abs( movingImageValue - this->m_ForegroundValue );

  /** Update the intermediate values. */
  if ( diffFixed < this->m_Epsilon ) fixedForegroundArea++;
  if ( diffMoving < this->m_Epsilon ) movingForegroundArea++;
  if ( diffFixed < this->m_Epsilon
    && diffMoving < this->m_Epsilon ) intersection++;

  /** Calculate the contributions to the derivatives with respect to each parameter. */
  if ( nzji.size() == this->GetNumberOfParameters() )
  {
    /** Loop over all Jacobians. */
    typename DerivativeType::const_iterator imjacit = imageJacobian.begin();
    typename DerivativeType::iterator sum1it = sum1.begin();
    typename DerivativeType::iterator sum2it = sum2.begin();
    for ( unsigned int mu = 0; mu < this->GetNumberOfParameters(); ++mu )
    {
      if ( diffFixed < this->m_Epsilon )
      {
        (*sum1it) += 2.0 * (*imjacit);
      }
      (*sum2it) += (*imjacit);

      /** Increase iterators. */
      ++imjacit; ++sum1it; ++sum2it;
    }
  }
  else
  {
    /** Only pick the nonzero Jacobians. */
    for ( unsigned int i = 0; i < nzji.size(); ++i )
    {
      const unsigned int index = nzji[ i ];
      const DerivativeValueType imjac = imageJacobian[ i ];
      if ( diffFixed < this->m_Epsilon )
      {
        sum1[ index ] += 2.0 * imjac;
      }
      sum2[ index ] += imjac;
    }
  }

} // end UpdateValueAndDerivativeTerms()


/**
 * *************** ComputeGradient ***************************
 *
 * Compute the moving image gradient (dM/dx) and assigns to m_GradientImage.
 * Overrides superclass implementation.
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedKappaStatisticImageToImageMetric<TFixedImage,TMovingImage>
::ComputeGradient( void )
{
  /** Typedefs. */
  typedef itk::ImageRegionIteratorWithIndex< GradientImageType >    GradientIteratorType;
  typedef itk::ImageRegionConstIteratorWithIndex< MovingImageType > MovingIteratorType;

  /** Create a temporary moving gradient image. */
  typename GradientImageType::Pointer tempGradientImage = GradientImageType::New();
  tempGradientImage->SetRegions( this->m_MovingImage->GetBufferedRegion().GetSize() );
  tempGradientImage->Allocate();

  /** Create and reset iterators. */
  GradientIteratorType git( tempGradientImage, tempGradientImage->GetBufferedRegion() );
  MovingIteratorType mit( this->m_MovingImage, this->m_MovingImage->GetBufferedRegion() );
  git.GoToBegin();
  mit.GoToBegin();

  /** Some temporary variables. */
  typename MovingImageType::IndexType minusIndex, plusIndex, currIndex;
  typename GradientImageType::PixelType tempGradPixel;
  typename MovingImageType::SizeType movingSize
    = this->m_MovingImage->GetBufferedRegion().GetSize();
  typename MovingImageType::IndexType movingIndex
    = this->m_MovingImage->GetBufferedRegion().GetIndex();

  /** Loop over the images. */
  while ( !mit.IsAtEnd() )
  {
    /** Get the current index. */
    currIndex = mit.GetIndex();
    minusIndex = currIndex; plusIndex = currIndex;
    for ( unsigned int i = 0; i < MovingImageDimension; i++ )
    {
      /** Check for being on the edge of the moving image. */
      if ( currIndex[ i ] == movingIndex[ i ]
        || currIndex[ i ] == static_cast<int>( movingIndex[ i ] + movingSize[ i ] - 1 ) )
      {
        tempGradPixel[ i ] = 0.0;
      }
      else
      {
        /** Get the left, center and right values. */
        minusIndex[ i ] = currIndex[ i ] - 1;
        plusIndex[ i ] = currIndex[ i ] + 1;
        const RealType minusVal = static_cast<RealType>( this->m_MovingImage->GetPixel( minusIndex ) );
        const RealType plusVal  = static_cast<RealType>( this->m_MovingImage->GetPixel( plusIndex ) );
        const RealType minusDiff = vnl_math_abs( minusVal - this->m_ForegroundValue );
        const RealType plusDiff  = vnl_math_abs(  plusVal - this->m_ForegroundValue );

        /** Calculate the gradient. */
        if ( minusDiff >= this->m_Epsilon && plusDiff < this->m_Epsilon )
        {
          tempGradPixel[ i ] = 1.0;
        }
        else if ( minusDiff < this->m_Epsilon && plusDiff >= this->m_Epsilon )
        {
          tempGradPixel[ i ] = -1.0;
        }
        else
        {
          tempGradPixel[ i ] = 0.0;
        }
      }

      /** Reset indices. */
      minusIndex = currIndex; plusIndex = currIndex;

    } // end for loop

    /** Set the gradient value and increase iterators. */
    git.Set( tempGradPixel );
    ++git; ++mit;

  } // end while loop

  this->m_GradientImage = tempGradientImage;

} // end ComputeGradient()


} // end namespace itk


#endif // end #ifndef _itkAdvancedKappaStatisticImageToImageMetric_txx


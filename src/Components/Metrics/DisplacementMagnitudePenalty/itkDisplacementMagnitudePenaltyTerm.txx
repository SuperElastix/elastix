/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkDisplacementMagnitudePenaltyTerm_txx
#define __itkDisplacementMagnitudePenaltyTerm_txx

#include "itkDisplacementMagnitudePenaltyTerm.h"
#include "itkVector.h"

namespace itk
{

/**
 * ****************** Constructor *******************************
 */

template< class TFixedImage, class TScalarType >
DisplacementMagnitudePenaltyTerm< TFixedImage, TScalarType >
::DisplacementMagnitudePenaltyTerm()
{
  /** Initialize member variables. */

  /** Turn on the sampler functionality */
  this->SetUseImageSampler( true );

} // end constructor


/**
 * ****************** PrintSelf *******************************
 *

template< class TFixedImage, class TScalarType >
void
DisplacementMagnitudePenaltyTerm< TFixedImage, TScalarType >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  this->Superclass::PrintSelf( os, indent );

  //     os << indent << "Transform: "
  //       << this->m_Transform->GetPointer() << std::endl;

} // end PrintSelf()
*/

/**
 * ****************** GetValue *******************************
 */

template< class TFixedImage, class TScalarType >
typename DisplacementMagnitudePenaltyTerm< TFixedImage, TScalarType >::MeasureType
DisplacementMagnitudePenaltyTerm< TFixedImage, TScalarType >
::GetValue( const ParametersType & parameters ) const
{
  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  RealType measure = NumericTraits<RealType>::Zero;

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  /** Update the imageSampler and get a handle to the sample container. */
  this->GetImageSampler()->Update();
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Loop over the fixed image samples to calculate the penalty term. */
  for ( fiter = fbegin; fiter != fend; ++fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;
    MovingImagePointType mappedPoint;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside mask. */
    if ( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
    }

    if ( sampleOk )
    {
      this->m_NumberOfPixelsCounted++;

      /** Compute the contribution of this point: ||T(x)-x||^2
      * \todo FixedImageDimension should be MovingImageDimension  */
      for ( unsigned int d = 0; d < FixedImageDimension; ++d )
      {
        measure += vnl_math_sqr( mappedPoint[d] - fixedPoint[d] );
      }

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** Update measure value. Avoid division by zero. */
  measure /= vnl_math_max( NumericTraits<RealType>::One,
    static_cast<RealType>( this->m_NumberOfPixelsCounted ) );

  /** Return the value. */
  return static_cast<MeasureType>( measure );

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template< class TFixedImage, class TScalarType >
void
DisplacementMagnitudePenaltyTerm< TFixedImage, TScalarType >
::GetDerivative(
  const ParametersType & parameters,
  DerivativeType & derivative ) const
{
  /** Slower, but works. */
  MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;
  this->GetValueAndDerivative( parameters, dummyvalue, derivative );

} // end GetDerivative()


/**
 * ****************** GetValueAndDerivative *******************************
 */

template< class TFixedImage, class TScalarType >
void
DisplacementMagnitudePenaltyTerm< TFixedImage, TScalarType >
::GetValueAndDerivative(
  const ParametersType & parameters,
  MeasureType & value,
  DerivativeType & derivative ) const
{
  typedef typename MovingImagePointType::VectorType     VectorType;

  /** Create and initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  RealType measure = NumericTraits< RealType >::Zero;
  derivative = DerivativeType( this->GetNumberOfParameters() );
  derivative.Fill( NumericTraits< DerivativeValueType >::Zero );

  /** Array that stores sparse jacobian+indices. */
  NonZeroJacobianIndicesType nzji( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
  const unsigned long nrNonZeroJacobianIndices = nzji.size();
  TransformJacobianType jacobian( FixedImageDimension, nrNonZeroJacobianIndices );
  jacobian.Fill( 0.0 );

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

  /** Loop over the fixed image to calculate the penalty term and its derivative. */
  for ( fiter = fbegin; fiter != fend; ++fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;
    MovingImagePointType mappedPoint;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside mask. */
    if ( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
    }

    if ( sampleOk )
    {
      this->m_NumberOfPixelsCounted++;

      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );

      /** Compute displacement */
      VectorType vec = mappedPoint - fixedPoint;

      /** Compute the contribution to the metric value of this point. */
      measure += vec.GetSquaredNorm();

      /** Compute the contribution to the derivative; (T(x)-x)' dT/dmu
       * \todo FixedImageDimension should be MovingImageDimension  */
      for( unsigned int d = 0; d < FixedImageDimension; ++d )
      {
        const double vecd = vec[d];
        for( unsigned int i = 0; i < nrNonZeroJacobianIndices; ++i )
        {
          const unsigned int mu = nzji[i];
          derivative[mu] += vecd * jacobian( d, i );
        }
      }
    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** Update measure value and derivative. The factor 2 in the derivative
   * originates from the square in ||T(x)-x||^2 */
  const RealType normalizationConstant = vnl_math_max(
    NumericTraits<RealType>::One,
    static_cast<RealType>( this->m_NumberOfPixelsCounted ) );
  measure /= normalizationConstant;
  derivative /= (normalizationConstant/2.0);

  /** The return value. */
  value = static_cast<MeasureType>( measure );

} // end GetValueAndDerivative()


} // end namespace itk

#endif // #ifndef __itkDisplacementMagnitudePenaltyTerm_txx


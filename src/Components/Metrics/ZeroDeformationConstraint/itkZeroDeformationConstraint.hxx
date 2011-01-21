/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkZeroDeformationConstraintMetric_hxx
#define __itkZeroDeformationConstraintMetric_hxx

#include "itkZeroDeformationConstraint.h"

namespace itk
{

  /**
  * ******************* Constructor *******************
  */

  template <class TFixedImage, class TMovingImage>
    ZeroDeformationConstraintMetric<TFixedImage,TMovingImage>
      ::ZeroDeformationConstraintMetric():
        m_CurrentPenaltyTermMultiplier( 1.0 ),
        m_CurrentLagrangeMultipliers( 0 ),
        m_CurrentMaximumAbsoluteDisplacement( 0.0 ),
        m_InitialLangrangeMultiplier( 0.0 )
  {
    this->SetUseImageSampler( true );
    this->SetUseFixedImageLimiter( false );
    this->SetUseMovingImageLimiter( false );

  } // end constructor


  /**
   * ******************* Initialize *******************
   */

  template <class TFixedImage, class TMovingImage>
    void
    ZeroDeformationConstraintMetric<TFixedImage,TMovingImage>
    ::Initialize(void) throw ( ExceptionObject )
  {

    /** Initialize transform, interpolator, etc. */
    Superclass::Initialize();

    /** Set right size of lagrangian multiplier vector and initialize to
     * initial langrange multiplier value as supplied by the user.
     */
    this->GetImageSampler()->Update();
    const unsigned int numConstraints = this->GetImageSampler()->GetOutput()->Size() * MovingImageDimension;
    this->m_CurrentLagrangeMultipliers.clear();
    this->m_CurrentLagrangeMultipliers.resize( numConstraints, this->m_InitialLangrangeMultiplier );

    /** Set right size of current penalty term value vector. These values are
     * used to determine new langrange multipliers.
     */
    this->m_CurrentPenaltyTermValues.clear();
    this->m_CurrentPenaltyTermValues.resize( numConstraints, 0.0f );

  } // end Initialize


  /**
   * ******************* PrintSelf *******************
   */

  template < class TFixedImage, class TMovingImage>
    void
    ZeroDeformationConstraintMetric<TFixedImage,TMovingImage>
    ::PrintSelf(std::ostream& os, Indent indent) const
  {
    Superclass::PrintSelf( os, indent );

  } // end PrintSelf


  /**
   * ******************* GetValue *******************
   */

  template <class TFixedImage, class TMovingImage>
    typename ZeroDeformationConstraintMetric<TFixedImage,TMovingImage>::MeasureType
    ZeroDeformationConstraintMetric<TFixedImage,TMovingImage>
    ::GetValue( const TransformParametersType & parameters ) const
  {
    itkDebugMacro( "GetValue( " << parameters << " ) " );

    /** Make sure the transform parameters are up to date. */
    this->SetTransformParameters( parameters );

    /** Update the imageSampler and get a handle to the sample container. */
    this->GetImageSampler()->Update();
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

    /** Create iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator fiter;
    typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

    /** Initialize some variables. */
    float sumDisplacement = 0.0f;
    float sumDisplacementSquared = 0.0f;
    this->m_CurrentInfeasibility = 0.0f;
    this->m_NumberOfPixelsCounted = 0;
    this->m_CurrentMaximumAbsoluteDisplacement = 0.0f;

    /** Loop over the samples to compute sums for computation of the metric. */
    int i = 0;
    for ( fiter = fbegin; fiter != fend; ++fiter )
    {
      /** Read fixed coordinates. */
      FixedImagePointType fixedPoint = (*fiter).Value().m_ImageCoordinates;

      /** Transform point and check if it is inside the b-spline support region. */
      MovingImagePointType mappedPoint;
      bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

      /** Check if point is inside mask. */
      if ( sampleOk )
      {
        sampleOk = this->IsInsideMovingMask( mappedPoint );
      }

      /** Compute the displacement. */
      if ( sampleOk )
      {
        this->m_NumberOfPixelsCounted++;

        /** Process all coordinate axes. */
        for ( unsigned int d = 0; d < MovingImageDimension; ++d )
        {
          /** Update displacement sums. */
          const float displacement =  mappedPoint[ d ] - fixedPoint[ d ];
          sumDisplacement += this->m_CurrentLagrangeMultipliers[ i ] * displacement;
          sumDisplacementSquared += displacement * displacement;

          /** Remember current penalty term value. */
          this->m_CurrentPenaltyTermValues[ i ] = displacement;
          this->m_CurrentMaximumAbsoluteDisplacement = std::max( this->m_CurrentMaximumAbsoluteDisplacement, static_cast<float> ( fabs( displacement ) ) );

          /** Update infeasibility value. */
          this->m_CurrentInfeasibility += fabs( displacement );

          ++i;
        }
      }
      else
      {
        /** Skip point. */
        i += 3;
      }

    } // end for loop over the image sample container

    /** Average infeasibility measure. */
    this->m_CurrentInfeasibility /= ( this->m_NumberOfPixelsCounted * MovingImageDimension );

    /** Check if enough samples were valid. */
    this->CheckNumberOfSamples(
      sampleContainer->Size(), this->m_NumberOfPixelsCounted );

    /** Return the mean squares measure value. */
    return ( -sumDisplacement + this->m_CurrentPenaltyTermMultiplier / 2.0 * sumDisplacementSquared ) / static_cast< float > ( this->m_NumberOfPixelsCounted * MovingImageDimension );

  } // end GetValue


  /**
   * ******************* GetDerivative *******************
   */

  template < class TFixedImage, class TMovingImage>
    void
    ZeroDeformationConstraintMetric<TFixedImage,TMovingImage>
    ::GetDerivative( const TransformParametersType & parameters,
    DerivativeType & derivative ) const
  {
    /** When the derivative is calculated, all information for calculating
     * the metric value is available. It does not cost anything to calculate
     * the metric value now. Therefore, we have chosen to only implement the
     * GetValueAndDerivative(), supplying it with a dummy value variable. */
    MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;
    this->GetValueAndDerivative( parameters, dummyvalue, derivative );

  } // end GetDerivative


  /**
   * ******************* GetValueAndDerivative *******************
   */

  template <class TFixedImage, class TMovingImage>
    void
    ZeroDeformationConstraintMetric<TFixedImage,TMovingImage>
    ::GetValueAndDerivative( const TransformParametersType & parameters,
    MeasureType & value, DerivativeType & derivative ) const
  {
    itkDebugMacro( "GetValueAndDerivative( " << parameters << " ) ");

    /** Define derivative and Jacobian types. */
    typedef typename DerivativeType::ValueType        DerivativeValueType;
    typedef typename TransformJacobianType::ValueType TransformJacobianValueType;

    /** Initialize measure and derivative to zero. */
    derivative = DerivativeType( this->GetNumberOfParameters() );
    derivative.Fill( NumericTraits< DerivativeValueType >::Zero );

    /** Arrays that store dM(x)/dmu, and the sparse jacobian+indices. */
    NonZeroJacobianIndicesType nzji( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
    DerivativeType imageJacobian( nzji.size() );
    TransformJacobianType jacobian;

    /** Make sure the transform parameters are up to date. */
    this->SetTransformParameters( parameters );

    /** Update the imageSampler and get a handle to the sample container. */
    this->GetImageSampler()->Update();
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

    /** Create iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator fiter;
    typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

    /** Initialize some variables. */
    float sumDisplacement           = 0.0;
    float sumDisplacementSquared    = 0.0;
    this->m_CurrentInfeasibility     = 0.0;
    this->m_NumberOfPixelsCounted    = 0;
    this->m_CurrentMaximumAbsoluteDisplacement = 0.0;

    /** Loop over the samples to compute sums for the metric value and derivative. */
    int i = 0;
    for ( fiter = fbegin; fiter != fend; ++fiter )
    {

      /** Read fixed coordinates. */
      FixedImagePointType fixedPoint = (*fiter).Value().m_ImageCoordinates;

      /** Transform point and check if it is inside the bspline support region. */
      MovingImagePointType mappedPoint;
      bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

      /** Check if point is inside mask. */
      if ( sampleOk )
      {
        sampleOk = this->IsInsideMovingMask( mappedPoint );
      }

      /** Compute the displacement. */
      if ( sampleOk )
      {

        this->m_NumberOfPixelsCounted++;

        /** Get the TransformJacobian dT/dMu (Jacobian). */
        this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );
        const typename NonZeroJacobianIndicesType::size_type numNonZeroJacobianIndices = nzji.size();

        /** Process all coordinate axes. */
        for ( unsigned int d = 0; d < MovingImageDimension; ++d )
        {

          /** Update displacement sums. */
          const float displacement =  mappedPoint[ d ] - fixedPoint[ d ];
          sumDisplacement += this->m_CurrentLagrangeMultipliers[ i ] * displacement;
          sumDisplacementSquared += displacement * displacement;

          /** Remember current penalty term value. */
          this->m_CurrentPenaltyTermValues[ i ] = displacement;
          this->m_CurrentMaximumAbsoluteDisplacement = std::max( this->m_CurrentMaximumAbsoluteDisplacement, static_cast<float>( fabs( displacement ) ) );

          /** Update infeasibility value. */
          this->m_CurrentInfeasibility += fabs( displacement );

          /** Part of the computation, which still needs to be multiplied with the Jacobian. */
          const float mult = - this->m_CurrentLagrangeMultipliers[ i ]
                             + this->m_CurrentPenaltyTermMultiplier * displacement;

          /** Compute (dT/dMu)T * mult. */
          for ( typename NonZeroJacobianIndicesType::size_type n = 0; n < numNonZeroJacobianIndices; ++n )
          {
            float jacScaledTransformation = jacobian( d, n ) * mult;

            /** Update derivative sum. */
            derivative[ nzji[ n ] ] += jacScaledTransformation;
          }

          ++i;

        }
      }
      else
      {

        /** Skip point. */
        i += 3;

      }

    } // end for loop over the image sample container

    /** Average infeasibility measure. */
    this->m_CurrentInfeasibility /= ( this->m_NumberOfPixelsCounted * MovingImageDimension );

    /** Check if enough samples were valid. */
    this->CheckNumberOfSamples( sampleContainer->Size(), this->m_NumberOfPixelsCounted );

    /** Return the measure value. */
    value = ( -sumDisplacement + this->m_CurrentPenaltyTermMultiplier / 2.0 * sumDisplacementSquared ) / static_cast< float > ( this->m_NumberOfPixelsCounted * MovingImageDimension );

    /** Determine final derivative. */
    for ( unsigned int p = 0; p < this->GetNumberOfParameters(); ++p)
    {
      derivative[ p ] /= static_cast< float > ( this->m_NumberOfPixelsCounted * MovingImageDimension );
    }

  } // end GetValueAndDerivative()

} // end namespace itk

#endif // end #ifndef _itkZeroDeformationConstraintMetric_hxx


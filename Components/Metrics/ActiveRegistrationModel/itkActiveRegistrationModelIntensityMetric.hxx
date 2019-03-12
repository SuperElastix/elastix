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
#ifndef _itkActiveRegistrationModelImageIntensityMetric_hxx
#define _itkActiveRegistrationModelImageIntensityMetric_hxx

#include "itkActiveRegistrationModelIntensityMetric.h"
#include "vnl/algo/vnl_matrix_update.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk {

/**
 * ******************* Constructor *******************
 */

template<class TFixedImage, class TMovingImage>
ActiveRegistrationModelIntensityMetric<TFixedImage, TMovingImage>
::ActiveRegistrationModelIntensityMetric() {
  this->SetUseImageSampler(true);
  this->SetUseFixedImageLimiter(false);
  this->SetUseMovingImageLimiter(false);

} // end Constructor

/**
 * ********************* Initialize ****************************
 */

template<class TFixedImage, class TMovingImage>
void
ActiveRegistrationModelIntensityMetric<TFixedImage, TMovingImage>
::Initialize(void) {
  /** Initialize transform, interpolator, etc. */
  Superclass::Initialize();
} // end Initialize()


/**
 * ******************* PrintSelf *******************
 */

template<class TFixedImage, class TMovingImage>
void
ActiveRegistrationModelIntensityMetric<TFixedImage, TMovingImage>
::PrintSelf(std::ostream &os, Indent indent) const {
  Superclass::PrintSelf(os, indent);
} // end PrintSelf()


/**
 * ******************* GetValue *******************
 */

template<class TFixedImage, class TMovingImage>
typename ActiveRegistrationModelIntensityMetric<TFixedImage, TMovingImage>::MeasureType
ActiveRegistrationModelIntensityMetric<TFixedImage, TMovingImage>
::GetValue( const TransformParametersType &parameters ) const {
  MeasureType value = NumericTraits< MeasureType >::ZeroValue();
  GetValue( value, parameters );
  return value;
}

template<class TFixedImage, class TMovingImage>
void
ActiveRegistrationModelIntensityMetric<TFixedImage, TMovingImage>
::GetValue( MeasureType &value, const TransformParametersType &parameters ) const {
  itkDebugMacro("GetValue( " << parameters << " ) ");
  value = 0.0;

  // Make sure transform parameters are up-to-date
  this->SetTransformParameters(parameters);

  /** Create iterator over the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  typename ImageSampleContainerType::ConstIterator fixedSampleContainerIterator = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fixedSampleContainerIteratorEnd = sampleContainer->End();

  unsigned int numberOfSamples = 0;
  while( fixedSampleContainerIterator != fixedSampleContainerIteratorEnd )
  {
    const FixedImagePointType& fixedPoint = fixedSampleContainerIterator->Value().m_ImageCoordinates;
    MovingImagePointType       movingPoint;
    RealType                   movingImageValue;

    bool sampleOk = this->TransformPoint( fixedPoint, movingPoint );

    if( sampleOk )
    {
        sampleOk = this->EvaluateMovingImageValueAndDerivative( movingPoint, movingImageValue, 0 );
    }

    if( sampleOk )
    {
      // TODO: GetLevel??????? Loop over models right?
      const unsigned int pointId = this->GetStatisticalModelContainer()->ElementAt( this->GetLevel() )->GetRepresenter()->GetPointIdForPoint( fixedSampleContainerIterator->Value().m_ImageCoordinates );
      movingImageValue -= this->GetStatisticalModelContainer()->ElementAt( this->GetLevel() )->DrawMeanAtPoint( pointId );
      const StatisticalModelVectorType pcaBasis = this->GetStatisticalModelOrthonormalPCABasisMatrixContainer()->ElementAt( this->GetLevel() ).get_row( pointId );
      value += movingImageValue * ( 1.0 - dot_product( pcaBasis, pcaBasis) ) * movingImageValue;

      ++numberOfSamples;
    }

    ++fixedSampleContainerIterator;
  }

  if( numberOfSamples > 0 )
  {
    value /= numberOfSamples;
  }

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template< class TFixedImage, class TMovingImage >
void
ActiveRegistrationModelIntensityMetric< TFixedImage, TMovingImage >
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
 * ******************* GetValueAndDerivative *******************
 */

template< class TFixedImage, class TMovingImage >
void
ActiveRegistrationModelIntensityMetric< TFixedImage, TMovingImage >
::GetValueAndDerivative(
  const TransformParametersType & parameters,
  MeasureType & value, DerivativeType & derivative ) const
{
  value = NumericTraits< MeasureType >::ZeroValue();
  derivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

  StatisticalModelMatrixType I(FixedImageDimension, FixedImageDimension);
  I.set_identity();

  // Make sure transform parameters are up to date
  this->SetTransformParameters( parameters );

  TransformJacobianType Jacobian;
  NonZeroJacobianIndicesType nzji( this->GetTransform()->GetNumberOfNonZeroJacobianIndices() );
  DerivativeType imageJacobian( nzji.size() );

  unsigned int numberOfSamples = 0u;
  unsigned int numberOfPrincipalComponents;

  double movingImageValueInnerProduct = 0.0;

  ImageSampleContainerPointer fixedSampleContainer = this->GetImageSampler()->GetOutput();
  this->GetImageSampler()->Update();
  typename ImageSampleContainerType::ConstIterator fixedSampleContainerIterator = fixedSampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fixedSampleContainerIteratorEnd = fixedSampleContainer->End();
  while( fixedSampleContainerIterator != fixedSampleContainerIteratorEnd )
  {
    const FixedImagePointType& fixedPoint = fixedSampleContainerIterator->Value().m_ImageCoordinates;
    MovingImagePointType       movingPoint;
    RealType                   movingImageValue;
    MovingImageDerivativeType  movingImageDerivative;

    bool sampleOk = this->TransformPoint( fixedPoint, movingPoint );
    if( sampleOk )
    {
      sampleOk = this->EvaluateMovingImageValueAndDerivative( movingPoint, movingImageValue, &movingImageDerivative );
    }

    if( sampleOk )
    {
      const unsigned int pointId = this->GetStatisticalModelContainer()->ElementAt( this->GetLevel() )->GetRepresenter()->GetPointIdForPoint( fixedSampleContainerIterator->Value().m_ImageCoordinates );

      // M'M
      movingImageValue -= this->GetStatisticalModelContainer()->ElementAt( this->GetLevel() )->DrawMeanAtPoint( pointId );
      movingImageValueInnerProduct += movingImageValue * movingImageValue;

      // I-VV^T
      const StatisticalModelVectorType PCABasis = this->GetStatisticalModelOrthonormalPCABasisMatrixContainer()->ElementAt( this->GetLevel() ).get_row( pointId );
      // const double intensityModelReconstructionFactor = movingImageValue * ( 1.0 - dot_product( PCABasis, PCABasis ) );
      const double intensityModelReconstructionFactor = (movingImageValue - meanImageValue) * ( I - dot_product( PCABasis, PCABasis ) );

      // (dM/dx)(dT/du)
      this->m_AdvancedTransform->EvaluateJacobianWithImageGradientProduct( fixedPoint, movingImageDerivative, imageJacobian, nzji );

      // Loop over Jacobian
      for( unsigned int i = 0; i < imageJacobian.GetSize(); ++i )
      {
        const unsigned int mu = nzji[ i ];
        derivative[ mu ] += intensityModelReconstructionFactor * imageJacobian[ i ];
      }

      ++numberOfSamples;
    }

    ++fixedSampleContainerIterator;
  }

  if( std::isnan( value ) )
  {
    itkExceptionMacro( "Model value is NaN.");
  }

  if( numberOfSamples > 0 )
  {
    value = movingImageValueInnerProduct / numberOfSamples;
    derivative *= 2.0 / numberOfSamples;
  }

  const bool useFiniteDifferenceDerivative = true;
  if( useFiniteDifferenceDerivative )
  {
    elxout << "Analytical: " << value << ", " << derivative << std::endl;
    value = NumericTraits< MeasureType >::ZeroValue();
    derivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );
    this->GetValueAndFiniteDifferenceDerivative( parameters, value, derivative );
  }

} // end GetValueAndDerivative()


/**
 * ******************* GetValueAndFiniteDifferenceDerivative *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
void
ActiveRegistrationModelIntensityMetric< TFixedPointSet, TMovingPointSet >
::GetValueAndFiniteDifferenceDerivative( const TransformParametersType & parameters,
                                         MeasureType & value,
                                         DerivativeType & derivative ) const
{
  // Initialize value container
  value = NumericTraits< MeasureType >::ZeroValue();
  derivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

  // Get value
  this->GetValue( value, parameters );

  // Get derivative
  this->GetFiniteDifferenceDerivative( derivative, parameters );
  elxout << "FiniteDifference   : " << value << ", " << derivative << std::endl;
}

/**
 * ******************* GetModelFiniteDifferenceDerivative *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
void
ActiveRegistrationModelIntensityMetric< TFixedPointSet, TMovingPointSet >
::GetFiniteDifferenceDerivative( DerivativeType & modelDerivative,
                                 const TransformParametersType & parameters ) const
{
  const double h = 0.01;

  // Get derivative (J(X)-W*(inv(C)*(W^T*J(X))))^T*f(X)
  unsigned int siz = parameters.size();
  for( unsigned int i = 0; i < parameters.size(); ++i )
{
    MeasureType plusModelValue = NumericTraits< MeasureType >::ZeroValue();
    MeasureType minusModelValue = NumericTraits< MeasureType >::ZeroValue();

    TransformParametersType plusParameters = parameters;
    TransformParametersType minusParameters = parameters;

    plusParameters[ i ] += h;
    minusParameters[ i ] -= h;

    this->GetValue( plusModelValue, plusParameters );
    this->GetValue( minusModelValue, minusParameters );

    modelDerivative[ i ] += ( plusModelValue - minusModelValue ) / ( 2*h );
  }

  this->SetTransformParameters( parameters );
}


} // end namespace itk

#endif // end #ifndef _itkAdvancedMeanSquaresImageToImageMetric_hxx

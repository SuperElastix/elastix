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
#include "vnl_sample.h"

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

template< class TFixedImage, class TMovingImage >
void
ActiveRegistrationModelIntensityMetric< TFixedImage, TMovingImage >
::Initialize() {
  /** Initialize transform, interpolator, etc. */
  Superclass::Initialize();
} // end Initialize()


/**
 * ******************* PrintSelf *******************
 */

template< class TFixedImage, class TMovingImage >
void
ActiveRegistrationModelIntensityMetric< TFixedImage, TMovingImage >
::PrintSelf( std::ostream &os, Indent indent ) const {
  Superclass::PrintSelf( os, indent );
} // end PrintSelf()


/**
 * ******************* GetValueAndFiniteDifferenceDerivative *******************
 */

template< class TFixedImage, class TMovingImage >
void
ActiveRegistrationModelIntensityMetric< TFixedImage, TMovingImage >
::GetValueAndFiniteDifferenceDerivative( const TransformParametersType & parameters,
                                         MeasureType& value,
                                         DerivativeType& derivative ) const
{
  value = NumericTraits< MeasureType >::ZeroValue();
  derivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

  // Loop over models
  for( const auto& statisticalModel : this->GetStatisticalModelContainer()->CastToSTLConstContainer() )
  {

    // Initialize value container
    MeasureType modelValue = NumericTraits< MeasureType >::ZeroValue();
    DerivativeType modelDerivative = DerivativeType( this->GetNumberOfParameters() );
    modelDerivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

    this->GetModelValue( parameters, statisticalModel, modelValue );
    this->GetModelFiniteDifferenceDerivative( parameters, statisticalModel, modelDerivative );

    value += modelValue;
    derivative += modelDerivative;
  }

  value /= this->GetStatisticalModelContainer()->Size();
  derivative /= this->GetStatisticalModelContainer()->Size();

  elxout << "FiniteDiff: " << value << ", " << derivative << std::endl;
}






/**
 * ******************* GetValue *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
typename ActiveRegistrationModelIntensityMetric< TFixedPointSet, TMovingPointSet >::MeasureType
ActiveRegistrationModelIntensityMetric< TFixedPointSet, TMovingPointSet >
::GetValue( const TransformParametersType& parameters ) const
{
  MeasureType value = NumericTraits< MeasureType >::ZeroValue();

  for( const auto& statisticalModel : this->GetStatisticalModelContainer()->CastToSTLConstContainer() )
  {
    this->GetModelValue( parameters, statisticalModel, value );
  }

  value /= this->GetStatisticalModelContainer()->Size();

  return value;
} // end GetValue()



/**
 * ******************* GetModelValue *******************
 */

template<class TFixedImage, class TMovingImage>
void
ActiveRegistrationModelIntensityMetric<TFixedImage, TMovingImage>
::GetModelValue( const TransformParametersType& parameters,
                 const StatisticalModelPointer statisticalModel,
                 MeasureType& modelValue ) const
{

  // Make sure transform parameters are up-to-date
  this->SetTransformParameters( parameters );

  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  typename StatisticalModelType::PointValueListType fixedPointMovingImageValues;

  FixedImagePointType fixedPoint;
  MovingImagePointType movingPoint;
  RealType movingImageValue;

  for( const auto& sample : sampleContainer->CastToSTLConstContainer() )
  {
    // Transform point
    fixedPoint = sample.m_ImageCoordinates;
    bool sampleOk = this->TransformPoint( fixedPoint, movingPoint );

    // Check if movingPoint is inside moving image
    if( sampleOk ) {
      sampleOk = this->m_Interpolator->IsInsideBuffer( movingPoint );
    } else {
      continue;
    }

    // Check if movingPoint is inside moving mask if moving mask is used
    if( sampleOk ) {
      sampleOk = this->IsInsideMovingMask(movingPoint);
    } else {
      continue;
    }

    // Sample moving image
    if( sampleOk ) {
      sampleOk = this->EvaluateMovingImageValueAndDerivative( movingPoint, movingImageValue, nullptr );
    } else {
      continue;
    }

    if( sampleOk )
    {
      fixedPointMovingImageValues.emplace_back( fixedPoint, movingImageValue );
    }
  }

  this->CheckNumberOfSamples( sampleContainer->Size(), fixedPointMovingImageValues.size() );

  const auto coeffs = statisticalModel->ComputeCoefficientsForPointValues( fixedPointMovingImageValues, statisticalModel->GetNoiseVariance() );

  // tmp = sum_J (M_j - mu_j) * (I - V_j V_j^T) * (M_j - mu_j)
  RealType tmp = 0;
  for( const auto& fixedPointMovingImageValue : fixedPointMovingImageValues ) {
    const auto& fixedPoint = fixedPointMovingImageValue.first;
    const auto& movingImageValue = fixedPointMovingImageValue.second;

    tmp += ( movingImageValue - statisticalModel->DrawMeanAtPoint( fixedPoint ) ) *
           ( movingImageValue - statisticalModel->DrawSampleAtPoint( coeffs, fixedPoint, true ) );
  }

  if( fixedPointMovingImageValues.size() > 0 )
  {
    modelValue += tmp / fixedPointMovingImageValues.size();
  }

} // end GetModelValue()


/**
 * ******************* GetModelFiniteDifferenceDerivative *******************
 */

template< class TFixedImage, class TMovingImage >
void
ActiveRegistrationModelIntensityMetric< TFixedImage, TMovingImage >
::GetModelFiniteDifferenceDerivative( const TransformParametersType & parameters,
                                      const StatisticalModelPointer statisticalModel,
                                      DerivativeType& modelDerivative ) const
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

    this->GetModelValue( plusParameters, statisticalModel, plusModelValue );
    this->GetModelValue( minusParameters, statisticalModel, minusModelValue );

    modelDerivative[ i ] += ( plusModelValue - minusModelValue ) / ( 2 * h );
  }

  this->SetTransformParameters( parameters );
}







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
  MeasureType & value, DerivativeType & derivative ) const {

  this->SetTransformParameters( parameters );

  value = NumericTraits< MeasureType >::ZeroValue();
  derivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

  DerivativeType Jacobian( this->GetTransform()->GetNumberOfNonZeroJacobianIndices() );
  NonZeroJacobianIndicesType nzji( this->GetTransform()->GetNumberOfNonZeroJacobianIndices() );

  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  // Loop over models
  for( const auto& statisticalModel : this->GetStatisticalModelContainer()->CastToSTLConstContainer() )
  {
    // Initialize value container
    MeasureType modelValue = NumericTraits< MeasureType >::ZeroValue();
    DerivativeType modelDerivative = DerivativeType( this->GetNumberOfParameters() );
    modelDerivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

    typename StatisticalModelType::PointValueListType fixedPointMovingImageValues;
    typename std::vector< MovingImageDerivativeType > movingImageDerivatives;

    for( const auto& sample : sampleContainer->CastToSTLConstContainer() )
    {
      MovingImagePointType movingPoint;
      RealType movingImageValue;
      MovingImageDerivativeType movingImageDerivative;

      // Transform point
      const FixedImagePointType& fixedPoint = sample.m_ImageCoordinates;
      bool sampleOk = this->TransformPoint( fixedPoint, movingPoint );

      // Check if movingPoint is inside moving image
      if( sampleOk ) {
        sampleOk = this->m_Interpolator->IsInsideBuffer( movingPoint );
      } else {
        continue;
      }

      // Check if movingPoint is inside moving mask if moving mask is used
      if( sampleOk ) {
        sampleOk = this->IsInsideMovingMask(movingPoint);
      } else {
        continue;
      }

      // Sample moving image
      if( sampleOk ) {
        sampleOk = this->EvaluateMovingImageValueAndDerivative(movingPoint, movingImageValue, &movingImageDerivative);
      } else {
        continue;
      }

      if( sampleOk )
      {
        fixedPointMovingImageValues.emplace_back( fixedPoint, movingImageValue );
        movingImageDerivatives.emplace_back( movingImageDerivative );
      }
    }

    this->CheckNumberOfSamples( sampleContainer->Size(), fixedPointMovingImageValues.size() );

    const auto coeffs = statisticalModel->ComputeCoefficientsForPointValues( fixedPointMovingImageValues, statisticalModel->GetNoiseVariance() );

    for( auto it = std::make_pair( fixedPointMovingImageValues.begin(), movingImageDerivatives.begin() );
         it.first != fixedPointMovingImageValues.end();
         it.first++, it.second++) {

      const FixedImagePointType& fixedPoint = it.first->first;
      const RealType& movingImageValue = it.first->second;
      const MovingImageDerivativeType& movingImageDerivative = *it.second;

      // tmp = (M_j - mu_j) * (I - V_j V_j^T)
      RealType tmp = movingImageValue - statisticalModel->DrawSampleAtPoint( coeffs, fixedPoint, true );
      modelValue += ( movingImageValue - statisticalModel->DrawMeanAtPoint( fixedPoint ) ) * tmp;

      // (dM/d{x,y,z})(dT/du)
      this->m_AdvancedTransform->EvaluateJacobianWithImageGradientProduct( fixedPoint, movingImageDerivative, Jacobian, nzji );

      // Loop over Jacobian
      for( unsigned int i = 0; i < nzji.size(); ++i )
      {
        const unsigned int& mu = nzji[ i ];
        modelDerivative[ mu ] += tmp * Jacobian[ i ];
      }
    }

    value += modelValue / fixedPointMovingImageValues.size();
    derivative += 2.0 * modelDerivative / fixedPointMovingImageValues.size();
  }

  value /= this->GetStatisticalModelContainer()->Size();
  derivative /= this->GetStatisticalModelContainer()->Size();

  const bool useFiniteDifferenceDerivative = false;
  if (useFiniteDifferenceDerivative) {
    elxout << "Analytical: " << value << ", " << derivative << std::endl;
    this->GetValueAndFiniteDifferenceDerivative( parameters, value, derivative );
    elxout << "Parameters: " << parameters << std::endl;
  }

  return;
}  // end GetValueAndDerivative()

} // end namespace itk

#endif // end #ifndef _itkAdvancedMeanSquaresImageToImageMetric_hxx

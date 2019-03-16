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
  for(std::tuple< StatisticalModelVectorContainerConstIterator,
          StatisticalModelMatrixContainerConstIterator,
          StatisticalModelScalarContainerConstIterator,
          StatisticalModelRepresenterContainerConstIterator > it =
          std::make_tuple( this->GetMeanVectorContainer()->Begin(),
                           this->GetBasisMatrixContainer()->Begin(),
                           this->GetNoiseVarianceContainer()->Begin(),
                           this->GetRepresenterContainer()->Begin() );
      std::get<0>(it) != this->GetMeanVectorContainer()->End();
      ++std::get<0>(it), ++std::get<1>(it), ++std::get<2>(it), ++std::get<3>(it))
  {
    const StatisticalModelVectorType& meanVector = std::get<0>(it)->Value();
    const StatisticalModelMatrixType& basisMatrix = std::get<1>(it)->Value();
    const StatisticalModelScalarType& noiseVariance = std::get<2>(it)->Value();
    const StatisticalModelRepresenterPointer representer = std::get<3>(it)->Value();

    // Initialize value container
    MeasureType modelValue = NumericTraits< MeasureType >::ZeroValue();
    DerivativeType modelDerivative = DerivativeType( this->GetNumberOfParameters() );
    modelDerivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

    this->GetModelValue( meanVector, basisMatrix, noiseVariance, representer, modelValue, parameters );
    this->GetModelFiniteDifferenceDerivative( meanVector, basisMatrix, noiseVariance, representer, modelDerivative, parameters );

    value += modelValue;
    derivative += modelDerivative;
  }

  value /= this->GetMeanVectorContainer()->Size();
  derivative /= this->GetMeanVectorContainer()->Size();

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

  // Loop over models
  for(std::tuple< StatisticalModelVectorContainerConstIterator,
          StatisticalModelMatrixContainerConstIterator,
          StatisticalModelScalarContainerConstIterator,
          StatisticalModelRepresenterContainerConstIterator > it =
          std::make_tuple( this->GetMeanVectorContainer()->Begin(),
                           this->GetBasisMatrixContainer()->Begin(),
                           this->GetNoiseVarianceContainer()->Begin(),
                           this->GetRepresenterContainer()->Begin() );
      std::get<0>(it) != this->GetMeanVectorContainer()->End();
      ++std::get<0>(it), ++std::get<1>(it), ++std::get<2>(it), ++std::get<3>(it))
  {
    const StatisticalModelVectorType& meanVector = std::get<0>(it)->Value();
    const StatisticalModelMatrixType& basisMatrix = std::get<1>(it)->Value();
    const StatisticalModelScalarType& noiseVariance = std::get<2>(it)->Value();
    const StatisticalModelRepresenterPointer representer = std::get<3>(it)->Value();

    // Initialize value container
    MeasureType modelValue = NumericTraits< MeasureType >::ZeroValue();
    DerivativeType modelDerivative = DerivativeType( this->GetNumberOfParameters() );
    modelDerivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

    this->GetModelValue( meanVector, basisMatrix, noiseVariance, representer, modelValue, parameters );

    value += modelValue;
  }

  value /= this->GetMeanVectorContainer()->Size();

  return value;
} // end GetValue()



/**
 * ******************* GetModelValue *******************
 */

template<class TFixedImage, class TMovingImage>
void
ActiveRegistrationModelIntensityMetric<TFixedImage, TMovingImage>
::GetModelValue( const StatisticalModelVectorType& meanVector,
                 const StatisticalModelMatrixType& basisMatrix,
                 const StatisticalModelScalarType& noiseVariance,
                 const StatisticalModelRepresenterPointer representer,
                 MeasureType& modelValue,
                 const TransformParametersType& parameters ) const {

  // Make sure transform parameters are up-to-date
  this->SetTransformParameters( parameters );

  unsigned int numberOfOkSamples = 0;

  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  typename ImageSampleContainerType::ConstIterator fixedSampleContainerIterator = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fixedSampleContainerIteratorEnd = sampleContainer->End();
  while( fixedSampleContainerIterator != fixedSampleContainerIteratorEnd ) {
    // Transform point
    const FixedImagePointType& fixedPoint = fixedSampleContainerIterator->Value().m_ImageCoordinates;
    MovingImagePointType movingPoint;
    RealType movingImageValue;
    bool sampleOk = this->TransformPoint( fixedPoint, movingPoint );

    // Check if movingPoint is inside moving image
    if( sampleOk )
    {
      sampleOk = this->m_Interpolator->IsInsideBuffer( movingPoint );
    }
    else
    {
      fixedSampleContainerIterator++;
      continue;
    }

    // Check if movingPoint is inside moving mask if moving mask is used
    if( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( movingPoint );
    }
    else
    {
      fixedSampleContainerIterator++;
      continue;
    }

    // Sample moving image
    if( sampleOk )
    {
      sampleOk = this->EvaluateMovingImageValueAndDerivative( movingPoint, movingImageValue, nullptr );
    }
    else
    {
      fixedSampleContainerIterator++;
      continue;
    }

    if( sampleOk )
    {
      RealType epsilon = 0;

      if( noiseVariance > 0 )
      {
        epsilon = vnl_sample_normal(0., 1.);
      }

      const unsigned int pointId = representer->GetPointIdForPoint( fixedPoint );
      movingImageValue -= meanVector[ pointId ];
      const StatisticalModelVectorType basisVector = basisMatrix.get_row( pointId );
      modelValue += movingImageValue * ( 1.0 - ( dot_product( basisVector, basisVector ) + epsilon ) ) * movingImageValue;

      numberOfOkSamples++;
    }

    fixedSampleContainerIterator++;
  }

  // Check number of samples
  this->CheckNumberOfSamples( sampleContainer->Size(), numberOfOkSamples );

  if( numberOfOkSamples > 0 )
  {
    modelValue /= numberOfOkSamples;
  }

} // end GetModelValue()


/**
 * ******************* GetModelFiniteDifferenceDerivative *******************
 */

template< class TFixedImage, class TMovingImage >
void
ActiveRegistrationModelIntensityMetric< TFixedImage, TMovingImage >
::GetModelFiniteDifferenceDerivative( const StatisticalModelVectorType& meanVector,
                                      const StatisticalModelMatrixType& basisMatrix,
                                      const StatisticalModelScalarType& noiseVariance,
                                      const StatisticalModelRepresenterPointer representer,
                                      DerivativeType& modelDerivative,
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

    this->GetModelValue( meanVector, basisMatrix, noiseVariance, representer, plusModelValue, plusParameters );
    this->GetModelValue( meanVector, basisMatrix, noiseVariance, representer, minusModelValue, minusParameters );

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

  MovingImagePointType movingPoint;
  RealType movingImageValue;
  MovingImageDerivativeType  movingImageDerivative;

  // Loop over models
  for(std::tuple< StatisticalModelVectorContainerConstIterator,
          StatisticalModelMatrixContainerConstIterator,
          StatisticalModelScalarContainerConstIterator,
          StatisticalModelRepresenterContainerConstIterator > it =
          std::make_tuple( this->GetMeanVectorContainer()->Begin(),
                           this->GetBasisMatrixContainer()->Begin(),
                           this->GetNoiseVarianceContainer()->Begin(),
                           this->GetRepresenterContainer()->Begin() );
      std::get<0>(it) != this->GetMeanVectorContainer()->End();
      ++std::get<0>(it), ++std::get<1>(it), ++std::get<2>(it), ++std::get<3>(it)) {
    const StatisticalModelVectorType& meanVector = std::get<0>(it)->Value();
    const StatisticalModelMatrixType& basisMatrix = std::get<1>(it)->Value();
    const StatisticalModelScalarType& noiseVariance = std::get<2>(it)->Value();
    const StatisticalModelRepresenterPointer representer = std::get<3>(it)->Value();

    RealType modelValue = NumericTraits< RealType >::ZeroValue();
    DerivativeType modelDerivative = DerivativeType( this->GetNumberOfParameters() );
    modelDerivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

    unsigned int numberOfOkSamples = 0;

    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
    typename ImageSampleContainerType::ConstIterator fixedSampleContainerIterator = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator fixedSampleContainerIteratorEnd = sampleContainer->End();
    while( fixedSampleContainerIterator != fixedSampleContainerIteratorEnd )
    {
      // Transform point
      const FixedImagePointType& fixedPoint = fixedSampleContainerIterator->Value().m_ImageCoordinates;
      bool sampleOk = this->TransformPoint( fixedPoint, movingPoint );

      // Check if movingPoint is inside moving image
      if( sampleOk )
      {
        sampleOk = this->m_Interpolator->IsInsideBuffer( movingPoint );
      }
      else
      {
        fixedSampleContainerIterator++;
        continue;
      }

      // Check if movingPoint is inside moving mask if moving mask is used
      if( sampleOk )
      {
        sampleOk = this->IsInsideMovingMask( movingPoint );
      }
      else
      {
        fixedSampleContainerIterator++;
        continue;
      }

      // Sample moving image
      if( sampleOk )
      {
        sampleOk = this->EvaluateMovingImageValueAndDerivative( movingPoint, movingImageValue, &movingImageDerivative );
      }
      else
      {
        fixedSampleContainerIterator++;
        continue;
      }

      if( sampleOk )
      {
        RealType epsilon = 0;

        if( noiseVariance > 0 )
        {
          epsilon = vnl_sample_normal(0., 1.);
        }

        const unsigned int pointId = representer->GetPointIdForPoint( fixedPoint );
        movingImageValue -= meanVector[ pointId ];
        const StatisticalModelVectorType basisVector = basisMatrix.get_row( pointId );
        RealType tmp =  movingImageValue * ( 1.0 - ( dot_product( basisVector, basisVector ) + epsilon ) );
        modelValue += tmp * movingImageValue;

        // (dM/d{x,y,z})(dT/du)
        this->m_AdvancedTransform->EvaluateJacobianWithImageGradientProduct( fixedPoint, movingImageDerivative, Jacobian, nzji );

        // Loop over Jacobian
        for( unsigned int i = 0; i < nzji.size(); ++i )
        {
          const unsigned int mu = nzji[ i ];
          modelDerivative[ mu ] += tmp * Jacobian[ i ];
        }

        numberOfOkSamples++;
      }

      fixedSampleContainerIterator++;
    }

    // Check number of samples
    this->CheckNumberOfSamples( sampleContainer->Size(), numberOfOkSamples );

    if( numberOfOkSamples > 0 )
    {
      value += modelValue / numberOfOkSamples;
      derivative += 2.0 * modelDerivative / numberOfOkSamples;
    }
  }

  value /= this->GetMeanVectorContainer()->Size();
  derivative /= this->GetMeanVectorContainer()->Size();

  const bool useFiniteDifferenceDerivative = true;
  if (useFiniteDifferenceDerivative) {
    elxout << "Analytical: " << value << ", " << derivative << std::endl;
    this->GetValueAndFiniteDifferenceDerivative( parameters, value, derivative );
  }

  return;

//  StatisticalModelMatrixType I(FixedImageDimension, FixedImageDimension);
//  I.set_identity();
//
//  // Make sure transform parameters are up to date
//  this->SetTransformParameters( parameters );
//
//  TransformJacobianType Jacobian;
//  NonZeroJacobianIndicesType nzji( this->GetTransform()->GetNumberOfNonZeroJacobianIndices() );
//  DerivativeType imageJacobian( nzji.size() );
//
//  unsigned int numberOfSamples = 0u;
//  unsigned int numberOfPrincipalComponents;
//
//  double movingImageValueInnerProduct = 0.0;
//
//  ImageSampleContainerPointer fixedSampleContainer = this->GetImageSampler()->GetOutput();
//  this->GetImageSampler()->Update();
//  typename ImageSampleContainerType::ConstIterator fixedSampleContainerIterator = fixedSampleContainer->Begin();
//  typename ImageSampleContainerType::ConstIterator fixedSampleContainerIteratorEnd = fixedSampleContainer->End();
//  while( fixedSampleContainerIterator != fixedSampleContainerIteratorEnd )
//  {
//    const FixedImagePointType& fixedPoint = fixedSampleContainerIterator->Value().m_ImageCoordinates;
//    MovingImagePointType       movingPoint;
//    RealType                   movingImageValue;
//    MovingImageDerivativeType  movingImageDerivative;
//
//    bool sampleOk = this->TransformPoint( fixedPoint, movingPoint );
//    if( sampleOk )
//    {
//      sampleOk = this->EvaluateMovingImageValueAndDerivative( movingPoint, movingImageValue, &movingImageDerivative );
//    }
//
//    if( sampleOk )
//    {
//      const unsigned int pointId = this->GetStatisticalModelContainer()->ElementAt( this->GetLevel() )->GetRepresenter()->GetPointIdForPoint( fixedSampleContainerIterator->Value().m_ImageCoordinates );
//
//      // M'M
//      movingImageValue -= this->GetStatisticalModelContainer()->ElementAt( this->GetLevel() )->DrawMeanAtPoint( pointId );
//      movingImageValueInnerProduct += movingImageValue * movingImageValue;
//
//      // I-VV^T
//      const StatisticalModelVectorType PCABasis = this->GetStatisticalModelOrthonormalPCABasisMatrixContainer()->ElementAt( this->GetLevel() ).get_row( pointId );
//      // const double intensityModelReconstructionFactor = movingImageValue * ( 1.0 - dot_product( PCABasis, PCABasis ) );
//      const double intensityModelReconstructionFactor = (movingImageValue - meanImageValue) * ( I - dot_product( PCABasis, PCABasis ) );
//
//      // (dM/dx)(dT/du)
//      this->m_AdvancedTransform->EvaluateJacobianWithImageGradientProduct( fixedPoint, movingImageDerivative, imageJacobian, nzji );
//
//      // Loop over Jacobian
//      for( unsigned int i = 0; i < imageJacobian.GetSize(); ++i )
//      {
//        const unsigned int mu = nzji[ i ];
//        derivative[ mu ] += intensityModelReconstructionFactor * imageJacobian[ i ];
//      }
//
//      ++numberOfSamples;
//    }
//
//    ++fixedSampleContainerIterator;
//  }
//
//  if( std::isnan( value ) )
//  {
//    itkExceptionMacro( "Model value is NaN.");
//  }
//
//  if( numberOfSamples > 0 )
//  {
//    value = movingImageValueInnerProduct / numberOfSamples;
//    derivative *= 2.0 / numberOfSamples;
//  }

}  // end GetValueAndDerivative()


} // end namespace itk

#endif // end #ifndef _itkAdvancedMeanSquaresImageToImageMetric_hxx

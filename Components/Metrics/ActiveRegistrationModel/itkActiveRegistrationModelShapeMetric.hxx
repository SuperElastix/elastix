/*======================================================================

This file is part of the elastix software.

Copyright (c) University Medical Center Utrecht. All rights reserved.
See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkActiveRegistrationModelShapeMetric_hxx__
#define __itkActiveRegistrationModelShapeMetric_hxx__

#include "itkActiveRegistrationModelShapeMetric.h"
#include "vnl_sample.h"

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
ActiveRegistrationModelShapeMetric< TFixedPointSet, TMovingPointSet >
::ActiveRegistrationModelShapeMetric()
{
} // end Constructor


/**
 * ******************* Destructor *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
ActiveRegistrationModelShapeMetric< TFixedPointSet, TMovingPointSet  >
::~ActiveRegistrationModelShapeMetric()
{} // end Destructor

/**
 * *********************** Initialize *****************************
 */

template< class TFixedPointSet, class TMovingPointSet  >
void
ActiveRegistrationModelShapeMetric< TFixedPointSet, TMovingPointSet >
::Initialize( void )
{
  if( !this->GetTransform() )
  {
    itkExceptionMacro( << "Transform is not present" );
  }

  if( this->GetMeanVectorContainer()->Size() == 0  )
  {
    itkExceptionMacro( << "MeanVectorContainer is empty." );
  }

  if( this->GetBasisMatrixContainer()->Size() == 0  )
  {
    itkExceptionMacro( << "BasisMatrixContainer is empty." );
  }

  if( this->GetVarianceContainer()->Size() == 0  )
  {
    itkExceptionMacro( << "VarianceContainer is empty." );
  }

  if( this->GetNoiseVarianceContainer()->Size() == 0  )
  {
    itkExceptionMacro( << "NoiseVarianceContainer is empty." );
  }
} // end Initialize()



/**
 * ******************* GetValue *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
typename ActiveRegistrationModelShapeMetric< TFixedPointSet, TMovingPointSet >::MeasureType
ActiveRegistrationModelShapeMetric< TFixedPointSet, TMovingPointSet >
::GetValue( const TransformParametersType& parameters ) const
{
  MeasureType value = NumericTraits< MeasureType >::Zero;

  // Loop over models
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

template< class TFixedPointSet, class TMovingPointSet >
void
ActiveRegistrationModelShapeMetric< TFixedPointSet, TMovingPointSet >
::GetModelValue( const TransformParametersType& parameters,
                 const StatisticalModelPointer statisticalModel,
                 MeasureType& modelValue ) const
{
  // Make sure transform parameters are up-to-date
  this->SetTransformParameters( parameters );

  const auto& meanMesh = statisticalModel->DrawMean();
  const auto& meanVector = statisticalModel->GetRepresenter()->SampleToSampleVector( meanMesh );

  const auto& movingMeanMesh = this->TransformMesh( meanMesh );
  const auto& movingMeanVector = statisticalModel->GetRepresenter()->SampleToSampleVector( this->TransformMesh( movingMeanMesh ) );

  const auto& coeffs = statisticalModel->ComputeCoefficients( movingMeanMesh );
  const auto& reconstructedVector = statisticalModel->GetRepresenter()->SampleToSampleVector( statisticalModel->DrawSample( coeffs, true ) );

  modelValue += ( movingMeanVector - meanVector ).dot( movingMeanVector - reconstructedVector ) / meanMesh->GetNumberOfPoints();
}


/**
 * ******************* GetValueAndFiniteDifferenceDerivative *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
void
ActiveRegistrationModelShapeMetric< TFixedPointSet, TMovingPointSet >
::GetValueAndFiniteDifferenceDerivative( const TransformParametersType& parameters,
                                         MeasureType& value,
                                         DerivativeType& derivative ) const
{
  value = NumericTraits< MeasureType >::ZeroValue();
  derivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

  for( const auto& statisticalModel : this->GetStatisticalModelContainer()->CastToSTLConstContainer() )
  {
    MeasureType modelValue = NumericTraits< MeasureType >::ZeroValue();
    DerivativeType modelDerivative = DerivativeType( this->GetNumberOfParameters() );
    modelDerivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

    this->GetModelValue( parameters, statisticalModel, value );
    this->GetModelFiniteDifferenceDerivative( parameters, statisticalModel, modelDerivative );

    value += modelValue;
    derivative += modelDerivative;
  }

  value /= this->GetStatisticalModelContainer()->Size();
  derivative /= this->GetStatisticalModelContainer()->Size();

  elxout << "FiniteDiff: " << value << ", " << derivative << std::endl;
}



/**
 * ******************* GetModelFiniteDifferenceDerivative *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
void
ActiveRegistrationModelShapeMetric< TFixedPointSet, TMovingPointSet >
::GetModelFiniteDifferenceDerivative( const TransformParametersType & parameters,
                                      const StatisticalModelPointer statisticalModel,
                                      DerivativeType& modelDerivative ) const
{
  const DerivativeValueType h = 0.01;

  for( unsigned int i = 0; i < parameters.size(); ++i )\
  {
    MeasureType plusModelValue = NumericTraits< DerivativeValueType >::ZeroValue();
    MeasureType minusModelValue = NumericTraits< DerivativeValueType >::ZeroValue();

    TransformParametersType plusParameters = parameters;
    TransformParametersType minusParameters = parameters;

    plusParameters[ i ] += h;
    minusParameters[ i ] -= h;

    this->GetModelValue( plusParameters, statisticalModel, plusModelValue );
    this->GetModelValue( minusParameters, statisticalModel, minusModelValue );

    modelDerivative[ i ] += ( plusModelValue - minusModelValue ) / ( 2 * h );
  }

  // Reset transform parameters
  this->SetTransformParameters( parameters );
}



/**
 * ******************* GetDerivative *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
void
ActiveRegistrationModelShapeMetric< TFixedPointSet, TMovingPointSet >
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

template< class TFixedPointSet, class TMovingPointSet >
void
ActiveRegistrationModelShapeMetric< TFixedPointSet, TMovingPointSet >
::GetValueAndDerivative( const TransformParametersType& parameters,
                         MeasureType& value,
                         DerivativeType& derivative ) const
{
  this->SetTransformParameters( parameters );

  value = NumericTraits< MeasureType >::ZeroValue();
  derivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

  TransformJacobianType Jacobian;
  NonZeroJacobianIndicesType nzji( this->GetTransform()->GetNumberOfNonZeroJacobianIndices() );

  // Loop over models
  for(std::tuple< StatisticalModelVectorContainerConstIterator,
          StatisticalModelMatrixContainerConstIterator,
          StatisticalModelScalarContainerConstIterator > it =
          std::make_tuple( this->GetMeanVectorContainer()->Begin(),
                           this->GetBasisMatrixContainer()->Begin(),
                           this->GetNoiseVarianceContainer()->Begin() );
      std::get<0>(it) != this->GetMeanVectorContainer()->End();
      ++std::get<0>(it), ++std::get<1>(it), ++std::get<2>(it))
  {
    const StatisticalModelVectorType& meanVector = std::get<0>(it)->Value();
    const StatisticalModelMatrixType& basisMatrix = std::get<1>(it)->Value();
    const StatisticalModelScalarType& noiseVariance = std::get<2>(it)->Value();

    DerivativeType modelDerivative = DerivativeType( this->GetNumberOfParameters() );
    modelDerivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

    StatisticalModelVectorType movingVector = StatisticalModelVectorType( meanVector.size(), 0. );
    for( unsigned int i = 0; i < meanVector.size(); i += FixedPointSetDimension )
    {
      const auto transformedPoint = this->GetTransform()->TransformPoint( meanVector.data_block() + i );
      movingVector.update( transformedPoint.GetVnlVector(), i );
    }

    movingVector -= meanVector;

    // tmp = (T(S) - mu) * (I - VV^T)
    const StatisticalModelVectorType tmp = movingVector - this->Reconstruct(movingVector, basisMatrix, noiseVariance);
    MeasureType modelValue = dot_product(tmp, movingVector);

    for(unsigned int i = 0; i < meanVector.size(); i += FixedPointSetDimension )
    {
      this->GetTransform()->GetJacobian( meanVector.data_block() + i, Jacobian, nzji );
      if( nzji.size() == this->GetNumberOfParameters() )
      {
        modelDerivative += tmp.extract( FixedPointSetDimension, i ) * Jacobian;
      }
      else
      {
        for( unsigned int j = 0; j < nzji.size(); j++ ) {
          const auto& mu = nzji[ j ];
          modelDerivative[ mu ] += dot_product( tmp.extract( FixedPointSetDimension, i ), Jacobian.get_column( j ) );
        }
      }
    }

    if( std::isnan( modelValue ) )
    {
        itkExceptionMacro( "Model value is NaN.")
    }

    if( FixedPointSetDimension * meanVector.size() > 0 )
    {
      value += modelValue * FixedPointSetDimension / meanVector.size();
      derivative += 2.0 * modelDerivative * FixedPointSetDimension / meanVector.size();
    }
  }

  value /= this->GetMeanVectorContainer()->Size();
  derivative /= this->GetMeanVectorContainer()->Size();

  const bool useFiniteDifferenceDerivative = true;
  if( useFiniteDifferenceDerivative )
  {
    elxout << "Analytical: " << value << ", " << derivative << std::endl;
    this->GetValueAndFiniteDifferenceDerivative( parameters, value, derivative );
  }
}



/**
 * ******************* Reconstruct *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
const typename ActiveRegistrationModelShapeMetric< TFixedPointSet, TMovingPointSet >::StatisticalModelVectorType
ActiveRegistrationModelShapeMetric< TFixedPointSet, TMovingPointSet >
::Reconstruct(const StatisticalModelVectorType& movingVector, const StatisticalModelMatrixType& basisMatrix,
              const StatisticalModelScalarType& noiseVariance ) const
{
  StatisticalModelVectorType epsilon = StatisticalModelVectorType(movingVector.size(), 0.);

  if( noiseVariance > 0 ) {
    for( unsigned int i = 0; i < movingVector.size(); i++) {
      epsilon[ i ] = vnl_sample_normal(0., 1.);
    }
  }

  // Compute movingShape * VV^T without compute VV^T to reduce peak memory
  const StatisticalModelVectorType coefficients = basisMatrix.transpose() * movingVector;
  return basisMatrix * coefficients + epsilon;
};



/**
 * ******************* TransformMesh *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
typename ActiveRegistrationModelShapeMetric< TFixedPointSet, TMovingPointSet >::StatisticalModelMeshPointer
ActiveRegistrationModelShapeMetric< TFixedPointSet, TMovingPointSet >
::TransformMesh( StatisticalModelMeshPointer fixedMesh ) const
{
  StatisticalModelMeshPointer movingMesh = StatisticalModelMeshType::New();
  movingMesh->GetPoints()->Reserve( fixedMesh->GetNumberOfPoints() );

  // Transform mesh
  StatisticalModelMeshConstIteratorType fixedMeshIterator = fixedMesh->GetPoints()->Begin();
  StatisticalModelMeshConstIteratorType fixedMeshIteratorEnd = fixedMesh->GetPoints()->End();
  StatisticalModelMeshIteratorType movingMeshIterator = movingMesh->GetPoints()->Begin();
  while( fixedMeshIterator != fixedMeshIteratorEnd )
  {
    movingMeshIterator->Value() = this->GetTransform()->TransformPoint( fixedMeshIterator->Value() );
    ++fixedMeshIterator;
    ++movingMeshIterator;
  }

  return movingMesh;
}



/**
 * ******************* PrintSelf *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
void
ActiveRegistrationModelShapeMetric< TFixedPointSet, TMovingPointSet >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf(os, indent);
  // TODO
}



} // end namespace itk

#endif // end #ifndef __itkActiveRegistrationModelShapeMetric_hxx__


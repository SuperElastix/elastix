/*======================================================================

This file is part of the elastix software.

Copyright (c) University Medical Center Utrecht. All rights reserved.
See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkActiveRegistrationModelPointDistributionShapeMetric_hxx__
#define __itkActiveRegistrationModelPointDistributionShapeMetric_hxx__

#include "itkActiveRegistrationModelShapeMetric.h"
#include "math.h"

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

  if( !this->GetStatisticalModelContainer()  )
  {
    itkExceptionMacro( << "StatisticalModelContainer has not been assigned." );
  }

  if( !this->GetStatisticalModelContainer()->Size() )
  {
    itkExceptionMacro( << "StatisticalModelContainer is empty." );
  }
} // end Initialize()





/**
 * ******************* GetValue *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
typename ActiveRegistrationModelShapeMetric< TFixedPointSet, TMovingPointSet >::MeasureType
ActiveRegistrationModelShapeMetric< TFixedPointSet, TMovingPointSet >
::GetValue( const TransformParametersType & parameters ) const
{
  MeasureType value = NumericTraits< MeasureType >::Zero;

  // Loop over models
  StatisticalModelContainerConstIterator statisticalModelIterator = this->GetStatisticalModelContainer()->Begin();
  StatisticalModelContainerConstIterator statisticalModelIteratorEnd = this->GetStatisticalModelContainer()->End();
  while( statisticalModelIterator != statisticalModelIteratorEnd )
  {
    // GetValue
    this->GetModelValue( statisticalModelIterator->Value(), value, parameters );
    ++statisticalModelIterator;
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
::GetModelValue( StatisticalModelConstPointer statisticalModel,
                 MeasureType & value,
                 const TransformParametersType & parameters ) const
{
  MeasureType modelValue = NumericTraits< MeasureType >::ZeroValue();

  // Make sure transform parameters are up-to-date
  this->SetTransformParameters( parameters );

  // Get model reconstruction of movingMesh
  const StatisticalModelMeshPointer meanMesh = statisticalModel->DrawMean();
  StatisticalModelMeshPointer movingMesh = StatisticalModelMeshType::New();
  movingMesh->GetPoints()->Reserve( meanMesh->GetNumberOfPoints() );

  // Warp fixed mesh
  StatisticalModelMeshConstIteratorType meanMeshIterator = meanMesh->GetPoints()->Begin();
  StatisticalModelMeshConstIteratorType meanMeshIteratorEnd = meanMesh->GetPoints()->End();
  StatisticalModelMeshIteratorType movingMeshIterator = movingMesh->GetPoints()->Begin();
  while( meanMeshIterator != meanMeshIteratorEnd )
  {
    movingMeshIterator->Value() = this->GetTransform()->TransformPoint( meanMeshIterator->Value() );
    ++meanMeshIterator;
    ++movingMeshIterator;
  }

  // Compute coefficients of model synthesized shape
  const StatisticalModelVectorType modelCoefficients = statisticalModel->ComputeCoefficients( movingMesh );

  // Get value f(X)
  meanMeshIterator = meanMesh->GetPoints()->Begin();
  StatisticalModelMeshConstIteratorType referenceMeshIterator = statisticalModel->GetRepresenter()->GetReference()->GetPoints()->Begin();
  movingMeshIterator = movingMesh->GetPoints()->Begin();
  while( meanMeshIterator != meanMeshIteratorEnd )
  {
    // f_i(X)
    const StatisticalModelPointType reconstructedPoint = statisticalModel->DrawSampleAtPoint( modelCoefficients, referenceMeshIterator->Value(), false );
    const StatisticalModelVectorType reconstructionError = movingMeshIterator->Value().GetVnlVector() - reconstructedPoint.GetVnlVector();

    // Value
    modelValue += reconstructionError.squared_magnitude();

    ++meanMeshIterator;
    ++referenceMeshIterator;
    ++movingMeshIterator;
  }

  value += modelValue / meanMesh->GetNumberOfPoints();
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
::GetValueAndDerivative( const TransformParametersType & parameters,
                         MeasureType & value,
                         DerivativeType & derivative ) const
{
  this->SetTransformParameters( parameters );

  value = NumericTraits< MeasureType >::ZeroValue();
  derivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

  TransformJacobianType Jacobian;
  NonZeroJacobianIndicesType nzji( this->GetTransform()->GetNumberOfNonZeroJacobianIndices() );

  // Loop over models
  StatisticalModelContainerConstIterator statisticalModelIterator = this->GetStatisticalModelContainer()->Begin();
  StatisticalModelContainerConstIterator statisticalModelIteratorEnd = this->GetStatisticalModelContainer()->End();
  while( statisticalModelIterator != statisticalModelIteratorEnd )
  {
    MeasureType modelValue;
    DerivativeType modelDerivative = DerivativeType( this->GetNumberOfParameters() );
    modelDerivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

    const auto meanShape = statisticalModelIterator->Value()->GetMeanVector();
    StatisticalModelVectorType movingShape = StatisticalModelVectorType( meanShape.size(), 0. );
    for( unsigned int i = 0; i < meanShape.size(); i += FixedPointSetDimension )
    {
      const auto transformedPoint = this->GetTransform()->TransformPoint( meanShape.data_block() + i );
      movingShape.update( transformedPoint.GetVnlVector(), i );
    }

    movingShape -= meanShape;

    // tmp = (T(S) - mu) * (I - VV^T)
    const StatisticalModelMatrixType PCABasisMatrix = statisticalModelIterator->Value()->GetOrthonormalPCABasisMatrix();
    const StatisticalModelVectorType tmp = movingShape - this->Reconstruct(movingShape, PCABasisMatrix);
    modelValue = dot_product(tmp, movingShape);

    for(unsigned int i = 0; i < meanShape.size(); i += FixedPointSetDimension)
    {
      this->GetTransform()->GetJacobian( meanShape.data_block() + i, Jacobian, nzji );
      for( unsigned int j = 0; j < nzji.size(); j++ ) {
        const auto& mu = nzji[ j ];
        modelDerivative[ mu ] += dot_product(tmp.extract( FixedPointSetDimension, i ), Jacobian.get_column( j ));
      }
    }

    if( std::isnan( modelValue ) )
    {
        itkExceptionMacro( "Model value is NaN.")
    }

    if( FixedPointSetDimension * meanShape.size() > 0 )
    {
      value += modelValue * FixedPointSetDimension / meanShape.size();
      derivative += 2.0 * modelDerivative * FixedPointSetDimension / meanShape.size();
    }

    ++statisticalModelIterator;
  }

  value /= this->GetStatisticalModelContainer()->Size();
  derivative /= this->GetStatisticalModelContainer()->Size();

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
::Reconstruct(const StatisticalModelVectorType& movingShape, const StatisticalModelMatrixType& basisMatrix) const
{
  const StatisticalModelVectorType tmp = movingShape * basisMatrix;
  return tmp * basisMatrix.transpose();
};

/**
 * ******************* GetValueAndFiniteDifferenceDerivative *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
void
ActiveRegistrationModelShapeMetric< TFixedPointSet, TMovingPointSet >
::GetValueAndFiniteDifferenceDerivative( const TransformParametersType & parameters,
                                         MeasureType & value,
                                         DerivativeType & derivative ) const
{
  // Loop over models
  StatisticalModelContainerConstIterator statisticalModelIterator = this->GetStatisticalModelContainer()->Begin();
  StatisticalModelContainerConstIterator statisticalModelIteratorEnd = this->GetStatisticalModelContainer()->End();
  while( statisticalModelIterator != statisticalModelIteratorEnd )
  {
    // Initialize value container
    MeasureType modelValue = NumericTraits< MeasureType >::ZeroValue();
    DerivativeType modelDerivative = DerivativeType( this->GetNumberOfParameters() );
    modelDerivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

    this->GetModelValue( statisticalModelIterator->Value(), modelValue, parameters );
    this->GetModelFiniteDifferenceDerivative( statisticalModelIterator->Value(), modelDerivative, parameters );

    value = modelValue / this->GetStatisticalModelContainer()->Size();
    derivative = modelDerivative / this->GetStatisticalModelContainer()->Size();

    ++statisticalModelIterator;
  }

  elxout << "FiniteDiff: " << value << ", " << derivative << std::endl;
}

/**
 * ******************* GetModelFiniteDifferenceDerivative *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
void
ActiveRegistrationModelShapeMetric< TFixedPointSet, TMovingPointSet >
::GetModelFiniteDifferenceDerivative( StatisticalModelConstPointer statisticalModel,
                                      DerivativeType & modelDerivative,
                                      const TransformParametersType & parameters ) const
{
  const double h = 0.01;

  // Get derivative (J(X)-W*(inv(C)*(W^T*J(X))))^T*f(X)
  for( unsigned int i = 0; i < parameters.size(); ++i )\
  {
    MeasureType plusModelValue = NumericTraits< MeasureType >::ZeroValue();
    MeasureType minusModelValue = NumericTraits< MeasureType >::ZeroValue();

    TransformParametersType plusParameters = parameters;
    TransformParametersType minusParameters = parameters;

    plusParameters[ i ] += h;
    minusParameters[ i ] -= h;

    this->GetModelValue( statisticalModel, plusModelValue, plusParameters );
    this->GetModelValue( statisticalModel, minusModelValue, minusParameters );

    modelDerivative[ i ] += ( plusModelValue - minusModelValue ) / ( 2*h );
  }

  // Reset transform parameters
  this->SetTransformParameters( parameters );
}



/**
 * ******************* TransformMesh *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
void
ActiveRegistrationModelShapeMetric< TFixedPointSet, TMovingPointSet >
::TransformMesh( StatisticalModelMeshPointer fixedMesh, StatisticalModelMeshPointer movingMesh ) const
{
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
}



/**
 * ******************* PrintSelf *******************
 */

template< class TFixedPointSet, class TMovingPointSet >
void
ActiveRegistrationModelShapeMetric< TFixedPointSet, TMovingPointSet >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
  //
  //   if ( this->m_ComputeSquaredDistance )
  //   {
  //     os << indent << "m_ComputeSquaredDistance: True" << std::endl;
  //   }
  //   else
  //   {
  //     os << indent << "m_ComputeSquaredDistance: False" << std::endl;
  //   }
} // end PrintSelf()



} // end namespace itk

#endif // end #ifndef __ActiveRegistrationModelShapeMetric_hxx__


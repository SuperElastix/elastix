/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkRecursiveBSplineTransform_hxx
#define __itkRecursiveBSplineTransform_hxx

#include "itkRecursiveBSplineTransform.h"


namespace itk
{

/**
 * ********************* Constructor ****************************
 */

template <typename TScalar, unsigned int NDimensions, unsigned int VSplineOrder>
RecursiveBSplineTransform< TScalar, NDimensions, VSplineOrder >
::RecursiveBSplineTransform() : Superclass()
{
  this->m_RecursiveBSplineWeightFunction = RecursiveBSplineWeightFunctionType::New();
} // end Constructor()


/**
 * ********************* TransformPoint ****************************
 */

// MS: this is slightly different from AdvancedBSplineDeformableTransform
// should we update that one and delete this one?
template <typename TScalar, unsigned int NDimensions, unsigned int VSplineOrder>
typename RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>
::OutputPointType
RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>
::TransformPoint( const InputPointType & point ) const
{
  /** Allocate memory on the stack: */
  const unsigned int numberOfWeights = RecursiveBSplineWeightFunctionType::NumberOfWeights;
  const unsigned int numberOfIndices = RecursiveBSplineWeightFunctionType::NumberOfIndices;
  typename WeightsType::ValueType weightsArray[ numberOfWeights ];
  typename ParameterIndexArrayType::ValueType indicesArray[ numberOfIndices ];
  WeightsType             weights( weightsArray, numberOfWeights, false );
  ParameterIndexArrayType indices( indicesArray, numberOfIndices, false );

  OutputPointType         outputPoint;
  bool                    inside;

  this->TransformPoint( point, outputPoint, weights, indices, inside );

  return outputPoint;
} // end TransformPoint()


/**
 * ********************* TransformPoint ****************************
 */

template< typename TScalar, unsigned int NDimensions, unsigned int VSplineOrder >
void
RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>
::TransformPoint(
  const InputPointType & point,
  OutputPointType & outputPoint,
  WeightsType & weights,
  ParameterIndexArrayType & indices,
  bool & inside ) const
{
  inside = true;
  InputPointType transformedPoint = point;

  /** Check if the coefficient image has been set. */
  if( !this->m_CoefficientImages[ 0 ] )
  {
    itkWarningMacro( << "B-spline coefficients have not been set" );
    for( unsigned int j = 0; j < SpaceDimension; j++ )
    {
      outputPoint[ j ] = transformedPoint[ j ];
    }
    return;
  }

  /***/
  ContinuousIndexType cindex;
  this->TransformPointToContinuousGridIndex( point, cindex );

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and return the input point
  inside = this->InsideValidRegion( cindex );
  if( !inside )
  {
    outputPoint = transformedPoint;
    return;
  }

  // Compute interpolation weighs and store them in weights
  // MS: compare with AdvancedBSplineDeformableTransform
  IndexType supportIndex;
  this->m_RecursiveBSplineWeightFunction->Evaluate( cindex, weights, supportIndex );

  // Allocation of memory
  long evaluateIndexData[ ( SplineOrder + 1 ) * SpaceDimension ];
  long stepsData[ ( SplineOrder + 1 ) * SpaceDimension ];
  vnl_matrix_ref<long> evaluateIndex( SpaceDimension, SplineOrder + 1, evaluateIndexData );
  double * weightsPointer = &(weights[0]);
  long * steps = &(stepsData[0]);

  for( unsigned int ii = 0; ii < SpaceDimension; ++ii )
  {
    for( unsigned int jj = 0; jj <= SplineOrder; ++jj )
    {
      evaluateIndex[ ii ][ jj ] = supportIndex[ ii ] + jj;
    }
  }

  IndexType offsetTable;
  for( unsigned int n = 0; n < SpaceDimension; ++n )
  {
    offsetTable[ n ] = this->m_CoefficientImages[ 0 ]->GetOffsetTable()[ n ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      steps[ ( SplineOrder + 1 ) * n + k ] = evaluateIndex[ n ][ k ] * offsetTable[ n ];
    }
  }

  // Call recursive interpolate function
  outputPoint.Fill( NumericTraits<TScalar>::Zero );
  for( unsigned int j = 0; j < SpaceDimension; ++j )
  {
    const TScalar *basePointer = this->m_CoefficientImages[ j ]->GetBufferPointer();
    unsigned int c = 0;
    outputPoint[ j ] = RecursiveBSplineTransformImplementation< SpaceDimension, SplineOrder, TScalar >
      ::InterpolateTransformPoint( basePointer,
      steps,
      weightsPointer,
      basePointer,
      indices,
      c );

    // The output point is the start point + displacement.
    outputPoint[ j ] += transformedPoint[ j ];
  } // end for

} // end TransformPoint()


/**
 * ********************* TransformPointVector ****************************
 */

template< typename TScalar, unsigned int NDimensions, unsigned int VSplineOrder >
void
RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>
::TransformPointVector(
  const InputPointType & point,
  OutputPointType & outputPoint,
  WeightsType & weights1D,
  ParameterIndexArrayType & indices,
  bool & inside ) const
{
  inside = true;
  InputPointType transformedPoint = point;

  /** Check if the coefficient image has been set. */
  if( !this->m_CoefficientImages[ 0 ] )
  {
    itkWarningMacro( << "B-spline coefficients have not been set" );
    for( unsigned int j = 0; j < SpaceDimension; j++ )
    {
      outputPoint[ j ] = transformedPoint[ j ];
    }
    return;
  }

  /***/
  ContinuousIndexType cindex;
  this->TransformPointToContinuousGridIndex( point, cindex );

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and return the input point
  inside = this->InsideValidRegion( cindex );
  if( !inside )
  {
    outputPoint = transformedPoint;
    return;
  }

  // Compute interpolation weighs and store them in weights
  // MS: compare with AdvancedBSplineDeformableTransform
  IndexType supportIndex;
  this->m_RecursiveBSplineWeightFunction->Evaluate( cindex, weights1D, supportIndex );

  // Allocation of memory
  long evaluateIndexData[ ( SplineOrder + 1 ) * SpaceDimension ];
  long stepsData[ ( SplineOrder + 1 ) * SpaceDimension ];
  vnl_matrix_ref<long> evaluateIndex( SpaceDimension, SplineOrder + 1, evaluateIndexData );
  double * weightsPointer = &(weights1D[0]);
  long * steps = &(stepsData[0]);

  for( unsigned int ii = 0; ii < SpaceDimension; ++ii )
  {
    for( unsigned int jj = 0; jj <= SplineOrder; ++jj )
    {
      evaluateIndex[ ii ][ jj ] = supportIndex[ ii ] + jj;
    }
  }

  IndexType offsetTable;
  for( unsigned int n = 0; n < SpaceDimension; ++n )
  {
    offsetTable[ n ] = this->m_CoefficientImages[ 0 ]->GetOffsetTable()[ n ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      steps[ ( SplineOrder + 1 ) * n + k ] = evaluateIndex[ n ][ k ] * offsetTable[ n ];
    }
  }

  // Call recursive interpolate function, vector version
  outputPoint.Fill( NumericTraits<TScalar>::Zero );
  ScalarType opp[ SpaceDimension ];
  ScalarType * basePointers[ SpaceDimension ];
  for( unsigned int j = 0; j < SpaceDimension; ++j )
  {
    opp[ j ] = 0.0;
    basePointers[ j ] = this->m_CoefficientImages[ j ]->GetBufferPointer();
  }
  RecursiveBSplineTransformImplementation2< SpaceDimension, SpaceDimension, SplineOrder, TScalar >
    ::InterpolateTransformPoint( opp, basePointers, steps,
      weightsPointer, basePointers );

  // The output point is the start point + displacement.
  for( unsigned int j = 0; j < SpaceDimension; ++j )
  {
    outputPoint[ j ] = opp[ j ] + transformedPoint[ j ];
  }

} // end TransformPointVector()


/**
 * ********************* GetJacobian ****************************
 */

template< class TScalar, unsigned int NDimensions, unsigned int VSplineOrder >
void
RecursiveBSplineTransform< TScalar, NDimensions, VSplineOrder >
::GetJacobian( const InputPointType & ipp, JacobianType & jacobian,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
#if 0
  Superclass::GetJacobian( ipp, jacobian, nonZeroJacobianIndices );
#else
  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  ContinuousIndexType cindex;
  this->TransformPointToContinuousGridIndex( ipp, cindex );

  /** Initialize. */
  const NumberOfParametersType nnzji = this->GetNumberOfNonZeroJacobianIndices();
  if( ( jacobian.cols() != nnzji ) || ( jacobian.rows() != SpaceDimension ) )
  {
    jacobian.SetSize( SpaceDimension, nnzji );
    jacobian.Fill( 0.0 );
  }

  /** NOTE: if the support region does not lie totally within the grid
   * we assume zero displacement and zero Jacobian.
   */
  if( !this->InsideValidRegion( cindex ) )
  {
    nonZeroJacobianIndices.resize( this->GetNumberOfNonZeroJacobianIndices() );
    for( NumberOfParametersType i = 0; i < this->GetNumberOfNonZeroJacobianIndices(); ++i )
    {
      nonZeroJacobianIndices[ i ] = i;
    }
    return;
  }

  /** Compute the number of affected B-spline parameters.
   * Allocate memory on the stack.
   */
  //const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;
  //typename WeightsType::ValueType weightsArray[ numberOfWeights ];
  //WeightsType weights( weightsArray, numberOfWeights, false );

  /** Compute the interpolation weights. 
   * In contrast to the normal B-spline weights function, the recursive version
   * returns the individual weights instead of the multiplied ones.
   */
  IndexType supportIndex;
  //this->m_WeightsFunction->ComputeStartIndex( cindex, supportIndex );
  //this->m_WeightsFunction->Evaluate( cindex, supportIndex, weights );

  const unsigned int numberOfWeights = RecursiveBSplineWeightFunctionType::NumberOfWeights;
  const unsigned int numberOfIndices = RecursiveBSplineWeightFunctionType::NumberOfIndices;
  typename WeightsType::ValueType weightsArray1D[ numberOfWeights ];
  typename ParameterIndexArrayType::ValueType indicesArray[ numberOfIndices ];
  WeightsType             weights1D( weightsArray1D, numberOfWeights, false );
  this->m_RecursiveBSplineWeightFunction->Evaluate( cindex, weights1D, supportIndex );


  ParameterIndexArrayType indices( indicesArray, numberOfIndices, false );

  // Allocation of memory, just copied from the transform point above, need to check
  long evaluateIndexData[ ( SplineOrder + 1 ) * SpaceDimension ];
  long stepsData[ ( SplineOrder + 1 ) * SpaceDimension ];
  vnl_matrix_ref<long> evaluateIndex( SpaceDimension, SplineOrder + 1, evaluateIndexData );
  double * weights1DPointer = &(weights1D[0]);
  long * steps = &(stepsData[0]);

  for( unsigned int ii = 0; ii < SpaceDimension; ++ii )
  {
    for( unsigned int jj = 0; jj <= SplineOrder; ++jj )
    {
      //evaluateIndex[ ii ][ jj ] = supportIndex[ ii ] + jj;
      evaluateIndex[ ii ][ jj ] = 0 + jj;
    }
  }

  IndexType offsetTable;
  offsetTable[ 0 ] = 1;
  offsetTable[ 1 ] = 4;
  offsetTable[ 2 ] = 8;
  for( unsigned int n = 0; n < SpaceDimension; ++n )
  {
    //offsetTable[ n ] = this->m_CoefficientImages[ 0 ]->GetOffsetTable()[ n ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      steps[ ( SplineOrder + 1 ) * n + k ] = evaluateIndex[ n ][ k ] * offsetTable[ n ];
    }
  }

  // Call the recursive function, this is where the magic happens
  //outputPoint.Fill( NumericTraits<TScalar>::Zero );
  unsigned int ii = 0;
  //typename WeightsType::ValueType jacobians[ numberOfIndices ]; // is correct type, double
  ScalarType jacobians[ numberOfIndices ]; // is float?
  for( unsigned int i = 0; i < numberOfIndices; ++i ){ jacobians[i] = 1.0; }

  unsigned int nn = 0;
  for( unsigned int n = 0; n < SpaceDimension; ++n )
  {
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      std::cout << weights1DPointer[nn] << " ";
      nn++;
    }
    std::cout << std::endl;
  }
  std::cout << "\n" << std::endl;
  

  // initialization is done also in the recursion below
  //const TScalar *basePointer = this->m_CoefficientImages[ j ]->GetBufferPointer(); // not needed
  //unsigned int c = 0;
  //weightsArray[ ii ] = 
  RecursiveBSplineTransformImplementation< SpaceDimension, SplineOrder, TScalar >
    ::InterpolateGetJacobian( jacobians, steps, weights1DPointer );
    //::InterpolateGetJacobian( jacobians, weights1DPointer );
  std::cout << "\n" << std::endl;

  /** Put at the right positions. */
  ParametersValueType * jacobianPointer = jacobian.data_block();
  for( unsigned int d = 0; d < SpaceDimension; ++d )
  {
    unsigned long offset = d * SpaceDimension * numberOfWeights + d * numberOfWeights;
    std::copy( jacobians, jacobians + numberOfWeights, jacobianPointer + offset );
  }

  /** Setup support region *
  RegionType supportRegion;
  supportRegion.SetSize( this->m_SupportSize );
  supportRegion.SetIndex( supportIndex );

  /** Put at the right positions. *
  ParametersValueType * jacobianPointer = jacobian.data_block();
  for( unsigned int d = 0; d < SpaceDimension; ++d )
  {
    unsigned long offset = d * SpaceDimension * numberOfWeights + d * numberOfWeights;
    std::copy( weightsArray, weightsArray + numberOfWeights, jacobianPointer + offset );
  }

  /** Compute the nonzero Jacobian indices.
   * Takes a significant portion of the computation time of this function.
   */
  //this->ComputeNonZeroJacobianIndices( nonZeroJacobianIndices, supportRegion );
#endif
} // end GetJacobian()


/**
 * ********************* GetSpatialJacobian ****************************
 */

template< class TScalar, unsigned int NDimensions, unsigned int VSplineOrder >
void
RecursiveBSplineTransform< TScalar, NDimensions, VSplineOrder >
::GetSpatialJacobian(
  const InputPointType & ipp,
  SpatialJacobianType & sj ) const
{
  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  ContinuousIndexType cindex;
  this->TransformPointToContinuousGridIndex( ipp, cindex );

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and identity spatial Jacobian
  if( !this->InsideValidRegion( cindex ) )
  {
    sj.SetIdentity();
    return;
  }

  /** Compute the number of affected B-spline parameters. */
  /** Allocate memory on the stack: */
  const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;
  typename WeightsType::ValueType weightsArray[ numberOfWeights ];
  WeightsType weights( weightsArray, numberOfWeights, false );

  typename WeightsType::ValueType derivativeWeightsArray[ numberOfWeights ];
  WeightsType derivativeWeights( derivativeWeightsArray, numberOfWeights, false );

  IndexType supportIndex;
  this->m_DerivativeWeightsFunctions[ 0 ]->ComputeStartIndex(
    cindex, supportIndex );
  RegionType supportRegion;
  supportRegion.SetSize( this->m_SupportSize );
  supportRegion.SetIndex( supportIndex );

  //    /** Compute the spatial Jacobian sj:
  //   *    dT_{dim} / dx_i = delta_{dim,i} + \sum coefs_{dim} * weights * PointToGridIndex.
  //   */
  //    typedef ImageScanlineConstIterator< ImageType > IteratorType;
  //    sj.Fill( 0.0 );
  //    for( unsigned int i = 0; i < SpaceDimension; ++i )
  //    {
  //        /** Compute the derivative weights. */
  //        this->m_DerivativeWeightsFunctions[ i ]->Evaluate( cindex, supportIndex, weights );

  //        /** Compute the spatial Jacobian sj:
  //     *    dT_{dim} / dx_i = \sum coefs_{dim} * weights.
  //     */
  //        for( unsigned int dim = 0; dim < SpaceDimension; ++dim )
  //        {
  //            /** Create an iterator over the correct part of the coefficient
  //       * image. Create an iterator over the weights vector.
  //       */
  //            IteratorType itCoef( this->m_CoefficientImages[ dim ], supportRegion );
  //            typename WeightsType::const_iterator itWeights = weights.begin();

  //            /** Compute the sum for this dimension. */
  //            double sum = 0.0;
  //            while( !itCoef.IsAtEnd() )
  //            {
  //                while( !itCoef.IsAtEndOfLine() )
  //                {
  //                    sum += itCoef.Value() * ( *itWeights );
  //                    ++itWeights;
  //                    ++itCoef;
  //                }
  //                itCoef.NextLine();
  //            }

  //            /** Update the spatial Jacobian sj. */
  //            sj( dim, i ) += sum;

  //        } // end for dim
  //    }   // end for i

  // Compute interpolation weighs and store them in weights
  this->m_RecursiveBSplineWeightFunction->Evaluate( cindex, weights, supportIndex );
  this->m_RecursiveBSplineWeightFunction->EvaluateDerivative( cindex, derivativeWeights, supportIndex );

  //Allocation of memory
  // MS: The following is a copy from TransformPoint and candidate for refactoring
  long evaluateIndexData[ ( SplineOrder + 1 ) * SpaceDimension ];
  long stepsData[ ( SplineOrder + 1 ) * SpaceDimension ];
  vnl_matrix_ref<long> evaluateIndex( SpaceDimension, SplineOrder + 1, evaluateIndexData );
  double * weightsPointer = &(weights[0]);
  double * derivativeWeightsPointer = &(derivativeWeights[0]);
  long * steps = &(stepsData[0]);

  for( unsigned int ii = 0; ii < SpaceDimension; ++ii )
  {
    for( unsigned int jj = 0; jj <= SplineOrder; ++jj )
    {
      evaluateIndex[ ii ][ jj ] = supportIndex[ ii ] + jj;
    }
  }

  IndexType offsetTable;
  for( unsigned int n = 0; n < SpaceDimension; ++n )
  {
    offsetTable[ n ] = this->m_CoefficientImages[ 0 ]->GetOffsetTable()[ n ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      steps[ ( SplineOrder + 1 ) * n + k ] = evaluateIndex[ n ][ k ] * offsetTable[ n ];
    }
  }

  // Call recursive interpolate function
  for( unsigned int j = 0; j < SpaceDimension; ++j )
  {
    TScalar derivativeValue[ SpaceDimension + 1 ];
    const TScalar *basePointer = this->m_CoefficientImages[ j ]->GetBufferPointer();
    RecursiveBSplineTransformImplementation< SpaceDimension, SplineOrder, TScalar >
      ::InterpolateSpatialJacobian( derivativeValue,
      basePointer,
      steps,
      weightsPointer,
      derivativeWeightsPointer );

    for( unsigned int dim = 0; dim < SpaceDimension; ++dim )
    {
      sj( dim, j ) = derivativeValue[ dim + 1 ]; //First element of derivativeValue is the value, not the derivative.
    }

    sj( j, j ) += 1.0;
  } // end for

  /** Take into account grid spacing and direction cosines. */
  sj = sj * this->m_PointToIndexMatrix2;

} // end GetSpatialJacobian()


}// namespace


#endif

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
} // end Constructor()


////TransformPoint 1 argument
// MS: this is slightly different from AdvancedBSplineDeformableTransform
// should we update that one and delete this one?
template <typename TScalar, unsigned int NDimensions, unsigned int VSplineOrder>
typename RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>
::OutputPointType
RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>
::TransformPoint(const InputPointType & point) const
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
}


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

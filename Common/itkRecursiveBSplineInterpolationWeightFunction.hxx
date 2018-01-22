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
#ifndef __itkRecursiveBSplineInterpolationWeightFunction_hxx
#define __itkRecursiveBSplineInterpolationWeightFunction_hxx

#include "itkRecursiveBSplineInterpolationWeightFunction.h"

#include "itkImage.h"
#include "itkMatrix.h"
#include "itkMath.h"
#include "itkImageRegionConstIteratorWithIndex.h"

namespace itk
{

/**
 * ********************* Constructor ****************************
 */

template< typename TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder >
RecursiveBSplineInterpolationWeightFunction< TCoordRep, VSpaceDimension, VSplineOrder >
::RecursiveBSplineInterpolationWeightFunction()
{
  // Initialize support region is a hypercube of length SplineOrder + 1
  this->m_SupportSize.Fill( SplineOrder + 1 );

  this->m_NumberOfWeights = 1;
  for( unsigned int i = 0; i < SpaceDimension; ++i )
  {
    this->m_NumberOfWeights *= this->m_SupportSize[ i ];
  }

  // Initialize the interpolation kernel
  this->m_Kernel                      = KernelType::New();
  this->m_DerivativeKernel            = DerivativeKernelType::New();
  this->m_SecondOrderDerivativeKernel = SecondOrderDerivativeKernelType::New();

} // end Constructor


/**
 * ********************* PrintSelf ****************************
 */

template< typename TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder >
void
RecursiveBSplineInterpolationWeightFunction< TCoordRep, VSpaceDimension, VSplineOrder >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "NumberOfWeights: " << m_NumberOfWeights << std::endl;
  os << indent << "SupportSize: " << m_SupportSize << std::endl;
} // end PrintSelf()


/**
 * ********************* Evaluate ****************************
 */

template< typename TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder >
typename RecursiveBSplineInterpolationWeightFunction< TCoordRep, VSpaceDimension, VSplineOrder >::WeightsType
RecursiveBSplineInterpolationWeightFunction< TCoordRep, VSpaceDimension, VSplineOrder >
::Evaluate( const ContinuousIndexType & index ) const
{
  WeightsType weights( this->m_NumberOfWeights );
  IndexType   startIndex;

  this->Evaluate( index, weights, startIndex );

  return weights;
} // end Evaluate()


/**
 * ********************* Evaluate ****************************
 */

template< typename TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder >
void
RecursiveBSplineInterpolationWeightFunction< TCoordRep, VSpaceDimension, VSplineOrder >
::Evaluate(
  const ContinuousIndexType & cindex,
  WeightsType & weights,
  IndexType & startIndex ) const
{
  typename WeightsType::ValueType * weightsPtr = &weights[ 0 ];
  for( unsigned int i = 0; i < SpaceDimension; ++i )
  {
    startIndex[ i ] = Math::Floor< IndexValueType >( cindex[i] + 0.5 - SplineOrder / 2.0 );
    double x = cindex[ i ] - static_cast< double >( startIndex[ i ] );
    this->m_Kernel->Evaluate( x, weightsPtr );
    weightsPtr += SplineOrder + 1;
  }

} // end Evaluate()


/**
 * ********************* EvaluateDerivative ****************************
 */

template< typename TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder >
void
RecursiveBSplineInterpolationWeightFunction< TCoordRep, VSpaceDimension, VSplineOrder >
::EvaluateDerivative(
  const ContinuousIndexType & cindex,
  WeightsType & derivativeWeights,
  const IndexType & startIndex ) const
{
  for( unsigned int i = 0; i < SpaceDimension; ++i )
  {
    double x = cindex[ i ] - static_cast< double >( startIndex[ i ] );
    this->m_DerivativeKernel->Evaluate( x, &derivativeWeights[ i * this->m_SupportSize[ i ] ] );
  }
} // end EvaluateDerivative()


/**
 * ********************* EvaluateSecondOrderDerivative ****************************
 */

template< typename TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder >
void
RecursiveBSplineInterpolationWeightFunction< TCoordRep, VSpaceDimension, VSplineOrder >
::EvaluateSecondOrderDerivative(
  const ContinuousIndexType & cindex,
  WeightsType & hessianWeights,
  const IndexType & startIndex ) const
{
  for( unsigned int i = 0; i < SpaceDimension; ++i )
  {
    double x = cindex[ i ] - static_cast< double >( startIndex[ i ] );
    this->m_SecondOrderDerivativeKernel->Evaluate( x, &hessianWeights[ i * this->m_SupportSize[ i ] ] );
  }
} // end EvaluateSecondOrderDerivative()


} // end namespace itk

#endif

/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkRecursiveBSplineInterpolateImageFunctionWrapper_hxx
#define __itkRecursiveBSplineInterpolateImageFunctionWrapper_hxx

#include "itkRecursiveBSplineInterpolateImageFunctionWrapper.h"

// MS: check which ones are needed
#include "itkRecursiveBSplineInterpolateImageFunction.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkVector.h"
#include "itkMatrix.h"


namespace itk
{

/**
 * ******************* Constructor ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType>
RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >
::RecursiveBSplineInterpolateImageFunctionWrapper()
{
  // MS: move to superclass
  //Set the spline order
  unsigned int SplineOrder = 3;
  this->SetSplineOrder( SplineOrder );
} // end Constructor()


/**
 * ******************* PrintSelf ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType >
void
RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  this->m_InterpolatorInstance->PrintSelf( os, indent );
} // end PrintSelf()


/**
 * ******************* SetSplineOrder ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType >
void
RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >
::SetSplineOrder( unsigned int splineOrder )
{
  if( splineOrder == this->m_SplineOrder )
  {
    return; // MS: how about first time usage? when so=3??
  }
  this->m_SplineOrder = splineOrder;

  switch( this->m_SplineOrder )
  {
  case 0:
    this->m_InterpolatorInstance = RecursiveBSplineInterpolateImageFunction<
      TImageType, TCoordRep, TCoefficientType, 0 >::New();
    break;
  case 1:
    this->m_InterpolatorInstance = RecursiveBSplineInterpolateImageFunction<
      TImageType, TCoordRep, TCoefficientType, 1 >::New();
    break;
  case 2:
    this->m_InterpolatorInstance = RecursiveBSplineInterpolateImageFunction<
      TImageType, TCoordRep, TCoefficientType, 2 >::New();
    break;
  case 3:
    this->m_InterpolatorInstance = RecursiveBSplineInterpolateImageFunction<
      TImageType, TCoordRep, TCoefficientType, 3 >::New();
    break;
  case 4:
    this->m_InterpolatorInstance = RecursiveBSplineInterpolateImageFunction<
      TImageType, TCoordRep, TCoefficientType, 4 >::New();
    break;
  case 5:
    this->m_InterpolatorInstance = RecursiveBSplineInterpolateImageFunction<
      TImageType, TCoordRep, TCoefficientType, 5 >::New();
  }

} // end SetSplineOrder()


/**
 * ******************* SetInputImage ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType >
void
RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >
::SetInputImage( const InputImageType * inputData )
{
  Superclass::SetInputImage( inputData );
  this->m_InterpolatorInstance->SetInputImage( inputData );
} // end SetInputImage()


/**
 * ******************* Evaluate ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType >
typename RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >::OutputType
RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >
::Evaluate( const PointType & point ) const
{
  return this->m_InterpolatorInstance->Evaluate( point );
} // end Evaluate()


/**
 * ******************* Evaluate ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType >
typename RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >::OutputType
RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >
::Evaluate( const PointType & point, ThreadIdType threadID ) const
{
  return this->m_InterpolatorInstance->Evaluate( point, threadID );
} // end Evaluate()


/**
 * ******************* EvaluateAtContinuousIndex ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType >
typename RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >::OutputType
RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >
::EvaluateAtContinuousIndex( const ContinuousIndexType & index ) const
{
  return this->m_InterpolatorInstance->EvaluateAtContinuousIndex( index );
} // end EvaluateAtContinuousIndex()


/**
 * ******************* EvaluateAtContinuousIndex ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType >
typename RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >::OutputType
RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >
::EvaluateAtContinuousIndex( const ContinuousIndexType & index, ThreadIdType threadID ) const
{
  return this->m_InterpolatorInstance->EvaluateAtContinuousIndex( index, threadID );
}// end EvaluateAtContinuousIndex()


/**
 * ******************* EvaluateDerivative ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType >
typename RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >::CovariantVectorType
RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >
::EvaluateDerivative( const PointType & point ) const
{
  return this->m_InterpolatorInstance->EvaluateDerivative( point );
} // end EvaluateDerivative()


/**
 * ******************* EvaluateDerivative ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType >
typename RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >::CovariantVectorType
RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >
::EvaluateDerivative( const PointType & point, ThreadIdType threadID ) const
{
  return this->m_InterpolatorInstance->EvaluateDerivative( point, threadID );
} // end EvaluateDerivative()


/**
 * ******************* EvaluateDerivativeAtContinuousIndex ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType >
typename RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >::CovariantVectorType
RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >
::EvaluateDerivativeAtContinuousIndex( const ContinuousIndexType & x ) const
{
  return this->m_InterpolatorInstance->EvaluateDerivativeAtContinuousIndex( x );
} // end EvaluateDerivativeAtContinuousIndex()


/**
 * ******************* EvaluateDerivativeAtContinuousIndex ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType >
typename RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >::CovariantVectorType
RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >
::EvaluateDerivativeAtContinuousIndex( const ContinuousIndexType & point, ThreadIdType threadID ) const
{
  return this->m_InterpolatorInstance->EvaluateDerivativeAtContinuousIndex( point, threadID );
} // end EvaluateDerivativeAtContinuousIndex()


/**
 * ******************* EvaluateValueAndDerivative ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType >
void
RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >
::EvaluateValueAndDerivative( const PointType & point, OutputType & value, CovariantVectorType & deriv ) const
{
  this->m_InterpolatorInstance->EvaluateValueAndDerivative( point, value, deriv );
} // end EvaluateValueAndDerivative()


/**
 * ******************* EvaluateValueAndDerivativeAtContinuousIndex ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType >
void
RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >
::EvaluateValueAndDerivativeAtContinuousIndex(
  const ContinuousIndexType & x, OutputType & value,
  CovariantVectorType & deriv ) const
{
  this->m_InterpolatorInstance->EvaluateValueAndDerivativeAtContinuousIndex( x, value, deriv );
} // end EvaluateValueAndDerivativeAtContinuousIndex()


/**
 * ******************* EvaluateValueAndDerivative ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType >
void
RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >
::EvaluateValueAndDerivative(
  const PointType & point, OutputType & value,
  CovariantVectorType & deriv, ThreadIdType threadID ) const
{
  this->m_InterpolatorInstance->EvaluateValueAndDerivative( point, value, deriv, threadID );
} // end EvaluateValueAndDerivative()


/**
 * ******************* EvaluateValueAndDerivativeAtContinuousIndex ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType >
void
RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >
::EvaluateValueAndDerivativeAtContinuousIndex(
  const ContinuousIndexType & x, OutputType & value,
  CovariantVectorType & deriv, ThreadIdType threadID ) const
{
  this->m_InterpolatorInstance->EvaluateValueAndDerivative( x, value, deriv, threadID );
} // end EvaluateValueAndDerivativeAtContinuousIndex()


/**
 * ******************* SetNumberOfThreads ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType >
void
RecursiveBSplineInterpolateImageFunctionWrapper< TImageType, TCoordRep, TCoefficientType >
::SetNumberOfThreads( ThreadIdType numThreads )
{
  this->m_InterpolatorInstance->SetNumberOfThreads( numThreads );
} // end SetNumberOfThreads()

} // end namespace itk

#endif //itkRecursiveBSplineInterpolateImageFunctionWrapper_hxx

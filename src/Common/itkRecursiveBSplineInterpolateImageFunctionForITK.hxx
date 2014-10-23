/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkRecursiveBSplineInterpolateImageFunctionForITK_hxx
#define __itkRecursiveBSplineInterpolateImageFunctionForITK_hxx

#include "itkRecursiveBSplineInterpolateImageFunctionForITK.h"


namespace itk
{

/**
 * ******************* Constructor ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
RecursiveBSplineInterpolateImageFunctionForITK< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::RecursiveBSplineInterpolateImageFunctionForITK()
{
} // end Constructor()


/**
 * ******************* Evaluate ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
typename RecursiveBSplineInterpolateImageFunctionForITK< TImageType, TCoordRep, TCoefficientType, SplineOrder >::OutputType
RecursiveBSplineInterpolateImageFunctionForITK< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::Evaluate( const PointType & point, ThreadIdType threadID ) const
{
  ContinuousIndexType cindex;
  this->GetInputImage()->TransformPhysicalPointToContinuousIndex( point,  cindex );
  return this->EvaluateAtContinuousIndex( cindex );
} // end Evaluate()


/**
 * ******************* EvaluateAtContinuousIndex ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
typename RecursiveBSplineInterpolateImageFunctionForITK< TImageType, TCoordRep, TCoefficientType, SplineOrder >::OutputType
RecursiveBSplineInterpolateImageFunctionForITK< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::EvaluateAtContinuousIndex( const ContinuousIndexType & cindex, ThreadIdType threadID ) const
{
  return this->EvaluateAtContinuousIndex( cindex );
} // end EvaluateAtContinuousIndex()


/**
 * ******************* EvaluateDerivative ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
typename RecursiveBSplineInterpolateImageFunctionForITK< TImageType, TCoordRep, TCoefficientType, SplineOrder >::CovariantVectorType
RecursiveBSplineInterpolateImageFunctionForITK< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::EvaluateDerivative( const PointType & point, ThreadIdType threadID ) const
{
  ContinuousIndexType cindex;
  this->GetInputImage()->TransformPhysicalPointToContinuousIndex( point, cindex );
  return this->EvaluateDerivativeAtContinuousIndex( cindex );
} // end EvaluateDerivative()


/**
 * ******************* EvaluateDerivativeAtContinuousIndex ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
typename RecursiveBSplineInterpolateImageFunctionForITK< TImageType, TCoordRep, TCoefficientType, SplineOrder >::CovariantVectorType
RecursiveBSplineInterpolateImageFunctionForITK< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::EvaluateDerivativeAtContinuousIndex( const ContinuousIndexType & cindex, ThreadIdType threadID ) const
{
  return this->EvaluateDerivativeAtContinuousIndex( cindex );
} // end EvaluateDerivativeAtContinuousIndex()


/**
 * ******************* EvaluateValueAndDerivative ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
void
RecursiveBSplineInterpolateImageFunctionForITK< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::EvaluateValueAndDerivative( const PointType & point,
  OutputType & value, CovariantVectorType & deriv, ThreadIdType threadID ) const
{
  ContinuousIndexType cindex;
  this->GetInputImage()->TransformPhysicalPointToContinuousIndex( point, cindex );
  this->EvaluateValueAndDerivativeAtContinuousIndex( cindex, value, deriv );
} // end EvaluateValueAndDerivative()


/**
 * ******************* EvaluateValueAndDerivativeAtContinuousIndex ***********************
 */

template< class TImageType, class TCoordRep, class TCoefficientType, unsigned int SplineOrder >
void
RecursiveBSplineInterpolateImageFunctionForITK< TImageType, TCoordRep, TCoefficientType, SplineOrder >
::EvaluateValueAndDerivativeAtContinuousIndex( const ContinuousIndexType & x,
  OutputType & value, CovariantVectorType & deriv, ThreadIdType threadID ) const
{
  this->EvaluateValueAndDerivativeAtContinuousIndex( x, value, deriv );
} // end EvaluateValueAndDerivativeAtContinuousIndex()


} // namespace itk

#endif

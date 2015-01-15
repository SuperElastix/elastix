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

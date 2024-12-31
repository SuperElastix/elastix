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
#ifndef itkRecursiveBSplineInterpolationWeightFunction_hxx
#define itkRecursiveBSplineInterpolationWeightFunction_hxx

#include "itkRecursiveBSplineInterpolationWeightFunction.h"

#include "itkImage.h"
#include "itkMatrix.h"
#include "itkMath.h"
#include "itkImageRegionConstIteratorWithIndex.h"

namespace itk
{

/**
 * ********************* Evaluate ****************************
 */

template <typename TCoordinate, unsigned int VSpaceDimension, unsigned int VSplineOrder>
auto
RecursiveBSplineInterpolationWeightFunction<TCoordinate, VSpaceDimension, VSplineOrder>::Evaluate(
  const ContinuousIndexType & index) const -> WeightsType
{
  IndexType startIndex;
  return this->Evaluate(index, startIndex);
} // end Evaluate()


/**
 * ********************* Evaluate ****************************
 */

template <typename TCoordinate, unsigned int VSpaceDimension, unsigned int VSplineOrder>
auto
RecursiveBSplineInterpolationWeightFunction<TCoordinate, VSpaceDimension, VSplineOrder>::Evaluate(
  const ContinuousIndexType & cindex,
  IndexType &                 startIndex) const -> WeightsType
{
  WeightsType weights;

  typename WeightsType::ValueType * weightsPtr = &weights[0];
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    startIndex[i] = Math::Floor<IndexValueType>(cindex[i] + 0.5 - SplineOrder / 2.0);
    double x = cindex[i] - static_cast<double>(startIndex[i]);
    KernelType::FastEvaluate(x, weightsPtr);
    weightsPtr += SplineOrder + 1;
  }

  return weights;

} // end Evaluate()


template <typename TCoordinate, unsigned int VSpaceDimension, unsigned int VSplineOrder>
void
RecursiveBSplineInterpolationWeightFunction<TCoordinate, VSpaceDimension, VSplineOrder>::Evaluate(
  const ContinuousIndexType & cindex,
  WeightsType &               weights,
  IndexType &                 startIndex) const
{
  weights = this->Evaluate(cindex, startIndex);

} // end Evaluate()


/**
 * ********************* EvaluateDerivative ****************************
 */

template <typename TCoordinate, unsigned int VSpaceDimension, unsigned int VSplineOrder>
auto
RecursiveBSplineInterpolationWeightFunction<TCoordinate, VSpaceDimension, VSplineOrder>::EvaluateDerivative(
  const ContinuousIndexType & cindex,
  const IndexType &           startIndex) const -> WeightsType
{
  WeightsType derivativeWeights;

  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    double x = cindex[i] - static_cast<double>(startIndex[i]);
    DerivativeKernelType::FastEvaluate(x, &derivativeWeights[i * (VSplineOrder + 1)]);
  }

  return derivativeWeights;

} // end EvaluateDerivative()


/**
 * ********************* EvaluateSecondOrderDerivative ****************************
 */

template <typename TCoordinate, unsigned int VSpaceDimension, unsigned int VSplineOrder>
auto
RecursiveBSplineInterpolationWeightFunction<TCoordinate, VSpaceDimension, VSplineOrder>::EvaluateSecondOrderDerivative(
  const ContinuousIndexType & cindex,
  const IndexType &           startIndex) const -> WeightsType
{
  WeightsType hessianWeights;

  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    double x = cindex[i] - static_cast<double>(startIndex[i]);
    SecondOrderDerivativeKernelType::FastEvaluate(x, &hessianWeights[i * (VSplineOrder + 1)]);
  }

  return hessianWeights;
} // end EvaluateSecondOrderDerivative()


} // end namespace itk

#endif

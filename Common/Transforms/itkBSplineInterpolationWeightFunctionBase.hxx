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
#ifndef itkBSplineInterpolationWeightFunctionBase_hxx
#define itkBSplineInterpolationWeightFunctionBase_hxx

#include "itkBSplineInterpolationWeightFunctionBase.h"

namespace itk
{

/**
 * ******************* PrintSelf *******************
 */

template <typename TCoordinate, unsigned int VSpaceDimension, unsigned int VSplineOrder>
void
BSplineInterpolationWeightFunctionBase<TCoordinate, VSpaceDimension, VSplineOrder>::PrintSelf(std::ostream & os,
                                                                                              Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "OffsetToIndexTable: " << this->m_OffsetToIndexTable << std::endl;

} // end PrintSelf()


/**
 * ******************* ComputeStartIndex *******************
 */

template <typename TCoordinate, unsigned int VSpaceDimension, unsigned int VSplineOrder>
auto
BSplineInterpolationWeightFunctionBase<TCoordinate, VSpaceDimension, VSplineOrder>::ComputeStartIndex(
  const ContinuousIndexType & cindex) -> IndexType
{
  IndexType startIndex;

  /** Find the starting index of the support region. */
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    startIndex[i] = static_cast<typename IndexType::IndexValueType>(
      std::floor(cindex[i] - static_cast<double>(VSplineOrder - 1.0) / 2.0));
  }

  return startIndex;

} // end ComputeStartIndex()


/**
 * ******************* Evaluate *******************
 */

template <typename TCoordinate, unsigned int VSpaceDimension, unsigned int VSplineOrder>
auto
BSplineInterpolationWeightFunctionBase<TCoordinate, VSpaceDimension, VSplineOrder>::Evaluate(
  const ContinuousIndexType & cindex) const -> WeightsType
{
  /** Construct arguments for the Evaluate function that really does the work. */
  const IndexType startIndex = Self::ComputeStartIndex(cindex);

  /** Call the Evaluate function that really does the work. */
  return this->Evaluate(cindex, startIndex);

} // end Evaluate()


/**
 * ******************* Evaluate *******************
 */

template <typename TCoordinate, unsigned int VSpaceDimension, unsigned int VSplineOrder>
auto
BSplineInterpolationWeightFunctionBase<TCoordinate, VSpaceDimension, VSplineOrder>::Evaluate(
  const ContinuousIndexType & cindex,
  const IndexType &           startIndex) const -> WeightsType
{
  static_assert(WeightsType::Dimension == NumberOfWeights);

  WeightsType weights;

  /** Compute the 1D weights. */
  OneDWeightsType weights1D;
  this->Compute1DWeights(cindex, startIndex, weights1D);

  /** Compute the vector of weights. */
  for (unsigned int k = 0; k < NumberOfWeights; ++k)
  {
    double     tmp1 = 1.0;
    const auto tmp2 = m_OffsetToIndexTable[k];
    for (unsigned int j = 0; j < SpaceDimension; ++j)
    {
      tmp1 *= weights1D[j][tmp2[j]];
    }
    weights[k] = tmp1;
  }

  return weights;

} // end Evaluate()


} // end namespace itk

#endif

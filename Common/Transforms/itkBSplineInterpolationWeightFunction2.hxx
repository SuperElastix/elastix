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
#ifndef itkBSplineInterpolationWeightFunction2_hxx
#define itkBSplineInterpolationWeightFunction2_hxx

#include "itkBSplineInterpolationWeightFunction2.h"

namespace itk
{

/**
 * ****************** Constructor *******************************
 */

template <class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
BSplineInterpolationWeightFunction2<TCoordRep, VSpaceDimension, VSplineOrder>::BSplineInterpolationWeightFunction2() =
  default;


/**
 * ******************* Compute1DWeights *******************
 */

template <class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
void
BSplineInterpolationWeightFunction2<TCoordRep, VSpaceDimension, VSplineOrder>::Compute1DWeights(
  const ContinuousIndexType & index,
  const IndexType &           startIndex,
  OneDWeightsType &           weights1D) const
{
  /** Compute the 1D weights. */
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    double x = index[i] - static_cast<double>(startIndex[i]);

    // Compute weights
    double weights[6]; // Sufficiently large: maximum implemented SplineOrder + 1
    KernelType::FastEvaluate(x, weights);

    // Copy
    for (unsigned int k = 0; k < this->m_SupportSize[i]; ++k)
    {
      weights1D[i][k] = weights[k];
    }
  }
} // end Compute1DWeights()


} // end namespace itk

#endif

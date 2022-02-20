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
#ifndef itkBSplineInterpolationDerivativeWeightFunction_hxx
#define itkBSplineInterpolationDerivativeWeightFunction_hxx

#include "itkBSplineInterpolationDerivativeWeightFunction.h"

namespace itk
{

/**
 * ****************** Constructor *******************************
 */

template <class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
BSplineInterpolationDerivativeWeightFunction<TCoordRep, VSpaceDimension, VSplineOrder>::
  BSplineInterpolationDerivativeWeightFunction()
{
  /** Initialize members. */
  this->m_DerivativeDirection = 0;

} // end Constructor


/**
 * ******************* SetDerivativeDirection *******************
 */

template <class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
void
BSplineInterpolationDerivativeWeightFunction<TCoordRep, VSpaceDimension, VSplineOrder>::SetDerivativeDirection(
  unsigned int dir)
{
  if (dir != this->m_DerivativeDirection)
  {
    if (dir < SpaceDimension)
    {
      this->m_DerivativeDirection = dir;

      this->Modified();
    }
  }

} // end SetDerivativeDirection()


/**
 * ******************* PrintSelf *******************
 */

template <class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
void
BSplineInterpolationDerivativeWeightFunction<TCoordRep, VSpaceDimension, VSplineOrder>::PrintSelf(std::ostream & os,
                                                                                                  Indent indent) const
{
  this->Superclass::PrintSelf(os, indent);

} // end PrintSelf()


/**
 * ******************* Compute1DWeights *******************
 */

template <class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
void
BSplineInterpolationDerivativeWeightFunction<TCoordRep, VSpaceDimension, VSplineOrder>::Compute1DWeights(
  const ContinuousIndexType & cindex,
  const IndexType &           startIndex,
  OneDWeightsType &           weights1D) const
{
  /** Compute the 1D weights. */
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    double x = cindex[i] - static_cast<double>(startIndex[i]);

    if (i != this->m_DerivativeDirection)
    {
      for (unsigned int k = 0; k < this->m_SupportSize[i]; ++k)
      {
        weights1D[i][k] = KernelType::FastEvaluate(x);
        x -= 1.0;
      }
    }
    else
    {
      for (unsigned int k = 0; k < this->m_SupportSize[i]; ++k)
      {
        weights1D[i][k] = DerivativeKernelType::FastEvaluate(x);
        x -= 1.0;
      }
    }
  }

} // end Compute1DWeights()


} // end namespace itk

#endif

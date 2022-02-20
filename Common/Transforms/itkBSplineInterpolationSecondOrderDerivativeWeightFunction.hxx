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
#ifndef itkBSplineInterpolationSecondOrderDerivativeWeightFunction_hxx
#define itkBSplineInterpolationSecondOrderDerivativeWeightFunction_hxx

#include "itkBSplineInterpolationSecondOrderDerivativeWeightFunction.h"
#include "itkImage.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include <vnl/vnl_vector.h>

namespace itk
{

/**
 * ****************** Constructor *******************************
 */

template <class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
BSplineInterpolationSecondOrderDerivativeWeightFunction<TCoordRep, VSpaceDimension, VSplineOrder>::
  BSplineInterpolationSecondOrderDerivativeWeightFunction()
{
  /** Initialize members. */
  this->m_DerivativeDirections.fill(0);
  this->m_EqualDerivativeDirections = true;

} // end Constructor


/**
 * ******************* SetDerivativeDirections *******************
 */

template <class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
void
BSplineInterpolationSecondOrderDerivativeWeightFunction<TCoordRep, VSpaceDimension, VSplineOrder>::
  SetDerivativeDirections(unsigned int dir0, unsigned int dir1)
{
  if (dir0 != this->m_DerivativeDirections[0] || dir1 != this->m_DerivativeDirections[1])
  {
    if (dir0 < SpaceDimension && dir1 < SpaceDimension)
    {
      this->m_DerivativeDirections[0] = dir0;
      this->m_DerivativeDirections[1] = dir1;
      this->m_EqualDerivativeDirections = false;
      if (dir0 == dir1)
      {
        this->m_EqualDerivativeDirections = true;
      }

      this->Modified();
    }
  }

} // end SetDerivativeDirections()


/**
 * ******************* PrintSelf *******************
 */

template <class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
void
BSplineInterpolationSecondOrderDerivativeWeightFunction<TCoordRep, VSpaceDimension, VSplineOrder>::PrintSelf(
  std::ostream & os,
  Indent         indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "DerivativeDirections: [" << this->m_DerivativeDirections[0] << ", "
     << this->m_DerivativeDirections[1] << "]" << std::endl;
  os << indent << "EqualDerivativeDirections: " << this->m_EqualDerivativeDirections << std::endl;

} // end PrintSelf()


/**
 * ******************* Compute1DWeights *******************
 */

template <class TCoordRep, unsigned int VSpaceDimension, unsigned int VSplineOrder>
void
BSplineInterpolationSecondOrderDerivativeWeightFunction<TCoordRep, VSpaceDimension, VSplineOrder>::Compute1DWeights(
  const ContinuousIndexType & index,
  const IndexType &           startIndex,
  OneDWeightsType &           weights1D) const
{
  /** Compute the 1D weights. */
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    double x = index[i] - static_cast<double>(startIndex[i]);

    if (i != this->m_DerivativeDirections[0] && i != this->m_DerivativeDirections[1])
    {
      for (unsigned int k = 0; k < this->m_SupportSize[i]; ++k)
      {
        weights1D[i][k] = KernelType::FastEvaluate(x);
        x -= 1.0;
      }
    }
    else
    {
      if (this->m_EqualDerivativeDirections)
      {
        for (unsigned int k = 0; k < this->m_SupportSize[i]; ++k)
        {
          weights1D[i][k] = SecondOrderDerivativeKernelType::FastEvaluate(x);
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
  }

} // end Compute1DWeights()


} // end namespace itk

#endif

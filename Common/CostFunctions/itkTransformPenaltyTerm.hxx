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
#ifndef itkTransformPenaltyTerm_hxx
#define itkTransformPenaltyTerm_hxx

#include "itkTransformPenaltyTerm.h"

namespace itk
{

/**
 * ****************** CheckForBSplineTransform *******************************
 */

template <class TFixedImage, class TScalarType>
bool
TransformPenaltyTerm<TFixedImage, TScalarType>::CheckForBSplineTransform2(BSplineOrder3TransformPointer & bspline) const
{
  /** The following checks for many spline orders. */
  this->CheckForBSplineTransform();

  /** Quit if the advanced transform is not a B-spline. */
  if (!this->m_TransformIsBSpline)
    return false;

  /** We will return the B-spline by reference, but only in case it is a third order B-spline. */
  BSplineOrder3TransformType * testPtr1 =
    dynamic_cast<BSplineOrder3TransformType *>(this->m_AdvancedTransform.GetPointer());
  CombinationTransformType * testPtr2a =
    dynamic_cast<CombinationTransformType *>(this->m_AdvancedTransform.GetPointer());

  if (testPtr1)
  {
    /** The transform is of type AdvancedBSplineDeformableTransform. */
    bspline = testPtr1;
  }
  else if (testPtr2a)
  {
    /** The transform is of type AdvancedCombinationTransform. */
    BSplineOrder3TransformType * testPtr2b =
      dynamic_cast<BSplineOrder3TransformType *>((testPtr2a->GetModifiableCurrentTransform()));
    if (testPtr2b)
    {
      /** The current transform is of type AdvancedBSplineDeformableTransform. */
      bspline = testPtr2b;
    }
  }

  return this->m_TransformIsBSpline;

} // end CheckForBSplineTransform()


} // end namespace itk

#endif // #ifndef itkTransformPenaltyTerm_hxx

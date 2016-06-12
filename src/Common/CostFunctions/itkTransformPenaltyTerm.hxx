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
#ifndef __itkTransformPenaltyTerm_hxx
#define __itkTransformPenaltyTerm_hxx

#include "itkTransformPenaltyTerm.h"

namespace itk
{

/**
 * ****************** CheckForBSplineTransform *******************************
 */

template< class TFixedImage, class TScalarType >
bool
TransformPenaltyTerm< TFixedImage, TScalarType >
::CheckForBSplineTransform( BSplineTransformPointer & bspline ) const
{
  /** Check if this transform is a B-spline transform. */
  BSplineTransformType * testPtr1
    = dynamic_cast< BSplineTransformType * >( this->m_AdvancedTransform.GetPointer() );
  CombinationTransformType * testPtr2a
    = dynamic_cast< CombinationTransformType * >( this->m_AdvancedTransform.GetPointer() );
  bool transformIsBSpline = false;
  if( testPtr1 )
  {
    /** The transform is of type AdvancedBSplineDeformableTransform. */
    transformIsBSpline = true;
    bspline            = testPtr1;
  }
  else if( testPtr2a )
  {
    /** The transform is of type AdvancedCombinationTransform. */
    BSplineTransformType * testPtr2b = dynamic_cast< BSplineTransformType * >(
      ( testPtr2a->GetCurrentTransform() ) );
    if( testPtr2b )
    {
      /** The current transform is of type AdvancedBSplineDeformableTransform. */
      transformIsBSpline = true;
      bspline            = testPtr2b;
    }
  }

  return transformIsBSpline;

} // end CheckForBSplineTransform()


} // end namespace itk

#endif // #ifndef __itkTransformPenaltyTerm_hxx

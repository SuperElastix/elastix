/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkTransformPenaltyTerm_txx
#define __itkTransformPenaltyTerm_txx

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
    = dynamic_cast<BSplineTransformType *>( this->m_AdvancedTransform.GetPointer() );
  CombinationTransformType * testPtr2a
    = dynamic_cast<CombinationTransformType *>( this->m_AdvancedTransform.GetPointer() );
  bool transformIsBSpline = false;
  if ( testPtr1 )
  {
    /** The transform is of type AdvancedBSplineDeformableTransform. */
    transformIsBSpline = true;
    bspline = testPtr1;
  }
  else if ( testPtr2a )
  {
    /** The transform is of type AdvancedCombinationTransform. */
    BSplineTransformType * testPtr2b = dynamic_cast<BSplineTransformType *>(
      (testPtr2a->GetCurrentTransform()) );
    if ( testPtr2b )
    {
      /** The current transform is of type AdvancedBSplineDeformableTransform. */
      transformIsBSpline = true;
      bspline = testPtr2b;
    }
  }

  return transformIsBSpline;

} // end CheckForBSplineTransform()


} // end namespace itk

#endif // #ifndef __itkTransformPenaltyTerm_txx


/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxBSplineInterpolatorFloat_hxx
#define __elxBSplineInterpolatorFloat_hxx

#include "elxBSplineInterpolatorFloat.h"

namespace elastix
{
using namespace itk;


  /**
   * ***************** BeforeEachResolution ***********************
   */

  template <class TElastix>
    void BSplineInterpolatorFloat<TElastix>::
    BeforeEachResolution( void )
  {
    /** Get the current resolution level. */
    unsigned int level = 
      ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

    /** Set the SplineOrder, default value = 1. */
    unsigned int splineOrder = 1;
    
    /** Read the desired splineOrder from the parameterFile. */
    this->GetConfiguration()->ReadParameter( splineOrder,
      "BSplineInterpolationOrder", this->GetComponentLabel(), level, 0 );
    
    /** Set the splineOrder. */
    this->SetSplineOrder( splineOrder );
     
  } // end BeforeEachResolution


} // end namespace elastix

#endif // end #ifndef __elxBSplineInterpolatorFloat_hxx


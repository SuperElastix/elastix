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

/**
 * ***************** BeforeEachResolution ***********************
 */

template< class TElastix >
void
BSplineInterpolatorFloat< TElastix >
::BeforeEachResolution( void )
{
  /** Get the current resolution level. */
  unsigned int level
    = ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

  /** Read the desired spline order from the parameter file. */
  unsigned int splineOrder = 1;
  this->GetConfiguration()->ReadParameter( splineOrder,
    "BSplineInterpolationOrder", this->GetComponentLabel(), level, 0 );

  /** Check. */
  if( splineOrder == 0 )
  {
    elx::xout[ "warning" ] << "WARNING: the BSplineInterpolationOrder is set to 0.\n"
                           << "         It is not possible to take derivatives with this setting.\n"
                           << "         Make sure you use a derivative free optimizer."
                           << std::endl;
  }

  /** Set the splineOrder. */
  this->SetSplineOrder( splineOrder );

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef __elxBSplineInterpolatorFloat_hxx

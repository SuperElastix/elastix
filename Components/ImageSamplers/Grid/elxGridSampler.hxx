/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxGridSampler_hxx
#define __elxGridSampler_hxx

#include "elxGridSampler.h"

namespace elastix
{

/**
 * ******************* BeforeEachResolution ******************
 */

template< class TElastix >
void
GridSampler< TElastix >
::BeforeEachResolution( void )
{
  const unsigned int level
    = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  GridSpacingType gridspacing;

  /** Read the desired grid spacing of the samples. */
  unsigned int spacing_dim;
  for( unsigned int dim = 0; dim < InputImageDimension; dim++ )
  {
    spacing_dim = 2;
    this->GetConfiguration()->ReadParameter(
      spacing_dim, "SampleGridSpacing",
      this->GetComponentLabel(), level * InputImageDimension + dim, -1 );
    gridspacing[ dim ] = static_cast< SampleGridSpacingValueType >( spacing_dim );
  }
  this->SetSampleGridSpacing( gridspacing );

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef __elxGridSampler_hxx

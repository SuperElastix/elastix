/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxImageSamplerBase_hxx
#define __elxImageSamplerBase_hxx

#include "elxImageSamplerBase.h"

namespace elastix
{

/**
 * ******************* BeforeEachResolutionBase ******************
 */

template< class TElastix >
void
ImageSamplerBase< TElastix >
::BeforeEachResolutionBase( void )
{
  /** Get the current resolution level. */
  unsigned int level
    = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** Check if NewSamplesEveryIteration is possible with the selected ImageSampler.
   * The "" argument means that no prefix is supplied.
   */
  bool newSamples = false;
  this->m_Configuration->ReadParameter( newSamples,
    "NewSamplesEveryIteration", "", level, 0, true );

  if( newSamples )
  {
    bool ret = this->GetAsITKBaseType()->SelectingNewSamplesOnUpdateSupported();
    if( !ret )
    {
      xl::xout[ "warning" ]
        << "WARNING: You want to select new samples every iteration,\n"
        << "but the selected ImageSampler is not suited for that."
        << std::endl;
    }
  }

  /** Temporary?: Use the multi-threaded version or not. */
  std::string useMultiThread = this->m_Configuration->GetCommandLineArgument( "-mts" ); // mts: multi-threaded samplers
  if( useMultiThread == "true" )
  {
    this->GetAsITKBaseType()->SetUseMultiThread( true );
  }
  else { this->GetAsITKBaseType()->SetUseMultiThread( false ); }

} // end BeforeEachResolutionBase()


} // end namespace elastix

#endif //#ifndef __elxImageSamplerBase_hxx

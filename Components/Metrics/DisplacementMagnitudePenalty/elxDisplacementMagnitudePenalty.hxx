/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxDisplacementMagnitudePenalty_HXX__
#define __elxDisplacementMagnitudePenalty_HXX__

#include "elxDisplacementMagnitudePenalty.h"

namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template< class TElastix >
void
DisplacementMagnitudePenalty< TElastix >
::Initialize( void ) throw ( itk::ExceptionObject )
{
  TimerPointer timer = TimerType::New();
  timer->StartTimer();
  this->Superclass1::Initialize();
  timer->StopTimer();
  elxout << "Initialization of DisplacementMagnitude penalty term took: "
         << static_cast< long >( timer->GetElapsedClockSec() * 1000 )
         << " ms." << std::endl;

} // end Initialize()


} // end namespace elastix

#endif // end #ifndef __elxDisplacementMagnitudePenalty_HXX__

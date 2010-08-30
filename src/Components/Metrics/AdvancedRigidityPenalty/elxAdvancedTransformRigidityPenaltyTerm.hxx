/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxAdvancedTransformRigidityPenaltyTerm_HXX__
#define __elxAdvancedTransformRigidityPenaltyTerm_HXX__

#include "elxAdvancedTransformRigidityPenaltyTerm.h"


namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template <class TElastix>
void AdvancedTransformRigidityPenalty<TElastix>
::Initialize( void ) throw (ExceptionObject)
{
  TimerPointer timer = TimerType::New();
  timer->StartTimer();
  this->Superclass1::Initialize();
  timer->StopTimer();
  elxout << "Initialization of AdvancedTransformRigidityPenalty term took: "
    << static_cast<long>( timer->GetElapsedClockSec() * 1000 )
    << " ms." << std::endl;

} // end Initialize()


/**
 * ***************** BeforeEachResolution ***********************
 */

//template <class TElastix>
//void AdvancedTransformRigidityPenalty<TElastix>
//::BeforeEachResolution( void )
//{
//  /** Get the current resolution level. */
//  unsigned int level =
//    ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();
//
//
//} // end BeforeEachResolution()


} // end namespace elastix


#endif // end #ifndef __elxAdvancedTransformRigidityPenaltyTerm_HXX__


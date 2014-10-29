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
#ifndef __xoutsimple_hxx
#define __xoutsimple_hxx

#include "xoutsimple.h"

namespace xoutlibrary
{
using namespace std;

/**
 * ********************* Constructor ****************************
 */

template< class charT, class traits >
xoutsimple< charT, traits >::xoutsimple()
{
  //nothing

}   // end Constructor


/**
 * ********************* Destructor *****************************
 */

template< class charT, class traits >
xoutsimple< charT, traits >::~xoutsimple()
{
  //nothing

}   // end Destructor


/**
 * **************** AddOutput (ostream_type) ********************
 */

template< class charT, class traits >
int
xoutsimple< charT, traits >::AddOutput( const char * name, ostream_type * output )
{
  return this->AddTargetCell( name, output );

}   // end AddOutput


/**
 * **************** AddOutput (xoutsimple) **********************
 */

template< class charT, class traits >
int
xoutsimple< charT, traits >::AddOutput( const char * name, Superclass * output )
{
  return this->AddTargetCell( name, output );

}   // end AddOutput


/**
 * ***************** RemoveOutput *******************************
 */

template< class charT, class traits >
int
xoutsimple< charT, traits >::RemoveOutput( const char * name )
{
  return this->RemoveTargetCell( name );

}   // end RemoveOutput


/**
 * **************** SetOutputs (ostream_types) ******************
 */

template< class charT, class traits >
void
xoutsimple< charT, traits >::SetOutputs( const CStreamMapType & outputmap )
{
  this->SetTargetCells( outputmap );

}   // end SetOutputs


/**
 * **************** SetOutputs (xoutobjects) ********************
 */

template< class charT, class traits >
void
xoutsimple< charT, traits >::SetOutputs( const XStreamMapType & outputmap )
{
  this->SetTargetCells( outputmap );

}   // end SetOutputs()


/**
 * **************** GetOutputs (map of xoutobjects) *************
 */

template< class charT, class traits >
const typename xoutsimple< charT, traits >::XStreamMapType
& xoutsimple< charT, traits >::GetXOutputs( void )
{
  return this->m_XTargetCells;

}   // end GetXOutputs()

/**
 * **************** GetOutputs (map of c-streams) ***************
 */

template< class charT, class traits >
const typename xoutsimple< charT, traits >::CStreamMapType
& xoutsimple< charT, traits >::GetCOutputs( void )
{
  return this->m_CTargetCells;

}   // end GetCOutputs()

} // end namespace xoutlibrary

#endif // end #ifndef __xoutsimple_hxx

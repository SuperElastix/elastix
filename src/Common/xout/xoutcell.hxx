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
#ifndef __xoutcell_hxx
#define __xoutcell_hxx

#include "xoutcell.h"

namespace xoutlibrary
{
using namespace std;

/**
 * ************************ Constructor *************************
 */

template< class charT, class traits >
xoutcell< charT, traits >::xoutcell()
{
  this->AddTargetCell( "InternalBuffer", &( this->m_InternalBuffer ) );

}   // end Constructor


/**
 * ********************* Destructor *****************************
 */

template< class charT, class traits >
xoutcell< charT, traits >::~xoutcell()
{
  //nothing

}   // end Destructor


/**
 * ******************** WriteBufferedData ***********************
 *
 * The buffered data is sent to the outputs.
 */

template< class charT, class traits >
void
xoutcell< charT, traits >::WriteBufferedData( void )
{
  /** Make sure all data is written to the string */
  this->m_InternalBuffer << flush;

  const std::string & strbuf = this->m_InternalBuffer.str();

  const char * charbuf = strbuf.c_str();

  /** Send the string to the outputs */
  for( CStreamMapIteratorType cit = this->m_COutputs.begin();
    cit != this->m_COutputs.end(); ++cit )
  {
    *( cit->second ) << charbuf << flush;
  }

  /** Send the string to the outputs */
  for( XStreamMapIteratorType xit = this->m_XOutputs.begin();
    xit != this->m_XOutputs.end(); ++xit )
  {
    *( xit->second ) << charbuf;
    xit->second->WriteBufferedData();
  }

  /** Empty the internal buffer */
  this->m_InternalBuffer.str( string( "" ) );

}   // end WriteBufferedData


} // end namespace xoutlibrary

#endif // end #ifndef __xoutcell_hxx

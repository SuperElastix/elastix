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

} // end Constructor


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

  const std::string strbuf = this->m_InternalBuffer.str();

  const char * const charbuf = strbuf.c_str();

  /** Send the string to the outputs */
  for( const auto& output : this->m_COutputs )
  {
    *( output.second ) << charbuf << flush;
  }

  /** Send the string to the outputs */
  for( const auto& output : this->m_XOutputs )
  {
    *( output.second ) << charbuf;
    output.second->WriteBufferedData();
  }

  /** Empty the internal buffer */
  this->m_InternalBuffer.str( string( "" ) );

} // end WriteBufferedData


} // end namespace xoutlibrary

#endif // end #ifndef __xoutcell_hxx

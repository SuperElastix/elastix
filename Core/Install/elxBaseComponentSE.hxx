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

#ifndef __elxBaseComponentSE_hxx
#define __elxBaseComponentSE_hxx

#include "elxBaseComponentSE.h"

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template< class TElastix >
BaseComponentSE< TElastix >::BaseComponentSE()
{
  /** Initialize.*/
  this->m_Elastix       = 0;
  this->m_Configuration = 0;
  this->m_Registration  = 0;
}


/**
 * *********************** SetElastix ***************************
 */

template< class TElastix >
void
BaseComponentSE< TElastix >::SetElastix( TElastix * _arg )
{
  /** If this->m_Elastix is not set, then set it. */
  if( this->m_Elastix != _arg )
  {
    this->m_Elastix = _arg;

    if( this->m_Elastix.IsNotNull() )
    {
      this->m_Configuration = this->m_Elastix->GetConfiguration();
      this->m_Registration  = dynamic_cast< RegistrationPointer >(
        this->m_Elastix->GetElxRegistrationBase() );
    }

    itk::Object * thisasobject = dynamic_cast< itk::Object * >( this );
    if( thisasobject )
    {
      thisasobject->Modified();
    }
  }

}   // end SetElastix


/**
 * *********************** SetConfiguration ***************************
 *
 * Added for transformix.
 */

template< class TElastix >
void
BaseComponentSE< TElastix >::SetConfiguration( ConfigurationType * _arg )
{
  /** If this->m_Configuration is not set, then set it.*/
  if( this->m_Configuration != _arg )
  {
    this->m_Configuration = _arg;

    itk::Object * thisasobject = dynamic_cast< itk::Object * >( this );
    if( thisasobject )
    {
      thisasobject->Modified();
    }
  }

}   // end SetConfiguration


} // end namespace elastix

#endif // end #ifndef __elxBaseComponentSE_hxx

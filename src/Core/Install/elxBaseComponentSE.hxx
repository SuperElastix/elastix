/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

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

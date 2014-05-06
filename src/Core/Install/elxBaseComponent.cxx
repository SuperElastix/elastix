/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#include "elxBaseComponent.h"


namespace elastix
{

/**
 * ****************** elxGetClassName ****************************
 */

const char * BaseComponent::elxGetClassName( void ) const
{
  return "BaseComponent";
} // end elxGetClassName()


/**
 * ****************** SetComponentLabel ****************************
 */

void BaseComponent::SetComponentLabel( const char * label, unsigned int idx )
{
  std::ostringstream makestring;
  makestring << label << idx;
  this->m_ComponentLabel = makestring.str();
} // end SetComponentLabel()


/**
 * ****************** GetComponentLabel ****************************
 */

const char * BaseComponent::GetComponentLabel( void ) const
{
  return this->m_ComponentLabel.c_str();
} // end GetComponentLabel()


/**
 * ****************** ConvertSecondsToDHMS ****************************
 */

std::string BaseComponent::ConvertSecondsToDHMS( const double totalSeconds ) const
{
  /** Define days, hours, minutes. */
  const std::size_t secondsPerMinute = 60;
  const std::size_t secondsPerHour   = 60 * secondsPerMinute;
  const std::size_t secondsPerDay    = 24 * secondsPerHour;

  /** Convert total seconds. */
  std::size_t iSeconds = static_cast<std::size_t>( totalSeconds );
  const std::size_t days = iSeconds / secondsPerDay;

  iSeconds %= secondsPerDay;
  const std::size_t hours = iSeconds / secondsPerHour;

  iSeconds %= secondsPerHour;
  const std::size_t minutes = iSeconds / secondsPerMinute;

  iSeconds %= secondsPerMinute;
  const std::size_t seconds = iSeconds;

  /** Create a string in days, hours, minutes and seconds. */
  std::ostringstream make_string( "" );
  if( days    != 0 ){ make_string << days    << " days, "; }
  if( hours   != 0 ){ make_string << hours   << " h, "; }
  if( minutes != 0 ){ make_string << minutes << " min, "; }
  if( seconds != 0 ){ make_string << seconds << " s"; }

  /** Return a value. */
  return make_string.str();

} // end ConvertSecondsToDHMS()

} //end namespace elastix

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

#include "elxBaseComponent.h"

#include <cmath>

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

std::string BaseComponent::ConvertSecondsToDHMS(
  const double totalSeconds, const unsigned int precision = 0 ) const
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

  //iSeconds %= secondsPerMinute;
  //const std::size_t seconds = iSeconds;
  const double dSeconds = fmod( totalSeconds, 60.0 );

  /** Create a string in days, hours, minutes and seconds. */
  bool nonzero = false;
  std::ostringstream make_string( "" );
  if( days    != 0            ){ make_string << days    << "d"; nonzero = true; }
  if( hours   != 0 || nonzero ){ make_string << hours   << "h"; nonzero = true; }
  if( minutes != 0 || nonzero ){ make_string << minutes << "m"; nonzero = true; }
  make_string << std::showpoint << std::fixed << std::setprecision( precision );
  make_string << dSeconds << "s";

  /** Return a value. */
  return make_string.str();

} // end ConvertSecondsToDHMS()


} //end namespace elastix

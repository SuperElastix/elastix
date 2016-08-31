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
#ifndef __elastix_h
#define __elastix_h

#include "itkUseMevisDicomTiff.h"

#include <iostream>
#include <iomanip>      // std::setprecision
#include <string>
#include <vector>
#include <queue>
#include "itkObject.h"
#include "itkDataObject.h"
#include <itksys/SystemTools.hxx>
#include <itksys/SystemInformation.hxx>
#include "itkTimeProbe.h"
#include <time.h>

/** Declare PrintHelp function.
 *
 * \commandlinearg --help: optional argument for elastix and transformix to call the help. \n
 *    example: <tt>elastix --help</tt> \n
 *    example: <tt>transformix --help</tt> \n
 * \commandlinearg --version: optional argument for elastix and transformix to call
 *    version information. \n
 *    example: <tt>elastix --version</tt> \n
 *    example: <tt>transformix --version</tt> \n
 */
void PrintHelp( void );

/** ConvertSecondsToDHMS
 *
 */
std::string
ConvertSecondsToDHMS( const double totalSeconds, const unsigned int precision = 0 )
{
  /** Define days, hours, minutes. */
  const std::size_t secondsPerMinute = 60;
  const std::size_t secondsPerHour   = 60 * secondsPerMinute;
  const std::size_t secondsPerDay    = 24 * secondsPerHour;

  /** Convert total seconds. */
  std::size_t       iSeconds = static_cast< std::size_t >( totalSeconds );
  const std::size_t days     = iSeconds / secondsPerDay;

  iSeconds %= secondsPerDay;
  const std::size_t hours = iSeconds / secondsPerHour;

  iSeconds %= secondsPerHour;
  const std::size_t minutes = iSeconds / secondsPerMinute;

  //iSeconds %= secondsPerMinute;
  //const std::size_t seconds = iSeconds;
  const double dSeconds = fmod( totalSeconds, 60.0 );

  /** Create a string in days, hours, minutes and seconds. */
  bool               nonzero = false;
  std::ostringstream make_string( "" );
  if( days    != 0            ) { make_string << days    << "d"; nonzero = true; }
  if( hours   != 0 || nonzero ) { make_string << hours   << "h"; nonzero = true; }
  if( minutes != 0 || nonzero ) { make_string << minutes << "m"; nonzero = true; }
  make_string << std::showpoint << std::fixed << std::setprecision( precision );
  make_string << dSeconds << "s";

  /** Return a value. */
  return make_string.str();

} // end ConvertSecondsToDHMS()


/** Returns current date and time as a string. */
std::string
GetCurrentDateAndTime( void )
{
  // Obtain current time
  time_t rawtime = time( NULL );
  // Convert to local time
  struct tm * timeinfo = localtime( &rawtime );
  // Convert to human-readable format
  std::string timeAsString = std::string( asctime( timeinfo ) );
  // Erase newline character at end
  timeAsString.erase( timeAsString.end() - 1 );
  //timeAsString.pop_back() // c++11 feature

  return timeAsString;
} // end GetCurrentDateAndTime()


#endif

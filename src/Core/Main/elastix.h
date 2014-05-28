/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elastix_h
#define __elastix_h

#include "itkUseMevisDicomTiff.h"

#include <iostream>
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
std::string ConvertSecondsToDHMS( const double totalSeconds )
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
  bool nonzero = false;
  std::ostringstream make_string( "" );
  if( days    != 0            ){ make_string << days    << "d"; nonzero = true; }
  if( hours   != 0 || nonzero ){ make_string << hours   << "h"; nonzero = true; }
  if( minutes != 0 || nonzero ){ make_string << minutes << "m"; nonzero = true; }
  make_string << seconds << "s";

  /** Return a value. */
  return make_string.str();

} // end ConvertSecondsToDHMS()


/** Returns current date and time as a string. */
std::string GetCurrentDateAndTime( void )
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

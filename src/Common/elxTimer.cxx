/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxTimer_CXX_
#define __elxTimer_CXX_

#include "elxTimer.h"

namespace tmr
{

/**
 * ********************* Constructor ****************************
 */

Timer::Timer()
{
  /** Initialize.*/
  this->m_ElapsedTimeDHMS.resize( 4 );
  this->m_StartTime  = 0;
  this->m_StartClock = 0;
  this->m_StopTime   = 0;
  this->m_StopClock  = 0;

} // end Constructor


/**
 * ********************** StartTimer ****************************
 */

void
Timer::StartTimer( void )
{
  /** Get the current time.*/
  this->m_StartTime  = time( NULL );
  this->m_StartClock = clock();

#ifdef ELX_USE_CLOCK_GETTIME
  clock_gettime( CLOCK_MONOTONIC, &this->m_StartClockMonotonic );
#endif

} // end StartTimer()


/**
 * *********************** StopTimer ****************************
 */

int
Timer::StopTimer( void )
{
  /** Check if m_StartTime != 0. */
  if( this->m_StartTime == 0 ) { return 1; }

  /** Get the current time. */
  this->m_StopTime  = time( NULL );
  this->m_StopClock = clock();

#ifdef ELX_USE_CLOCK_GETTIME
  clock_gettime( CLOCK_MONOTONIC, &this->m_StopClockMonotonic );
#endif

  /** Get the elapsed time. */
  this->ElapsedClockAndTime();

  return 0;

} // end StoptTimer()


/**
 * *********************** ElapsedClockAndTime **************************
 */

int
Timer::ElapsedClockAndTime( void )
{
  /** Check if if m_StopTime != 0. */
  if( this->m_StopTime == 0 ) { return 1; }

  /** Calculate time difference = m_Elapsedtime. */
  this->m_ElapsedTime  = difftime( this->m_StopTime, this->m_StartTime );
  this->m_ElapsedClock = this->m_StopClock - this->m_StartClock;

  /** Fill m_ElapsedTimeSec. */
  this->m_ElapsedTimeSec = static_cast< std::size_t >( this->m_ElapsedTime );

  /** Fill m_ElapsedClockSec. */
#ifndef ELX_USE_CLOCK_GETTIME
  this->m_ElapsedClockSec = static_cast< double >( this->m_ElapsedClock ) / CLOCKS_PER_SEC;
#else
  this->m_ElapsedClockSec  = ( this->m_StopClockMonotonic.tv_sec - this->m_StartClockMonotonic.tv_sec );
  this->m_ElapsedClockSec += ( this->m_StopClockMonotonic.tv_nsec - this->m_StartClockMonotonic.tv_nsec ) / 1.0e9;
#endif

  /** Fill m_TimeDHMS. */
  const std::size_t secondsPerMinute = 60;
  const std::size_t secondsPerHour   = 60 * secondsPerMinute;
  const std::size_t secondsPerDay    = 24 * secondsPerHour;

  std::size_t elapsedSeconds = this->m_ElapsedTimeSec;
  this->m_ElapsedTimeDHMS[ 0 ] = elapsedSeconds / secondsPerDay;

  elapsedSeconds              %= secondsPerDay;
  this->m_ElapsedTimeDHMS[ 1 ] = elapsedSeconds / secondsPerHour;

  elapsedSeconds              %= secondsPerHour;
  this->m_ElapsedTimeDHMS[ 2 ] = elapsedSeconds / secondsPerMinute;

  elapsedSeconds              %= secondsPerMinute;
  this->m_ElapsedTimeDHMS[ 3 ] = elapsedSeconds;

  return 0;

} // end ElapsedClockAndTime()


/**
 * ******************** PrintStartTime **************************
 */

const std::string &
Timer::PrintStartTime( void )
{
  /** Convert time to string. */
  struct tm * sStartTime = localtime( &( this->m_StartTime ) );
  this->m_StartTimeString = asctime( sStartTime );
  this->m_StartTimeString.erase( this->m_StartTimeString.end() - 1 );

  /** Return a value. */
  return this->m_StartTimeString;

} // end PrintStartTime()


/**
 * ******************** PrintStopTime **************************
 */

const std::string &
Timer::PrintStopTime( void )
{
  /** Convert time to string. */
  struct tm * sStopTime = localtime( &( this->m_StopTime ) );
  this->m_StopTimeString = asctime( sStopTime );
  this->m_StopTimeString.erase( this->m_StopTimeString.end() - 1 );

  /** Return a value. */
  return this->m_StopTimeString;

} // end PrintStopTime()


/**
 * ***************** PrintElapsedTimeDHMS ***********************
 */

const std::string &
Timer::PrintElapsedTimeDHMS( void )
{
  /** Print m_ElapsedTime in Days, Hours, Minutes and Seconds. */
  std::ostringstream make_string( "" );
  if( this->m_ElapsedTimeDHMS[ 0 ] != 0 )
  {
    make_string << this->m_ElapsedTimeDHMS[ 0 ] << " Days, ";
  }
  if( this->m_ElapsedTimeDHMS[ 1 ] != 0 )
  {
    make_string << this->m_ElapsedTimeDHMS[ 1 ] << " Hours, ";
  }
  if( this->m_ElapsedTimeDHMS[ 2 ] != 0 )
  {
    make_string << this->m_ElapsedTimeDHMS[ 2 ] << " Minutes, ";
  }
  make_string << this->m_ElapsedTimeDHMS[ 3 ] << " Seconds";

  /** Return a value. */
  this->m_ElapsedTimeDHMSString = make_string.str();
  return this->m_ElapsedTimeDHMSString;

} // end PrintElapsedTimeDHMS()


/**
 * ***************** PrintElapsedTimeSec ************************
 */

const std::string &
Timer::PrintElapsedTimeSec( void )
{
  /** Print m_ElapsedTime in seconds. */
  std::ostringstream make_string( "" );
  make_string << this->m_ElapsedTimeSec;
  this->m_ElapsedTimeSecString = make_string.str();

  return this->m_ElapsedTimeSecString;

} // end PrintElapsedTimeSec()


/**
 * ******************* PrintElapsedClock ************************
 */

const std::string &
Timer::PrintElapsedClock( void )
{
  /** Print m_ElapsedClock. */
  std::ostringstream make_string( "" );
  make_string << this->m_ElapsedClock;
  this->m_ElapsedClockString = make_string.str();

  return this->m_ElapsedClockString;

} // end PrintElapsedClock()


/**
 * ******************* PrintElapsedClockSec ************************
 */

const std::string &
Timer::PrintElapsedClockSec( void )
{
  /** Print m_ElapsedClockSec. */
  std::ostringstream make_string( "" );
  make_string << this->m_ElapsedClockSec;
  this->m_ElapsedClockSecString = make_string.str();

  return this->m_ElapsedClockSecString;

} // end PrintElapsedClockSec()


} // end namespace tmr

#endif // end #ifndef __elxTimer_CXX_

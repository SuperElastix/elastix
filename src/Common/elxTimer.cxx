#ifndef __elxTimer_CXX_
#define __elxTimer_CXX_

#include "elxTimer.h"

namespace tmr
{
using namespace itk;

	
	/**
	 * ********************* Constructor ****************************
	 */
	
	Timer::Timer()
	{
		/** Initialize.*/
		m_ElapsedTimeDHMS.resize( 4 );
		m_StartTime = 0;
		m_StartClock = 0;
		m_StopTime = 0;
		m_StopClock = 0;
		
	} // end Constructor
	
	
	/**
	 * ********************** Destructor ****************************
	 */
	
	Timer::~Timer()
	{
	} // end Destructor
	
	
	/**
	 * ********************** StartTimer ****************************
	 */
	
	void Timer::StartTimer(void)
	{
		/** Get the current time.*/
		m_StartTime = time( '\0' );
		m_StartClock = clock();
		
	} // end StartTimer
	
	
	/**
	 * *********************** StopTimer ****************************
	 */
	
	int Timer::StopTimer(void)
	{
		/** Check if m_StartTime != 0.*/
		if ( m_StartTime == 0 ) return 1;
		
		/** Get the current time.*/
		m_StopTime = time( '\0' );
		m_StopClock = clock();
		
		/** Get the elapsed time.*/
		this->ElapsedClockAndTime();
		
		return 0;
		
	} // end StoptTimer
	
	
	/**
	 * *********************** ElapsedClockAndTime **************************
	 */
	
	int Timer::ElapsedClockAndTime(void)
	{
		/** Check if if m_StopTime != 0.*/
		if ( m_StopTime == 0 ) return 1;
		
		/** Calculate time difference = m_Elapsedtime.*/
		m_ElapsedTime = static_cast<time_t>( difftime( m_StopTime, m_StartTime ) );
		m_ElapsedClock = m_StopClock - m_StartClock;
		
		/** Fill m_ElapsedTimeSec.*/
		m_ElapsedTimeSec = static_cast<int>( m_ElapsedTime );
		
		/** Fill m_ElapsedClockSec.*/
		m_ElapsedClockSec = static_cast<double>( m_ElapsedClock ) / CLOCKS_PER_SEC;
		
		/** Fill m_TimeDHMS.*/
		struct tm *sElapsedTime = localtime( &m_ElapsedTime );
		m_ElapsedTimeDHMS[ 0 ] = sElapsedTime->tm_yday;
		m_ElapsedTimeDHMS[ 1 ] = sElapsedTime->tm_hour - 1;
		m_ElapsedTimeDHMS[ 2 ] = sElapsedTime->tm_min;
		m_ElapsedTimeDHMS[ 3 ] = sElapsedTime->tm_sec;
		
		return 0;

	} // end ElapsedClockAndTime
	
	
	/**
	 * ******************** PrintStartTime **************************
	 */
	
	const std::string & Timer::PrintStartTime( void )
	{
				
		/** Convert time to string.*/
		struct tm *sStartTime = localtime( &m_StartTime );

		m_StartTimeString =  asctime( sStartTime );

		m_StartTimeString.erase( m_StartTimeString.end() - 1 );
		
				
		return m_StartTimeString;

	} // end PrintStartTime
	
	
	/**
	 * ******************** PrintStopTime **************************
	 */
	
	const std::string & Timer::PrintStopTime( void )
	{
				
		/** Convert time to string.*/
		struct tm *sStopTime = localtime( &m_StopTime );

		m_StopTimeString = asctime( sStopTime );

		m_StopTimeString.erase( m_StopTimeString.end() - 1 );

						
		return m_StopTimeString;

	} // end PrintStopTime
		
	
	/**
	 * ***************** PrintElapsedTimeDHMS ***********************
	 */
	
	const std::string & Timer::PrintElapsedTimeDHMS( void )
	{
		
		/** Print m_ElapsedTime in Days, Hours, Minutes and Seconds.*/
		std::ostringstream make_string( "" );
		if ( m_ElapsedTimeDHMS[ 0 ] != 0 )
			make_string << m_ElapsedTimeDHMS[ 0 ] << " Days, ";
		if ( m_ElapsedTimeDHMS[ 1 ] != 0 )
			make_string << m_ElapsedTimeDHMS[ 1 ] << " Hours, ";
		if ( m_ElapsedTimeDHMS[ 2 ] != 0 )
			make_string << m_ElapsedTimeDHMS[ 2 ] << " Minutes, ";
		make_string << m_ElapsedTimeDHMS[ 3 ] << " Seconds";
		
		m_ElapsedTimeDHMSString = make_string.str();
		
		return m_ElapsedTimeDHMSString;

	} // end PrintElapsedTimeDHMS
	
	
	/**
	 * ***************** PrintElapsedTimeSec ************************
	 */
	
	const std::string & Timer::PrintElapsedTimeSec( void )
	{
		

		/** Print m_ElapsedTime in seconds.*/
		std::ostringstream make_string( "" );
		make_string << m_ElapsedTimeSec;

		m_ElapsedTimeSecString = make_string.str();
		
		return m_ElapsedTimeSecString;

	} // end PrintElapsedTimeSec
	
	
	/**
	 * ******************* PrintElapsedClock ************************
	 */
	
	const std::string & Timer::PrintElapsedClock( void )
	{
				
		/** Print m_ElapsedClock.*/
		std::ostringstream make_string( "" );
		make_string << m_ElapsedClock;

		m_ElapsedClockString = make_string.str();
		
		return m_ElapsedClockString;

	} // end PrintElapsedClock

	
	/**
	 * ******************* PrintElapsedClockSec ************************
	 */
	
	const std::string & Timer::PrintElapsedClockSec( void )
	{
		
		/** Print m_ElapsedClockSec.*/

		std::ostringstream make_string( "" );
		make_string << m_ElapsedClockSec;
		
		m_ElapsedClockSecString = make_string.str();
		
		return m_ElapsedClockSecString;

	} // end PrintElapsedClockSec
	

	

} // end namespace tmr

#endif // end #ifndef __elxTimer_CXX_


/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#include "elxTimer.h"
#include "vnl/vnl_math.h"

int
TestStartStop( void )
{
  tmr::Timer::Pointer pTmr = tmr::Timer::New();
  pTmr->StartTimer();

  if( pTmr->StopTimer() != 0 )
  {
    std::cerr << "StopTimer() failed.\n";
    return 1;
  }

  return 0;
} // end TestStartStop()


int
TestZeroTimeOutput( void )
{
  tmr::Timer::Pointer pTmr = tmr::Timer::New();

  pTmr->StartTimer();
  pTmr->StopTimer();

  if( pTmr->GetElapsedTimeSec() != 0 )
  {
    std::cerr << "GetElapsedTimeSec() != 0\n";
    return 1;
  }

  if( pTmr->PrintElapsedTimeSec() != "0" )
  {
    std::cerr << "PrintElapsedTimeSec() != 0\n";
    return 1;
  }

  if( pTmr->PrintElapsedTimeDHMS() != "0 Seconds" )
  {
    std::cerr << "PrintElapsedTimeDHMS() failed.\n";
    return 1;
  }

  return 0;

} // end TestZeroTimeOutput()


int
main( int argc, char * argv[] )
{
#ifndef NDEBUG
  const double N = 1e8;
#else
  const double N = 1e9;
#endif

  /** Time some dummy work. */
  tmr::Timer::Pointer pTmr = tmr::Timer::New();
  pTmr->StartTimer();
  double dummy = 0.0;
  for( double i = 0; i < N; i++ )
  {
    dummy += vcl_sqrt( static_cast< double >( i ) );
  }
  pTmr->StopTimer();

  /** Print unformatted. */
  std::cerr << "Start time          : " << pTmr->GetStartTime() << std::endl;
  std::cerr << "Stop time           : " << pTmr->GetStopTime() << std::endl;
  std::cerr << "Elapsed time        : " << pTmr->GetElapsedTime() << std::endl;
  std::cerr << "Elapsed time (Sec)  : " << pTmr->GetElapsedTimeSec() << std::endl;
  std::cerr << "Elapsed clock       : " << pTmr->GetElapsedClock() << std::endl;
  std::cerr << "Elapsed clock (Sec) : " << pTmr->GetElapsedClockSec() << std::endl;
  std::cerr << std::endl;

  /** Print formatted. */
  std::cerr << "Start time          : " << pTmr->PrintStartTime() << std::endl;
  std::cerr << "Stop time           : " << pTmr->PrintStopTime() << std::endl;
  std::cerr << "Elapsed time (DHMS) : " << pTmr->PrintElapsedTimeDHMS() << std::endl;
  std::cerr << "Elapsed time (Sec)  : " << pTmr->PrintElapsedTimeSec() << std::endl;
  std::cerr << "Elapsed clock       : " << pTmr->PrintElapsedClock() << std::endl;
  std::cerr << "Elapsed clock (Sec) : " << pTmr->PrintElapsedClockSec() << std::endl;

  /** Zero test. */
  return TestStartStop() || TestZeroTimeOutput();

} // end main()

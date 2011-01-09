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


int TestStartStop( void )
{
  tmr::Timer::Pointer pTmr = tmr::Timer::New();
  pTmr->StartTimer();

  if ( pTmr->StopTimer() != 0 )
  {
    std::cerr << "StopTimer() failed.\n";
    return 1;
  }

  return 0;
} // end TestStartStop()


int TestZeroTimeOutput( void )
{
  tmr::Timer::Pointer pTmr = tmr::Timer::New();

  pTmr->StartTimer();
  pTmr->StopTimer();

  if ( pTmr->GetElapsedTimeSec() != 0 )
  {
    std::cerr << "GetElapsedTimeSec() != 0\n";
    return 1;
  }

  if ( pTmr->PrintElapsedTimeSec() != "0" )
  {
    std::cerr << "PrintElapsedTimeSec() != 0\n";
    return 1;
  }

  if ( pTmr->PrintElapsedTimeDHMS() != "0 Seconds" )
  {
    std::cerr << "PrintElapsedTimeDHMS() failed.\n";
    return 1;
  }

  return 0;

} // end TestZeroTimeOutput()


int main( int argc, char *argv[] )
{
	int errorCode = TestStartStop() || TestZeroTimeOutput();
	
	std::cerr << "Elastix code was successfully used outside the Elastix tree!" << std::endl;

	return errorCode;
} // end main()


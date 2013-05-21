/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxTimer_H_
#define __elxTimer_H_

#include "itkObject.h"
#include "itkObjectFactory.h"
#include <ctime>
#include <sstream>

/**
 * *********************** clock() *********************************
 *
 * Return Value
 *
 * clock returns the number of clock ticks of elapsed processor time.
 * The returned value is the product of the amount of time that has
 * elapsed since the start of a process and the value of the CLOCKS_PER_SEC
 * constant. If the amount of elapsed time is unavailable, the function
 * returns -1, cast as a clock_t.
 *
 * Remarks
 *
 * The clock function tells how much processor time the calling process has
 * used. The time in seconds is approximated by dividing the clock return
 * value by the value of the CLOCKS_PER_SEC constant. In other words,
 * clock returns the number of processor timer ticks that have elapsed.
 * A timer tick is approximately equal to 1/CLOCKS_PER_SEC second. In
 * versions of Microsoft C before 6.0, the CLOCKS_PER_SEC constant
 * was called CLK_TCK.
 */

namespace tmr
{

/**
 * \class Timer
 * \brief A class to time the different parts of the registration.
 *
 * This class is a wrap around ctime.h. It is used to time the registration,
 * to get the time per iteration, or whatever.
 *
 * For precise timings we use clock() or clock_gettime().
 * On Windows clock_gettime() does not exist. clock() seems to give accurate
 * timings, also on multi-threaded systems.
 * For GCC / linux we use clock_gettime(), since clock() reports erroneous
 * results on linux on multi-threaded systems: it reports the elapsed time
 * multiplied by the number of threads that have been used.
 * Ugly #ifdefs are needed however, and elxCommon requires linking to the
 * library rt, but on linux only.
 *
 * \ingroup Timer
 */

class Timer : public itk::Object
{
public:
  /** Standard ITK-stuff.*/
  typedef Timer                            Self;
  typedef itk::Object                      Superclass;
  typedef itk::SmartPointer<Self>          Pointer;
  typedef itk::SmartPointer<const Self>    ConstPointer;

  /** Method for creation through the object factory.*/
  itkNewMacro( Self );

  /** Run-time type information (and related methods).*/
  itkTypeMacro( Timer, itk::Object );

  /** My typedef's.*/
  typedef std::vector<std::size_t>         TimeDHMSType;

  /** Member functions.*/
  void StartTimer( void );
  int StopTimer( void );
  int ElapsedClockAndTime( void );

  /** Formatted Output Functions
   * (return the time as a string, with comments)
   */
  const std::string & PrintStartTime( void );
  const std::string & PrintStopTime( void );
  const std::string & PrintElapsedTimeDHMS( void );
  const std::string & PrintElapsedTimeSec( void );
  const std::string & PrintElapsedClock( void );
  const std::string & PrintElapsedClockSec( void );

  /** Communication with outside world.*/
  itkGetConstMacro( StartTime, time_t );
  itkGetConstMacro( StopTime, time_t );
  itkGetConstMacro( ElapsedTime, double );
  //  itkGetConstMacro( ElapsedTimeDHMS, TimeDHMSType );
  itkGetConstMacro( ElapsedTimeSec, std::size_t );
  itkGetConstMacro( ElapsedClock, double );
  itkGetConstMacro( ElapsedClockSec, double );

protected:

  Timer();
  virtual ~Timer(){};

  /** Variables that store program arguments.*/
  time_t        m_StartTime;
  clock_t       m_StartClock;
  time_t        m_StopTime;
  clock_t       m_StopClock;
  double        m_ElapsedTime;
  clock_t       m_ElapsedClock;
  TimeDHMSType  m_ElapsedTimeDHMS;
  std::size_t   m_ElapsedTimeSec;
  double        m_ElapsedClockSec;

  /** GCC specific. We can use clock_gettime(). */
#if defined( __GNUC__ ) && !defined( __APPLE__ ) && !defined( _WIN32 )
#define ELX_USE_CLOCK_GETTIME
  struct timespec m_StartClockMonotonic;
  struct timespec m_StopClockMonotonic;
#endif

  /** Strings that serve as output of the Formatted Output Functions */
  std::string m_StartTimeString;
  std::string m_StopTimeString;
  std::string m_ElapsedTimeDHMSString;
  std::string m_ElapsedTimeSecString;
  std::string m_ElapsedClockString;
  std::string m_ElapsedClockSecString;

private:

  Timer( const Self& );           // purposely not implemented
  void operator=( const Self& );  // purposely not implemented

}; // end class Timer


} // end namespace tmr


#endif // end #ifndef __elxTimer_H_

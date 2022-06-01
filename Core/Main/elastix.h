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
#ifndef elastix_h
#define elastix_h

#include <cassert>
#include <ctime>
#include <cmath>   // For fmod.
#include <iomanip> // std::setprecision
#include <sstream>
#include <string>


/** ConvertSecondsToDHMS
 *
 */
inline std::string
ConvertSecondsToDHMS(const double totalSeconds, const unsigned int precision = 0)
{
  /** Define days, hours, minutes. */
  const std::size_t secondsPerMinute = 60;
  const std::size_t secondsPerHour = 60 * secondsPerMinute;
  const std::size_t secondsPerDay = 24 * secondsPerHour;

  /** Convert total seconds. */
  std::size_t       iSeconds = static_cast<std::size_t>(totalSeconds);
  const std::size_t days = iSeconds / secondsPerDay;

  iSeconds %= secondsPerDay;
  const std::size_t hours = iSeconds / secondsPerHour;

  iSeconds %= secondsPerHour;
  const std::size_t minutes = iSeconds / secondsPerMinute;

  // iSeconds %= secondsPerMinute;
  // const std::size_t seconds = iSeconds;
  const double dSeconds = std::fmod(totalSeconds, 60.0);

  /** Create a string in days, hours, minutes and seconds. */
  bool               nonzero = false;
  std::ostringstream make_string;
  if (days != 0)
  {
    make_string << days << "d";
    nonzero = true;
  }
  if (hours != 0 || nonzero)
  {
    make_string << hours << "h";
    nonzero = true;
  }
  if (minutes != 0 || nonzero)
  {
    make_string << minutes << "m";
  }
  make_string << std::showpoint << std::fixed << std::setprecision(precision);
  make_string << dSeconds << "s";

  /** Return a value. */
  return make_string.str();

} // end ConvertSecondsToDHMS()


/** Returns current date and time as a string. */
inline std::string
GetCurrentDateAndTime()
{
  // Obtain current time
  const std::time_t rawtime{ std::time(nullptr) };

  // Convert to local time
  // Note: std::localtime is not threadsafe!
  const std::tm * const localTimePtr{ std::localtime(&rawtime) };

  if (localTimePtr == nullptr)
  {
    assert(!"std::localtime should not return null!");
    return {};
  }

  // Make a copy of the internal object from std::localtime, to reduce the
  // risk of a race condition.
  const std::tm localTimeValue(*localTimePtr);

  constexpr std::size_t maxNumberOfChars{ 32 };
  char                  timeAsString[maxNumberOfChars]{};
  static_assert(maxNumberOfChars > sizeof("Thu Aug 23 14:55:02 2001"),
                "timeAsString should be large enough to hold a typical example date and time");

  if (std::strftime(timeAsString, maxNumberOfChars, "%c", &localTimeValue) == 0)
  {
    assert(!"std::strftime has failed!");
    return {};
  }

  return timeAsString;
} // end GetCurrentDateAndTime()


#endif

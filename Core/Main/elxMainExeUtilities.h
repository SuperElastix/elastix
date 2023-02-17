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
#ifndef elxMainExeUtilities_h
#define elxMainExeUtilities_h

#include "elxlog.h"

#include <exception>
#include <string>


namespace elastix
{
/** Reports the exception that is terminating the process to the user. */
void
ReportTerminatingException(const char * const executableName, const std::exception & stdException) noexcept;

/** Prints extended version information to standard output. */
std::string
GetExtendedVersionInformation(const char * const executableName, const char * const indentation = "");

/** Makes a string of all command-line arguments. */
std::string
MakeStringOfCommandLineArguments(const char * const * const arguments);

bool
ToLogLevel(const std::string & str, log::level & logLevel);

} // namespace elastix

#endif

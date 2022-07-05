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

// Its own header file:
#include "elxMainExeUtilities.h"

#include "xoutmain.h"
#include <itkMacro.h>

// Standard Library header files:
#include <cassert>
#include <exception>
#include <sstream>
#include <typeinfo>


void
elastix::ReportTerminatingException(const char * const executableName, const std::exception & stdException) noexcept
{
  try
  {
    std::ostringstream outputStringStream;

    outputStringStream << "ERROR: " << executableName
                       << " terminated because of the following exception:\nException type: "
                       << typeid(stdException).name();

    const auto itkExceptionObject = dynamic_cast<const itk::ExceptionObject *>(&stdException);

    if (itkExceptionObject)
    {
      // itk::ExceptionObject provides the most information by inserting the object directly, instead of calling what().
      outputStringStream << *itkExceptionObject;
    }
    else
    {
      outputStringStream << "\nWhat message: " << stdException.what() << '\n';
    }

    const std::string message = outputStringStream.str();

    // Insert the message into the standard error stream, as well as into xout, because xout might not yet be set up.
    std::cerr << message;
    xl::xout["error"] << message << std::flush;
  }
  catch (...)
  {
    // Enforce that this function itself will never throw any exception.
    assert(!"Unhandled exception!");
  }
}

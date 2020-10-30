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
/**
 * itkOpenCLMacro.h defines standard OpenCL macros, constants, and other
 * parameters. Macros are available for built-in types; for string classes;
 * vector arrays; object pointers; and debug, warning,
 * and error printout information.
 */

#ifndef itkOpenCLMacro_h
#define itkOpenCLMacro_h

#include "itkMacro.h"
#include "itkOpenCLExport.h"

#include <string>
#include <sstream>

//! This macro is used to print out debug message to the current message handler
//! in instance methods
//! itkOpenCLDebugMacro(<< "Debug message" << this->SomeVariable);
#define itkOpenCLDebugMacro(x) itkOpenCLDebugWithObjectMacro(this, x)

//! This macro is used to print out warning message to the current message
//! handler in instance methods
//! itkOpenCLWarningMacro(<< "Warning message" << this->SomeVariable);
#define itkOpenCLWarningMacro(x) itkOpenCLWarningWithObjectMacro(this, x)

//! This macro is used to print out error message to the current message handler
//! in instance methods
//! itkOpenCLErrorMacro(<< "Error message" << this->SomeVariable);
#define itkOpenCLErrorMacro(x) itkOpenCLErrorWithObjectMacro(this, x)

//! This macro is used to print out debug message to the current message handler
//! in main() function or generic functions
//! itkOpenCLDebugMacroGeneric(<< "Debug message generic" << SomeVariable);
#define itkOpenCLDebugMacroGeneric(x)                                                                                  \
  {                                                                                                                    \
    std::stringstream itkmsg;                                                                                          \
    itkmsg << __FILE__ << "(" << __LINE__ << "): itkOpenCL generic debug."                                             \
           << "\nDebug: in function: " << __FUNCTION__ << "\nDetails: " x << "\n\n";                                   \
    ::itk::OutputWindowDisplayDebugText(itkmsg.str().c_str());                                                         \
  }

//! This macro is used to print out warning message to the current message
//! handler in main() function or generic functions
//! itkOpenCLWarningMacroGeneric(<< "Warning message generic" << SomeVariable);
#define itkOpenCLWarningMacroGeneric(x)                                                                                \
  {                                                                                                                    \
    std::stringstream itkmsg;                                                                                          \
    itkmsg << __FILE__ << "(" << __LINE__ << "): itkOpenCL generic warning."                                           \
           << "\nWarning: in function: " << __FUNCTION__ << "\nDetails: " x << "\n\n";                                 \
    ::itk::OutputWindowDisplayWarningText(itkmsg.str().c_str());                                                       \
  }

//! This macro is used to print out error message to the current message handler
//! in main() function or generic functions
//! itkOpenCLErrorMacroGeneric(<< "Error message generic" << SomeVariable);
#define itkOpenCLErrorMacroGeneric(x)                                                                                  \
  {                                                                                                                    \
    std::stringstream itkmsg;                                                                                          \
    itkmsg << __FILE__ << "(" << __LINE__ << "): itkOpenCL generic error."                                             \
           << "\nError: in function: " << __FUNCTION__ << "\nDetails: " x << "\n\n";                                   \
    ::itk::OutputWindowDisplayErrorText(itkmsg.str().c_str());                                                         \
  }

/** This macro is used to print out debug statements. For example:
 * \code
 * itkOpenCLDebugWithObjectMacro(self, "Debug message" << variable);
 * \endcode
 * File and line information will be printed in Visual Studio format.
 * \sa itkOpenCLWarningWithObjectMacro(), itkOpenCLErrorWithObjectMacro() */
#define itkOpenCLDebugWithObjectMacro(self, x)                                                                         \
  {                                                                                                                    \
    std::stringstream itkmsg;                                                                                          \
    itkmsg << __FILE__ << "(" << __LINE__ << "): itkOpenCL debug."                                                     \
           << "\nDebug: in function: " << __FUNCTION__ << "; Name: " << self->GetNameOfClass() << " (" << self << ")"  \
           << "\nDetails: " x << "\n\n";                                                                               \
    ::itk::OutputWindowDisplayDebugText(itkmsg.str().c_str());                                                         \
  }

/** This macro is used to print out warnings. For example:
 * \code
 * itkOpenCLWarningWithObjectMacro(self, "Warning message" << variable);
 * \endcode
 * File and line information will be printed in Visual Studio format.
 * \sa itkOpenCLDebugWithObjectMacro(), itkOpenCLErrorWithObjectMacro() */
#define itkOpenCLWarningWithObjectMacro(self, x)                                                                       \
  {                                                                                                                    \
    std::stringstream itkmsg;                                                                                          \
    itkmsg << __FILE__ << "(" << __LINE__ << "): itkOpenCL warning."                                                   \
           << "\nWarning: in function: " << __FUNCTION__ << "; Name: " << self->GetNameOfClass() << " (" << self       \
           << ")"                                                                                                      \
           << "\nDetails: " x << "\n\n";                                                                               \
    ::itk::OutputWindowDisplayWarningText(itkmsg.str().c_str());                                                       \
  }

/** This macro is used to print out errors. For example:
 * \code
 * itkOpenCLErrorWithObjectMacro(self, "Error message" << variable);
 * \endcode
 * File and line information will be printed in Visual Studio format.
 * \sa itkOpenCLWarningWithObjectMacro(), itkOpenCLDebugWithObjectMacro() */
#define itkOpenCLErrorWithObjectMacro(self, x)                                                                         \
  {                                                                                                                    \
    std::stringstream itkmsg;                                                                                          \
    itkmsg << __FILE__ << "(" << __LINE__ << "): itkOpenCL error."                                                     \
           << "\nError: in function: " << __FUNCTION__ << "; Name: " << self->GetNameOfClass() << " (" << self << ")"  \
           << "\nDetails: " x << "\n\n";                                                                               \
    ::itk::OutputWindowDisplayErrorText(itkmsg.str().c_str());                                                         \
  }

namespace itk
{
/** \class OpenCLCompileError
 * Exception thrown when OpenCL program failed to build.
 */
class ITKOpenCL_EXPORT OpenCLCompileError : public ExceptionObject
{
public:
  /** Default constructor. Needed to ensure the exception object can be copied. */
  OpenCLCompileError()
    : ExceptionObject()
  {}

  /** Constructor. Needed to ensure the exception object can be copied. */
  OpenCLCompileError(const char * file, unsigned int lineNumber)
    : ExceptionObject(file, lineNumber)
  {}

  /** Constructor. Needed to ensure the exception object can be copied. */
  OpenCLCompileError(const std::string & file, unsigned int lineNumber)
    : ExceptionObject(file, lineNumber)
  {}

  /** Virtual destructor needed for subclasses. */
  ~OpenCLCompileError() override = default;

  const char *
  GetNameOfClass() const override
  {
    return "OpenCLCompileError";
  }
};

} // namespace itk
#endif /* itkOpenCLMacro_h */

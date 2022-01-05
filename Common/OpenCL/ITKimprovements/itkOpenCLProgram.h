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
#ifndef itkOpenCLProgram_h
#define itkOpenCLProgram_h

#include "itkOpenCLDevice.h"
#include "itkOpenCLKernel.h"

#include <string>

namespace itk
{
/**
 * \class OpenCLProgram
 * \brief The OpenCLProgram class represents an OpenCL program object.
 * \ingroup OpenCL
 */

// Forward declaration
class OpenCLContext;

class ITKOpenCL_EXPORT OpenCLProgram
{
public:
  /** Standard class typedefs. */
  using Self = OpenCLProgram;

  /** Constructs a null OpenCL program object. */
  OpenCLProgram();

  /** Constructs an OpenCL program object from the native identifier \a id,
   * and associates it with \a context and debug file \a fileName.
   * This class will take over ownership of \a id and will release it in the destructor. */
  OpenCLProgram(OpenCLContext * context, cl_program id, const std::string & fileName = std::string());

  /** Constructs a copy of \a other. */
  OpenCLProgram(const OpenCLProgram & other);

  /** Releases this OpenCL program object.
   * If this is the last reference to the program, it will be destroyed. */
  ~OpenCLProgram();

  /** Assigns \a other to this object. */
  OpenCLProgram &
  operator=(const OpenCLProgram & other);

  /** Returns null if this OpenCL program object is null. */
  bool
  IsNull() const
  {
    return this->m_Id == 0;
  }

  /** Returns the OpenCL context that this program was created within. */
  OpenCLContext *
  GetContext() const
  {
    return this->m_Context;
  }

  /** Returns the native OpenCL identifier for this program. */
  cl_program
  GetProgramId() const
  {
    return this->m_Id;
  }

  /** Returns the debug filename that this program was created within. */
  std::string
  GetFileName() const
  {
    return this->m_FileName;
  }

  /** Builds this program from the sources and binaries that were supplied,
   * with extra build compiler options specified by \a extraBuildOptions.
   * The main compiler options are provided during CMake configuration
   * (see group OPENCL in CMake).
   * Returns true if the program was built; false otherwise.
   * \sa GetLog(), CreateKernel() */
  bool
  Build(const std::string & extraBuildOptions = std::string());

  /** \overload
   * Builds this program from the sources and binaries that were supplied,
   * with extra build compiler options specified by \a extraBuildOptions.
   * The main compiler options are provided during CMake configuration
   * (see group OPENCL in CMake).
   * If \a devices is not empty, the program will only be built for devices
   * in the specified list. Otherwise the program will be built for all
   * devices on the program's context.
   * Returns true if the program was built; false otherwise.
   * \sa GetLog(), CreateKernel() */
  bool
  Build(const std::list<OpenCLDevice> & devices, const std::string & extraBuildOptions = std::string());

  /** Returns the error GetLog that resulted from the last build().
   * \sa Build() */
  std::string
  GetLog() const;

  /** Returns the list of devices that this program is associated with.
   * \sa GetBinaries() */
  std::list<OpenCLDevice>
  GetDevices() const;

  /** Creates a kernel for the entry point associated with \a name
   * in this program.
   * \sa Build() */
  OpenCLKernel
  CreateKernel(const std::string & name) const;

  /** Creates a list of kernels for all of the entry points in this program. */
  std::list<OpenCLKernel>
  CreateKernels() const;

private:
  OpenCLContext * m_Context;
  cl_program      m_Id;
  std::string     m_FileName;
};

/** Operator ==
 * Returns true if \a lhs OpenCL program is the same as \a rhs, false otherwise.
 * \sa operator!= */
bool ITKOpenCL_EXPORT
     operator==(const OpenCLProgram & lhs, const OpenCLProgram & rhs);

/** Operator !=
 * Returns true if \a lhs OpenCL program is not the same as \a rhs, false otherwise.
 * \sa operator== */
bool ITKOpenCL_EXPORT
     operator!=(const OpenCLProgram & lhs, const OpenCLProgram & rhs);

/** Stream out operator for OpenCLProgram */
template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const OpenCLProgram & program)
{
  if (program.IsNull())
  {
    strm << "OpenCLProgram(null)";
    return strm;
  }

  const char indent = ' ';

  strm << "OpenCLProgram\n" << indent << "Id: " << program.GetProgramId() << std::endl;

  return strm;
}


} // end namespace itk

#endif /* itkOpenCLProgram_h */

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
#ifndef itkOpenCLLogger_h
#define itkOpenCLLogger_h

#include "itkLoggerBase.h"
#include "itkStdStreamLogOutput.h"

namespace itk
{
/** \class OpenCLLogger
 * \brief Used for logging OpenCL compiler errors during a run.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 */
class OpenCLLogger : public LoggerBase
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(OpenCLLogger);

  using Self = OpenCLLogger;
  using Superclass = LoggerBase;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(OpenCLLogger, Object);

  /** This is a singleton pattern New. There will only be ONE
   * reference to a OpenCLLogger object per process. Clients that
   * call this must call Delete on the object so that the reference
   * counting will work. The single instance will be unreferenced when
   * the program exits. */
  static Pointer
  New();

  /** Return the singleton instance with no reference counting. */
  static Pointer
  GetInstance();

  /** Set log filename prefix. */
  void
  SetLogFileNamePrefix(const std::string & prefix);

  /** Get the log filename. */
  std::string
  GetLogFileName() const
  {
    return this->m_FileName;
  }

  /** Set output directory for logger. */
  itkSetStringMacro(OutputDirectory);

  /** Returns true if the underlying OpenCL logger has been
   * created, false otherwise. */
  bool
  IsCreated() const;

  /** Overloaded. */
  void
  Write(PriorityLevelEnum level, std::string const & content) override;

protected:
  /** Constructor */
  OpenCLLogger();

  /** Destructor */
  ~OpenCLLogger() override;

  /** Initialize */
  void
  Initialize();

private:
  static Pointer m_Instance;

  std::string m_FileName;
  std::string m_OutputDirectory;

  itk::StdStreamLogOutput::Pointer m_Stream;
  std::ostream *                   m_FileStream;
  bool                             m_Created;
};

} // end namespace itk

#endif // end #ifndef itkOpenCLLogger_h

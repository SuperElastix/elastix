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
#ifndef itkOpenCLProfilingTimeProbe_h
#define itkOpenCLProfilingTimeProbe_h

#include "itkOpenCLExport.h"
#include "itkTimeProbe.h"

#include <string>

namespace itk
{
/** \class OpenCLProfilingTimeProbe
 * \brief Computes the time passed between two points in code.
 *
 * \ingroup OpenCL
 * \sa TimeProbe
 */
class ITKOpenCL_EXPORT OpenCLProfilingTimeProbe
{
public:
  /** Constructor */
  OpenCLProfilingTimeProbe(const std::string & message);

  /** Destructor */
  ~OpenCLProfilingTimeProbe();

private:
  TimeProbe   m_Timer;
  std::string m_ProfilingMessage;
};

} // end namespace itk

#endif // itkOpenCLProfilingTimeProbe_h

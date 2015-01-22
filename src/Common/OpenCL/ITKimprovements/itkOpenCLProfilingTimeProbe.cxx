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
#include "itkOpenCLProfilingTimeProbe.h"

namespace itk
{
OpenCLProfilingTimeProbe::OpenCLProfilingTimeProbe( const std::string & message ) :
  m_ProfilingMessage( message )
{
  this->m_Timer.Start();
}


//------------------------------------------------------------------------------
OpenCLProfilingTimeProbe::~OpenCLProfilingTimeProbe()
{
  this->m_Timer.Stop();
  std::cout << this->m_ProfilingMessage << " took "
            << this->m_Timer.GetMean() << " seconds." << std::endl;
}


} // end namespace itk

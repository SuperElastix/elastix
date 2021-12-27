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

#include "xoutcell.h"

namespace xoutlibrary
{

/**
 * ************************ Constructor *************************
 */

xoutcell::xoutcell()
{
  this->AddTargetCell("InternalBuffer", &(this->m_InternalBuffer));

} // end Constructor


/**
 * ******************** WriteBufferedData ***********************
 *
 * The buffered data is sent to the outputs.
 */

void
xoutcell::WriteBufferedData()
{
  const std::string strbuf = this->m_InternalBuffer.str();

  /** Send the string to the outputs */
  for (const auto & output : this->m_COutputs)
  {
    *(output.second) << strbuf << std::flush;
  }

  /** Send the string to the outputs */
  for (const auto & output : this->m_XOutputs)
  {
    *(output.second) << strbuf;
    output.second->WriteBufferedData();
  }

  /** Empty the internal buffer */
  this->m_InternalBuffer.str(std::string(""));

} // end WriteBufferedData


} // end namespace xoutlibrary

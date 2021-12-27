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

#include "elxBaseComponent.h"

#include <sstream> // For ostringstream.

namespace
{
bool
IsElastixLibrary(const bool initialValue = true)
{
  // By default, assume that this is the elastix library (not the elastix executable).

  // Note that the initialization of this static variable is thread-safe,
  // as supported by C++11 "magic statics".
  static const bool isElastixLibrary{ initialValue };

  return isElastixLibrary;
}
} // namespace

namespace elastix
{

/**
 * ****************** elxGetClassName ****************************
 */

const char *
BaseComponent::elxGetClassName() const
{
  return "BaseComponent";
} // end elxGetClassName()


/**
 * ****************** SetComponentLabel ****************************
 */

void
BaseComponent::SetComponentLabel(const char * label, unsigned int idx)
{
  std::ostringstream makestring;
  makestring << label << idx;
  this->m_ComponentLabel = makestring.str();
} // end SetComponentLabel()


/**
 * ****************** GetComponentLabel ****************************
 */

const char *
BaseComponent::GetComponentLabel() const
{
  return this->m_ComponentLabel.c_str();
} // end GetComponentLabel()


bool
BaseComponent::IsElastixLibrary()
{
  return ::IsElastixLibrary();
}

void
BaseComponent::InitializeElastixExecutable()
{
  ::IsElastixLibrary(false);
}

} // end namespace elastix

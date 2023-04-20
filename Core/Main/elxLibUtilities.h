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
#ifndef elxLibUtilities_h
#define elxLibUtilities_h

#include <map>
#include <string>
#include <vector>


namespace elastix::LibUtilities
{
using ParameterValuesType = std::vector<std::string>;
using ParameterMapType = std::map<std::string, ParameterValuesType>;


/** Sets the specified parameter value. Warns when it overrides existing parameter values. */
void
SetParameterValueAndWarnOnOverride(ParameterMapType &  parameterMap,
                                   const std::string & parameterName,
                                   const std::string & parameterValue);

} // namespace elastix::LibUtilities

#endif

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
#include "elxLibUtilities.h"

#include "elxlog.h"

namespace elastix
{
void
LibUtilities::SetParameterValueAndWarnOnOverride(ParameterMapType &  parameterMap,
                                                 const std::string & parameterName,
                                                 const std::string & parameterValue)
{
  if (const auto found = parameterMap.find(parameterName); found == parameterMap.end())
  {
    parameterMap[parameterName] = { parameterValue };
  }
  else
  {
    if (found->second.size() != 1 || found->second.front() != parameterValue)
    {
      found->second = { parameterValue };
      log::warn("WARNING: The values of parameter \"" + parameterName +
                "\" are automatically overridden!\n  The value \"" + parameterValue + "\" is used instead.");
    }
  }
}


std::string
LibUtilities::RetrievePixelTypeParameterValue(const ParameterMapType & parameterMap, const std::string & parameterName)
{
  if (const auto found = parameterMap.find(parameterName); found != parameterMap.end())
  {
    if (const auto & second = found->second; !second.empty())
    {
      if (const auto & front = second.front(); !front.empty())
      {
        return front;
      }
    }
  }
  return "float";
}


} // namespace elastix

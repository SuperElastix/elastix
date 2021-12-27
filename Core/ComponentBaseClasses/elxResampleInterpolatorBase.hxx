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

#ifndef elxResampleInterpolatorBase_hxx
#define elxResampleInterpolatorBase_hxx

#include "elxResampleInterpolatorBase.h"
#include "elxConversion.h"

namespace elastix
{

/**
 * ******************* ReadFromFile *****************************
 */

template <class TElastix>
void
ResampleInterpolatorBase<TElastix>::ReadFromFile()
{
  // nothing, but must be here

} // end ReadFromFile


/**
 * ******************* WriteToFile ******************************
 */

template <class TElastix>
void
ResampleInterpolatorBase<TElastix>::WriteToFile(xl::xoutsimple & transformationParameterInfo) const
{
  ParameterMapType parameterMap;
  this->CreateTransformParametersMap(parameterMap);

  /** Write ResampleInterpolator specific things. */
  transformationParameterInfo << ("\n// ResampleInterpolator specific\n" +
                                  Conversion::ParameterMapToString(parameterMap));

} // end WriteToFile()


/**
 * ******************* CreateTransformParametersMap ****************
 */

template <class TElastix>
void
ResampleInterpolatorBase<TElastix>::CreateTransformParametersMap(ParameterMapType & parameterMap) const
{
  /** Store the name of this transform. */
  parameterMap["ResampleInterpolator"] = { this->elxGetClassName() };

  // Derived classes may add some extra parameters
  for (auto & keyAndValue : this->CreateDerivedTransformParametersMap())
  {
    const auto & key = keyAndValue.first;
    assert(parameterMap.count(key) == 0);
    parameterMap[key] = std::move(keyAndValue.second);
  }

} // end CreateTransformParametersMap()


} // end namespace elastix

#endif // end #ifndef elxResampleInterpolatorBase_hxx

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

#ifndef elxRandomSamplerSparseMask_hxx
#define elxRandomSamplerSparseMask_hxx

#include "elxRandomSamplerSparseMask.h"

namespace elastix
{

/**
 * ******************* BeforeEachResolution ******************
 */

template <class TElastix>
void
RandomSamplerSparseMask<TElastix>::BeforeEachResolution()
{
  const unsigned int level = (this->m_Registration->GetAsITKBaseType())->GetCurrentLevel();

  /** Set the NumberOfSpatialSamples. */
  unsigned long numberOfSpatialSamples = 5000;
  this->GetConfiguration()->ReadParameter(
    numberOfSpatialSamples, "NumberOfSpatialSamples", this->GetComponentLabel(), level, 0);

  this->SetNumberOfSamples(numberOfSpatialSamples);

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef elxRandomSamplerSparseMask_hxx

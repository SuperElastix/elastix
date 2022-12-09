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
#ifndef elxFixedGenericPyramid_hxx
#define elxFixedGenericPyramid_hxx

#include "elxFixedGenericPyramid.h"
#include "../../elxGenericPyramidHelper.h"

namespace elastix
{

/**
 * ******************* SetFixedSchedule ***********************
 */

template <class TElastix>
void
FixedGenericPyramid<TElastix>::SetFixedSchedule()
{
  GenericPyramidHelper::SetSchedule(*this);

} // end SetFixedSchedule()


/**
 * ******************* BeforeEachResolution ***********************
 */

template <class TElastix>
void
FixedGenericPyramid<TElastix>::BeforeEachResolution()
{
  /** What is the current resolution level? */
  const unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** We let the pyramid filter know that we are in a next level.
   * Depending on a flag only at this point the output of the current level is computed,
   * or it was computed for all levels at once at initialization.
   */
  this->SetCurrentLevel(level);

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef elxFixedGenericPyramid_hxx

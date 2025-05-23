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
#ifndef elxDisplacementMagnitudePenalty_hxx
#define elxDisplacementMagnitudePenalty_hxx

#include "elxDisplacementMagnitudePenalty.h"
#include "itkTimeProbe.h"

namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template <typename TElastix>
void
DisplacementMagnitudePenalty<TElastix>::Initialize()
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  log::info(std::ostringstream{} << "Initialization of DisplacementMagnitude metric took: "
                                 << static_cast<std::int64_t>(timer.GetMean() * 1000) << " ms.");

} // end Initialize()


} // end namespace elastix

#endif // end #ifndef elxDisplacementMagnitudePenalty_hxx

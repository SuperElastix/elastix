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

#ifndef elxImageSamplerBase_hxx
#define elxImageSamplerBase_hxx

#include "elxImageSamplerBase.h"
#include "elxDeref.h"

namespace elastix
{

/**
 * ******************* BeforeRegistrationBase ******************
 */

template <class TElastix>
void
ImageSamplerBase<TElastix>::BeforeRegistrationBase()
{
  const Configuration & configuration = Deref(Superclass::GetConfiguration());
  ITKBaseType &         sampler = GetSelf();
  sampler.SetUseMultiThread(configuration.RetrieveParameterValue(true, "UseMultiThreadingForSamplers", 0, false));
}

/**
 * ******************* BeforeEachResolutionBase ******************
 */

template <class TElastix>
void
ImageSamplerBase<TElastix>::BeforeEachResolutionBase()
{
  /** Get the current resolution level. */
  unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  const Configuration & configuration = Deref(Superclass::GetConfiguration());

  /** Check if NewSamplesEveryIteration is possible with the selected ImageSampler.
   * The "" argument means that no prefix is supplied.
   */
  bool newSamples = false;
  configuration.ReadParameter(newSamples, "NewSamplesEveryIteration", "", level, 0, true);

  if (newSamples)
  {
    bool ret = this->GetAsITKBaseType()->SelectingNewSamplesOnUpdateSupported();
    if (!ret)
    {
      log::warn(std::ostringstream{} << "WARNING: You want to select new samples every iteration,\n"
                                     << "but the selected ImageSampler is not suited for that.");
    }
  }
} // end BeforeEachResolutionBase()


} // end namespace elastix

#endif //#ifndef elxImageSamplerBase_hxx

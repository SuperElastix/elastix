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
#ifndef elxAdvancedMeanSquaresMetric_hxx
#define elxAdvancedMeanSquaresMetric_hxx

#include "elxAdvancedMeanSquaresMetric.h"
#include "itkTimeProbe.h"
#include <itkDeref.h>

namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template <typename TElastix>
void
AdvancedMeanSquaresMetric<TElastix>::Initialize()
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  log::info(std::ostringstream{} << "Initialization of AdvancedMeanSquares metric took: "
                                 << static_cast<std::int64_t>(timer.GetMean() * 1000) << " ms.");

} // end Initialize()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <typename TElastix>
void
AdvancedMeanSquaresMetric<TElastix>::BeforeEachResolution()
{
  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());

  /** Get the current resolution level. */
  unsigned int level = (this->m_Registration->GetAsITKBaseType())->GetCurrentLevel();

  /** Get and set the normalization. */
  bool useNormalization = false;
  configuration.ReadParameter(useNormalization, "UseNormalization", BaseComponent::GetComponentLabel(), level, 0);
  this->SetUseNormalization(useNormalization);

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef elxAdvancedMeanSquaresMetric_hxx

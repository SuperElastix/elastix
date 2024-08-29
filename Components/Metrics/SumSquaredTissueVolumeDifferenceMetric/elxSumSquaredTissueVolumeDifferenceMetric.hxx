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
#ifndef elxSumSquaredTissueVolumeDifferenceMetric_hxx
#define elxSumSquaredTissueVolumeDifferenceMetric_hxx

#include "elxSumSquaredTissueVolumeDifferenceMetric.h"
#include <itkDeref.h>


namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template <typename TElastix>
void
SumSquaredTissueVolumeDifferenceMetric<TElastix>::Initialize()
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  log::info(std::ostringstream{} << "Initialization of SumSquaredTissueVolumeDifference metric took: "
                                 << static_cast<long>(timer.GetMean() * 1000) << " ms.");

} // end Initialize()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <typename TElastix>
void
SumSquaredTissueVolumeDifferenceMetric<TElastix>::BeforeEachResolution()
{
  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());
  const std::string     componentLabel = BaseComponent::GetComponentLabel();

  /** Get the current resolution level. */
  unsigned int level = (this->m_Registration->GetAsITKBaseType())->GetCurrentLevel();

  /** Get and set the AirValue. */
  float AirValue = -1000.0;
  configuration.ReadParameter(AirValue, "AirValue", componentLabel, level, 0);
  this->SetAirValue(AirValue);

  /** Get and set the TissueValue. */
  float TissueValue = 55.0;
  configuration.ReadParameter(TissueValue, "TissueValue", componentLabel, level, 0);
  this->SetTissueValue(TissueValue);

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef elxSumSquaredTissueVolumeDifferenceMetric_hxx

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
#ifndef elxAdvancedKappaStatisticMetric_hxx
#define elxAdvancedKappaStatisticMetric_hxx

#include "elxAdvancedKappaStatisticMetric.h"

#include "itkTimeProbe.h"

namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template <class TElastix>
void
AdvancedKappaStatisticMetric<TElastix>::Initialize()
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  elxout << "Initialization of AdvancedKappaStatistic metric took: " << static_cast<long>(timer.GetMean() * 1000)
         << " ms." << std::endl;

} // end Initialize()


/**
 * ***************** BeforeRegistration ***********************
 */

template <class TElastix>
void
AdvancedKappaStatisticMetric<TElastix>::BeforeRegistration()
{
  /** Get and set taking the complement. */
  bool useComplement = true;
  this->GetConfiguration()->ReadParameter(useComplement, "UseComplement", this->GetComponentLabel(), 0, -1);
  this->SetComplement(useComplement);

  /** Get and set the use of the foreground value:
   * true) compare with a foreground value
   * false) compare if larger than zero
   */
  bool useForegroundValue = true;
  this->GetConfiguration()->ReadParameter(useForegroundValue, "UseForegroundValue", this->GetComponentLabel(), 0, -1);
  this->SetUseForegroundValue(useForegroundValue);

  /** Get and set the foreground value. */
  double foreground = 1.0;
  this->GetConfiguration()->ReadParameter(foreground, "ForegroundValue", this->GetComponentLabel(), 0, -1);
  this->SetForegroundValue(foreground);

} // end BeforeRegistration()


} // end namespace elastix

#endif // end #ifndef elxAdvancedKappaStatisticMetric_hxx
